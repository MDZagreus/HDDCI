# -*- coding: utf-8 -*-
"""
Модуль обработки данных и анализа временных рядов.
Логика перенесена из analysis.ipynb.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # для работы без GUI (Streamlit)
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf


def newey_west_se(data, max_lags=None):
    """Стандартная ошибка с Newey-West корректировкой."""
    n = len(data)
    mean_data = np.mean(data)
    if max_lags is None:
        max_lags = int(4 * (n / 100) ** (2/9))
    max_lags = min(max_lags, n - 1)
    gamma_0 = np.var(data, ddof=0)
    hac_variance = gamma_0
    for lag in range(1, max_lags + 1):
        weight = 1 - lag / (max_lags + 1)
        autocov = np.mean((data[lag:] - mean_data) * (data[:-lag] - mean_data))
        hac_variance += 2 * weight * autocov
    return np.sqrt(hac_variance / n), max_lags


def diebold_mariano_test(forecast1, forecast2, actual, h=1, power=2):
    """Тест Диболда-Мариано для сравнения точности двух прогнозов."""
    forecast1 = np.array(forecast1)
    forecast2 = np.array(forecast2)
    actual = np.array(actual)
    loss1 = np.abs(forecast1 - actual) ** power
    loss2 = np.abs(forecast2 - actual) ** power
    d = loss1 - loss2
    d_mean = np.mean(d)
    n = len(d)
    gamma_0 = np.var(d, ddof=0)
    max_lags = min(int(4 * (n / 100) ** (2/9)), n - 1, h - 1)
    hac_variance = gamma_0
    for lag in range(1, max_lags + 1):
        weight = 1 - lag / (max_lags + 1)
        if lag < n:
            autocov = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
            hac_variance += 2 * weight * autocov
    hac_se = np.sqrt(hac_variance / n)
    dm_stat = d_mean / hac_se if hac_se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat))) if hac_se > 0 else 1.0
    return dm_stat, p_value, d_mean, hac_se


def process(df, target_platform='tta_android', target_product='avia',
            test_period_start=None, test_period_end=None):
    """
    Обрабатывает DataFrame и выполняет полный анализ.

    Возвращает dict с ключами:
    - info: текстовая информация о данных
    - metrics: метрики модели
    - feature_importance: DataFrame важности признаков
    - time_series_df: DataFrame с датами, фактом и прогнозом
    - residuals_info: информация об остатках
    - hac_results: результаты t-теста с HAC
    - dm_results: результаты теста Диболда-Мариано
    - summary: итоговая сводка
    - figures: список matplotlib.Figure для отображения
    """
    D_result_d = df.copy()

    # Поддержка колонки order_date (парсим даты при необходимости)
    if 'order_date' in D_result_d.columns:
        if not pd.api.types.is_datetime64_any_dtype(D_result_d['order_date']):
            D_result_d['order_date'] = pd.to_datetime(D_result_d['order_date'])

    D_result = D_result_d.drop(columns=['order_date'], errors='ignore') if 'order_date' in D_result_d.columns else D_result_d.copy()

    target_column = None
    for col in D_result.columns:
        if (target_platform in col and target_product in col and col.endswith('_revenue')):
            target_column = col
            break

    if target_column is None:
        rev_cols = [c for c in D_result.columns if 'revenue' in c]
        raise ValueError(
            f"Целевая колонка для platform='{target_platform}' и product='{target_product}' не найдена. "
            f"Доступные колонки с revenue: {rev_cols}"
        )

    train_data = D_result[D_result['test_period'] == 0].copy()
    test_data = D_result[D_result['test_period'] == 1].copy()

    feature_columns = [col for col in D_result.columns
                       if col not in ['test_period', target_column]]
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    # Обучение модели
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Восстановление дат для визуализации
    has_order_date = 'order_date' in D_result_d.columns
    if has_order_date:
        D_result_with_dates = D_result_d.dropna().reset_index(drop=True)
        test_data_with_dates = D_result_with_dates[D_result_with_dates['test_period'] == 1].copy()
        feature_columns_dates = [c for c in D_result_with_dates.columns
                                if c not in ['test_period', target_column, 'order_date']]
        X_test_with_dates = test_data_with_dates[feature_columns_dates]
        y_test_pred_with_dates = model.predict(X_test_with_dates)
        test_predictions_with_dates = pd.DataFrame({
            'date': test_data_with_dates['order_date'].values,
            'actual': test_data_with_dates[target_column].values,
            'predicted': y_test_pred_with_dates
        }).sort_values('date').reset_index(drop=True)
        dates = test_predictions_with_dates['date']
    else:
        test_predictions_with_dates = pd.DataFrame({
            'date': test_data.index.astype(str),
            'actual': y_test.values,
            'predicted': y_test_pred
        })
        dates = pd.Series(test_predictions_with_dates['date'])

    actual_values = test_predictions_with_dates['actual'].values
    predicted_values = test_predictions_with_dates['predicted'].values
    residuals_test = actual_values - predicted_values
    n_test = len(residuals_test)

    # Тест Люнга-Бокса
    max_lag = min(10, n_test // 4) if n_test // 4 > 0 else 0
    has_autocorr = False
    lb_stat, lb_pvalue = None, None
    if max_lag > 0:
        try:
            lb_result = acorr_ljungbox(residuals_test, lags=max_lag, return_df=True)
            lb_stat = float(lb_result['lb_stat'].iloc[-1])
            lb_pvalue = float(lb_result['lb_pvalue'].iloc[-1])
        except Exception:
            lb_stat_arr, lb_pvalue_arr = acorr_ljungbox(residuals_test, lags=max_lag, return_df=False)
            lb_stat = lb_stat_arr[-1] if hasattr(lb_stat_arr, '__getitem__') else lb_stat_arr
            lb_pvalue = lb_pvalue_arr[-1] if hasattr(lb_pvalue_arr, '__getitem__') else lb_pvalue_arr
        has_autocorr = lb_pvalue < 0.05

    # HAC t-тест
    hac_se, n_lags_used = newey_west_se(residuals_test)
    mean_residuals = float(np.mean(residuals_test))
    t_stat_hac = mean_residuals / hac_se
    p_value_hac = float(2 * (1 - stats.t.cdf(abs(t_stat_hac), n_test - 1)))
    t_critical = stats.t.ppf(0.975, n_test - 1)
    ci_mean_hac = (mean_residuals - t_critical * hac_se, mean_residuals + t_critical * hac_se)
    conclusion = "ЗНАЧИМОЕ" if p_value_hac < 0.05 else "НЕЗНАЧИМОЕ"

    # Тест Диболда-Мариано
    dm_stat, dm_pvalue, d_mean, dm_se = diebold_mariano_test(
        predicted_values, actual_values, actual_values, h=1, power=2
    )
    dm_conclusion = "ЗНАЧИМОЕ" if dm_pvalue < 0.05 else "НЕЗНАЧИМОЕ"

    info_lines = [
        f"Целевая колонка: {target_column}",
        f"Колонки в D_result: {list(D_result.columns)}",
        f"Размер D_result: {D_result.shape}",
        f"Распределение test_period:\n{D_result['test_period'].value_counts()}",
        f"Размер train данных: {train_data.shape}",
        f"Размер test данных: {test_data.shape}",
        f"Количество признаков: {len(feature_columns)}",
        f"Пропуски в train: {X_train.isnull().sum().sum()}",
        f"Пропуски в test: {X_test.isnull().sum().sum()}",
    ]

    residuals_info = {
        'n_test': n_test, 'mean': float(np.mean(residuals_test)),
        'std': float(np.std(residuals_test)), 'median': float(np.median(residuals_test)),
        'has_autocorr': has_autocorr, 'lb_stat': lb_stat, 'lb_pvalue': lb_pvalue, 'max_lag': max_lag,
    }

    hac_results = {
        'mean_residuals': mean_residuals, 'hac_se': hac_se, 'n_lags_used': n_lags_used,
        't_stat_hac': t_stat_hac, 'p_value_hac': p_value_hac, 'ci_mean_hac': ci_mean_hac,
        'conclusion': conclusion,
    }

    dm_results = {
        'dm_stat': dm_stat, 'dm_pvalue': dm_pvalue, 'd_mean': d_mean, 'dm_se': dm_se,
        'dm_conclusion': dm_conclusion,
    }

    # --- Построение графиков ---
    figures = []
    bg_dark = '#1a1d24'
    text_light = '#eaeaea'
    line_actual = '#00D4FF'
    line_pred = '#FF6B6B'
    line_ci = '#9B59B6'
    line_mean = '#F39C12'
    grid_color = '#4a4a5a'

    def _dark_axes(ax):
        ax.set_facecolor(bg_dark)
        ax.tick_params(colors=text_light)
        ax.xaxis.label.set_color(text_light)
        ax.yaxis.label.set_color(text_light)
        ax.title.set_color(text_light)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.grid(True, alpha=0.5, color=grid_color, linestyle='--')

    fig_ts, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig_ts.patch.set_facecolor(bg_dark)
    ax1, ax2 = axes[0], axes[1]

    ax1.plot(dates, actual_values, marker='o', linestyle='-', linewidth=2.5, markersize=6,
             label='Фактические значения', color=line_actual)
    ax1.plot(dates, predicted_values, marker='s', linestyle='--', linewidth=2.5, markersize=6,
             label='Прогноз модели', color=line_pred)
    ci_lower = predicted_values + ci_mean_hac[0]
    ci_upper = predicted_values + ci_mean_hac[1]
    x_vals = dates if has_order_date else range(len(dates))
    ax1.fill_between(x_vals, ci_lower, ci_upper, alpha=0.35, color=line_ci,
                     label='95% ДИ для среднего остатков (HAC)')
    ax1.set_xlabel('Дата')
    ax1.set_ylabel(f'Относительное изменение {target_column}')
    ax1.set_title(f'Временной ряд: факт vs прогноз — {target_platform} / {target_product}')
    textstr = f'MAE: {test_mae:.6f}\nRMSE: {test_rmse:.6f}\nR²: {test_r2:.4f}'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#262730', edgecolor=grid_color, alpha=0.95),
             color=text_light)
    ax1.legend(loc='upper right', fontsize=9, facecolor=bg_dark, edgecolor=grid_color, labelcolor=text_light)
    _dark_axes(ax1)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', color=text_light)

    bins = min(20, max(3, n_test // 3))
    ax2.hist(residuals_test, bins=bins, alpha=0.85, color=line_actual, edgecolor=text_light, linewidth=0.8, label='Остатки')
    ax2.axvline(mean_residuals, color=line_mean, linestyle='--', linewidth=2.5,
                label=f'Среднее: {mean_residuals:.6f}')
    ax2.axvline(ci_mean_hac[0], color=line_ci, linestyle=':', linewidth=2, label='95% ДИ (HAC)')
    ax2.axvline(ci_mean_hac[1], color=line_ci, linestyle=':', linewidth=2)
    ax2.axvline(0, color=text_light, linestyle='-', linewidth=1.2, alpha=0.8, label='Ноль (H₀)')
    ax2.set_xlabel('Остатки (факт - прогноз)')
    ax2.set_ylabel('Частота')
    ax2.set_title(f'Распределение остатков • t-тест с HAC: p-value = {p_value_hac:.6f}')
    ax2.legend(loc='best', fontsize=9, facecolor=bg_dark, edgecolor=grid_color, labelcolor=text_light)
    _dark_axes(ax2)
    plt.tight_layout()
    figures.append(fig_ts)

    fig_acf, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig_acf.patch.set_facecolor(bg_dark)
    if n_test > 1:
        acf_vals = acf(residuals_test, nlags=min(20, n_test - 1), fft=False)
        lags = range(len(acf_vals))
        axes[0].stem(lags, acf_vals, linefmt=line_actual, basefmt=' ', markerfmt='o')
        axes[0].axhline(y=0, color=text_light, linestyle='-', linewidth=1)
        conf_int = 1.96 / np.sqrt(n_test)
        axes[0].axhline(y=conf_int, color=line_pred, linestyle='--', linewidth=1.5, alpha=0.9, label='95% ДИ')
        axes[0].axhline(y=-conf_int, color=line_pred, linestyle='--', linewidth=1.5, alpha=0.9)
        axes[0].set_xlabel('Лаг')
        axes[0].set_ylabel('Автокорреляция')
        axes[0].set_title('ACF остатков', fontweight='bold')
        axes[0].legend(facecolor=bg_dark, edgecolor=grid_color, labelcolor=text_light)
        _dark_axes(axes[0])

    axes[1].plot(dates, residuals_test, marker='o', linestyle='-', linewidth=2, markersize=5,
                 color=line_actual)
    axes[1].axhline(y=0, color=text_light, linestyle='--', linewidth=2, label='Ноль')
    axes[1].axhline(y=mean_residuals, color=line_mean, linestyle=':', linewidth=2.5,
                   label=f'Среднее: {mean_residuals:.6f}')
    axes[1].set_xlabel('Дата')
    axes[1].set_ylabel('Остатки')
    axes[1].set_title('Остатки во времени', fontweight='bold')
    axes[1].legend(facecolor=bg_dark, edgecolor=grid_color, labelcolor=text_light)
    _dark_axes(axes[1])
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right', color=text_light)
    plt.tight_layout()
    figures.append(fig_acf)

    summary_lines = [
        "ИТОГОВАЯ СВОДКА СТАТИСТИЧЕСКИХ ТЕСТОВ",
        f"1. Автокорреляция: {'обнаружена' if has_autocorr else 'не обнаружена'}",
        f"2. T-тест с HAC (Newey-West): p-value = {p_value_hac:.6f} — {'значимо' if p_value_hac < 0.05 else 'незначимо'}",
        f"3. Тест Диболда-Мариано: p-value = {dm_pvalue:.6f} — {'значимо' if dm_pvalue < 0.05 else 'незначимо'}",
        f"ОБЩИЙ ВЫВОД: Различие между прогнозом и фактом — {conclusion}",
    ]

    return {
        'info': info_lines,
        'target_column': target_column,
        'metrics': {
            'train_mae': train_mae, 'train_rmse': train_rmse, 'train_r2': train_r2,
            'test_mae': test_mae, 'test_rmse': test_rmse, 'test_r2': test_r2,
        },
        'feature_importance': feature_importance,
        'time_series_df': test_predictions_with_dates,
        'residuals_info': residuals_info,
        'hac_results': hac_results,
        'dm_results': dm_results,
        'summary': summary_lines,
        'figures': figures,
        'target_platform': target_platform,
        'target_product': target_product,
    }
