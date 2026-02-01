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


def cusumsq(resid):
    """CUSUM of squares path and statistic."""
    e2 = resid ** 2
    S = np.cumsum(e2) / np.sum(e2)
    t = np.arange(1, len(resid) + 1) / len(resid)
    stat = np.max(np.abs(S - t))
    return S, stat


def cusumsq_bootstrap(resid, n_boot=2000, seed=42):
    """Bootstrap CUSUMSQ for small samples (T <= 45)."""
    rng = np.random.default_rng(seed)
    T = len(resid)
    S_obs, stat_obs = cusumsq(resid - resid.mean())
    stats_boot = []
    S_boot = []
    for _ in range(n_boot):
        rb = rng.choice(resid, size=T, replace=True)
        Sb, sb = cusumsq(rb - rb.mean())
        stats_boot.append(sb)
        S_boot.append(Sb)
    stats_boot = np.array(stats_boot)
    S_boot = np.array(S_boot)
    p_value = np.mean(stats_boot >= stat_obs)
    lo = np.quantile(S_boot, 0.025, axis=0)
    hi = np.quantile(S_boot, 0.975, axis=0)
    return {"S": S_obs, "stat": stat_obs, "p_value": p_value, "band_lo": lo, "band_hi": hi}


class ForecastDeviationDiagnostics:
    """
    Диагностика отклонений прогноза (факт vs прогноз).
    Вход: DataFrame с колонками date, actual, predicted.
    """

    def __init__(self, df: pd.DataFrame):
        required = {"date", "actual", "predicted"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns {required}")
        self.df = (
            df[["date", "actual", "predicted"]]
            .dropna()
            .sort_values("date")
            .reset_index(drop=True)
        )
        if len(self.df) < 15:
            raise ValueError("Too few observations (need at least ~15)")
        self.y = self.df["actual"].values
        self.mu = self.df["predicted"].values
        self.t = self.df["date"].values
        self.resid = self.y - self.mu
        self.sigma = np.std(self.resid, ddof=1)
        self.z = self.resid / self.sigma
        self.pit = stats.norm.cdf(self.z)
        self.results = None

    def run_tests(self):
        results = {}
        ks_stat, ks_p = stats.kstest(self.pit, "uniform")
        results["PIT_KS"] = (ks_stat, ks_p)
        cusq = cusumsq_bootstrap(self.z)
        results["CUSUMSQ"] = cusq
        self.results = results
        return results

    def summary_lines(self):
        r = self.results
        return [
            f"Sample size: {len(self.z)}",
            f"PIT (KS): stat={r['PIT_KS'][0]:.3f}, p={r['PIT_KS'][1]:.3f}",
            f"CUSUMSQ p-value: {r['CUSUMSQ']['p_value']:.3f}",
        ]

    def build_figures(self, bg_dark, text_light, line_actual, line_pred, line_ci, grid_color):
        """Строит графики в стиле приложения (тёмный фон, контрастные линии)."""
        figures = []
        cusq = self.results["CUSUMSQ"]
        S, lo, hi = cusq["S"], cusq["band_lo"], cusq["band_hi"]

        def _dark_axes(ax):
            ax.set_facecolor(bg_dark)
            ax.tick_params(colors=text_light)
            ax.xaxis.label.set_color(text_light)
            ax.yaxis.label.set_color(text_light)
            ax.title.set_color(text_light)
            for spine in ax.spines.values():
                spine.set_color(grid_color)
            ax.grid(True, alpha=0.5, color=grid_color, linestyle='--')

        # 1. z_t over time
        fig1, ax = plt.subplots(figsize=(10, 4))
        fig1.patch.set_facecolor(bg_dark)
        ax.plot(self.t, self.z, marker='o', linestyle='-', linewidth=2, markersize=5, color=line_actual)
        ax.axhline(0, color=text_light, linestyle='--', linewidth=1.5)
        ax.axhline(2, color=line_pred, linestyle='--', linewidth=1, alpha=0.8)
        ax.axhline(-2, color=line_pred, linestyle='--', linewidth=1, alpha=0.8)
        ax.set_title('Стандартизированные ошибки прогноза (z_t)')
        ax.set_ylabel('z')
        _dark_axes(ax)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', color=text_light)
        plt.tight_layout()
        figures.append(fig1)

        # 2. PIT histogram
        fig2, ax = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor(bg_dark)
        ax.hist(self.pit, bins=8, density=True, color=line_actual, edgecolor=text_light, linewidth=0.8)
        ax.axhline(1.0, color=text_light, linestyle='--', linewidth=2)
        ax.set_title('PIT histogram')
        ax.set_xlabel('PIT')
        _dark_axes(ax)
        plt.tight_layout()
        figures.append(fig2)

        # 3. CUSUMSQ
        x = np.arange(1, len(S) + 1) / len(S)
        fig3, ax = plt.subplots(figsize=(10, 4))
        fig3.patch.set_facecolor(bg_dark)
        ax.plot(x, S, label='CUSUMSQ', color=line_actual, linewidth=2)
        ax.plot(x, x, linestyle='--', label='Expected', color=text_light, linewidth=1.5)
        ax.plot(x, lo, linestyle='--', alpha=0.7, color=line_ci)
        ax.plot(x, hi, linestyle='--', alpha=0.7, color=line_ci)
        ax.set_title(f"CUSUMSQ (p-value = {cusq['p_value']:.3f})")
        ax.set_xlabel('t / T')
        ax.legend(facecolor=bg_dark, edgecolor=grid_color, labelcolor=text_light)
        _dark_axes(ax)
        plt.tight_layout()
        figures.append(fig3)

        return figures


def process(df, target_platform='tta_android', target_product='avia', target_metric='revenue',
            test_period_start=None, test_period_end=None):
    """
    Обрабатывает DataFrame и выполняет полный анализ.

    Возвращает dict с ключами:
    - info: текстовая информация о данных
    - metrics: метрики модели
    - feature_importance: DataFrame важности признаков
    - time_series_df: DataFrame с датами, фактом и прогнозом
    - residuals_info: информация об остатках
    - diagnostics: PIT и CUSUMSQ (если n_test >= 15)
    - summary: итоговая сводка
    - figures: список matplotlib.Figure для отображения
    """
    D_result_d = df.copy()

    # Поддержка колонки order_date (парсим даты при необходимости)
    if 'order_date' in D_result_d.columns:
        if not pd.api.types.is_datetime64_any_dtype(D_result_d['order_date']):
            D_result_d['order_date'] = pd.to_datetime(D_result_d['order_date'])

    # Как в analysis.ipynb: убираем order_date из таблицы для обучения
    D_result = D_result_d.drop(columns=['order_date'], errors='ignore') if 'order_date' in D_result_d.columns else D_result_d.copy()

    # Целевая колонка: {platform}_{product}_{metric}
    exact_target = f"{target_platform}_{target_product}_{target_metric}"
    if exact_target in D_result.columns:
        target_column = exact_target
    else:
        target_column = None
        prefix = f"{target_platform}_{target_product}_"
        for col in D_result.columns:
            if col.startswith(prefix) and col.endswith(f"_{target_metric}"):
                target_column = col
                break

    if target_column is None:
        matching = [c for c in D_result.columns if c.startswith(f"{target_platform}_{target_product}_")]
        raise ValueError(
            f"Целевая колонка для platform='{target_platform}', product='{target_product}', metric='{target_metric}' не найдена. "
            f"Доступные колонки для этой пары: {matching[:15]}..."
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
    mean_residuals = float(np.mean(residuals_test))

    # Тест Люнга-Бокса (для информации об остатках)
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

    info_lines = [
        f"Целевая колонка: {target_column}",
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

    # --- Диагностика отклонений прогноза (PIT и CUSUMSQ) ---
    diagnostics_results = None
    diagnostics_figures = []
    if n_test >= 15:
        try:
            diag = ForecastDeviationDiagnostics(test_predictions_with_dates)
            diag.run_tests()
            diagnostics_results = {
                'PIT_KS': diag.results['PIT_KS'],
                'CUSUMSQ_pvalue': diag.results['CUSUMSQ']['p_value'],
                'summary_lines': diag.summary_lines(),
            }
            bg_dark = '#1a1d24'
            text_light = '#eaeaea'
            line_actual = '#00D4FF'
            line_pred = '#FF6B6B'
            line_ci = '#9B59B6'
            grid_color = '#4a4a5a'
            diagnostics_figures = diag.build_figures(
                bg_dark, text_light, line_actual, line_pred, line_ci, grid_color
            )
        except Exception:
            diagnostics_results = None
            diagnostics_figures = []

    # --- Построение графиков: факт vs прогноз, затем z_t, PIT, CUSUMSQ ---
    figures = []
    bg_dark = '#1a1d24'
    text_light = '#eaeaea'
    line_actual = '#00D4FF'
    line_pred = '#FF6B6B'
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

    # График факт vs прогноз
    fig_ts, ax = plt.subplots(figsize=(12, 5))
    fig_ts.patch.set_facecolor(bg_dark)
    ax.plot(dates, actual_values, marker='o', linestyle='-', linewidth=2.5, markersize=6,
            label='Фактические значения', color=line_actual)
    ax.plot(dates, predicted_values, marker='s', linestyle='--', linewidth=2.5, markersize=6,
            label='Прогноз модели', color=line_pred)
    ax.set_xlabel('Дата')
    ax.set_ylabel(f'Относительное изменение {target_column}')
    ax.set_title(f'Факт vs прогноз — {target_platform} / {target_product} / {target_metric}')
    textstr = f'MAE: {test_mae:.6f}\nRMSE: {test_rmse:.6f}\nR²: {test_r2:.4f}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#262730', edgecolor=grid_color, alpha=0.95),
            color=text_light)
    ax.legend(loc='upper right', fontsize=9, facecolor=bg_dark, edgecolor=grid_color, labelcolor=text_light)
    _dark_axes(ax)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', color=text_light)
    plt.tight_layout()
    figures.append(fig_ts)

    figures.extend(diagnostics_figures)

    summary_lines = [
        "ИТОГОВАЯ СВОДКА",
        f"1. Автокорреляция остатков (Ljung-Box): {'обнаружена' if has_autocorr else 'не обнаружена'}",
    ]
    if diagnostics_results is not None:
        summary_lines.append(
            f"2. PIT (KS): stat={diagnostics_results['PIT_KS'][0]:.3f}, p={diagnostics_results['PIT_KS'][1]:.3f}"
        )
        summary_lines.append(
            f"3. CUSUMSQ p-value: {diagnostics_results['CUSUMSQ_pvalue']:.3f}"
        )
    else:
        summary_lines.append("2. Диагностика PIT/CUSUMSQ: недоступна (нужно n_test >= 15)")

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
        'diagnostics': diagnostics_results,
        'summary': summary_lines,
        'figures': figures,
        'target_platform': target_platform,
        'target_product': target_product,
        'target_metric': target_metric,
    }
