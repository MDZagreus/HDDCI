# -*- coding: utf-8 -*-
"""
Веб-приложение Streamlit для анализа данных временных рядов.
Тёмная тема по умолчанию.
"""

import streamlit as st
import pandas as pd
from data_processor import process

# Тёмная палитра (хороший контраст текста)
DARK = {
    'bg': '#0e1117',
    'sidebar': '#1a1d24',
    'surface': '#262730',
    'text': '#eaeaea',
    'text_muted': '#9ca3af',
    'accent': '#7EB8DA',
    'accent_soft': '#4b6b7e',
}

st.set_page_config(
    page_title='Анализ временных рядов',
    page_icon='○',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Стили под тёмную тему — текст всегда читаемый на тёмном фоне
st.markdown(f"""
<style>
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {DARK['bg']} !important;
    }}
    .stSidebar, [data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
        background-color: {DARK['sidebar']} !important;
    }}
    /* Основной текст — светлый на тёмном */
    .stMarkdown, .stMarkdown p, .stMarkdown li, [data-testid="stMarkdown"] p,
    [data-testid="stMarkdown"] li, div[data-testid="stSidebar"] .stMarkdown,
    div[data-testid="stSidebar"] .stMarkdown p, .stText {{
        color: {DARK['text']} !important;
    }}
    h1, h2, h3, h4, [data-testid="stHeader"] {{
        color: {DARK['text']} !important;
        font-weight: 500;
    }}
    /* Метрики */
    .stMetric label, [data-testid="stMetricLabel"] {{
        color: {DARK['text_muted']} !important;
    }}
    .stMetric [data-testid="stMetricValue"], [data-testid="stMetricValue"] {{
        color: {DARK['text']} !important;
    }}
    [data-testid="stMetric"] {{
        background-color: {DARK['surface']};
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 3px solid {DARK['accent']};
    }}
    /* Инпуты и лейблы в сайдбаре */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {{
        color: {DARK['text']} !important;
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }}
    hr {{
        border-color: {DARK['accent_soft']};
        opacity: 0.5;
    }}
    /* Инфо/успех/предупреждение — читаемый текст */
    [data-testid="stAlert"] {{
        color: {DARK['text']};
    }}
</style>
""", unsafe_allow_html=True)


def main():
    st.title('○ Анализ временных рядов')
    st.markdown('Загрузите CSV с данными и выберите параметры анализа.')
    st.markdown('---')

    with st.sidebar:
        st.header('· Параметры')
        target_platform = st.text_input('Платформа (target_platform)', value='tta_android')
        target_product = st.text_input('Продукт (target_product)', value='avia')

    uploaded = st.file_uploader('Загрузите CSV-файл', type=['csv'], key='csv_upload')

    if uploaded is None:
        st.info('Загрузите CSV-файл, чтобы начать анализ.')
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f'Ошибка чтения файла: {e}')
        return

    if df.empty:
        st.warning('Файл пуст.')
        return

    # Проверка нужных колонок
    if 'test_period' not in df.columns:
        st.error("В данных должна быть колонка 'test_period'.")
        return

    # Доступные метрики для выбранной платформы и продукта
    prefix = f"{target_platform}_{target_product}_"
    available_metrics = sorted({
        c[len(prefix):] for c in df.columns
        if c.startswith(prefix) and len(c) > len(prefix)
    })
    default_metrics = ['revenue', 'orders', 'gmv', 'aov', 'searchers', 'cr']
    metric_options = available_metrics if available_metrics else default_metrics
    default_idx = metric_options.index('revenue') if 'revenue' in metric_options else 0

    with st.sidebar:
        target_metric = st.selectbox(
            'Метрика (target_metric)',
            options=metric_options,
            index=default_idx,
            help='Целевая метрика: колонка {platform}_{product}_{metric}'
        )

    target_col_candidate = f"{target_platform}_{target_product}_{target_metric}"
    if target_col_candidate not in df.columns:
        matching = [c for c in df.columns if c.startswith(prefix)]
        st.error(f"Целевая колонка '{target_col_candidate}' не найдена. Доступные: {matching[:15]}...")
        return

    with st.spinner('Выполняется анализ...'):
        try:
            result = process(
                df,
                target_platform=target_platform,
                target_product=target_product,
                target_metric=target_metric
            )
        except Exception as e:
            st.exception(e)
            return

    st.success('Анализ завершён.')

    # --- Информация о данных ---
    st.subheader('· Информация о данных')
    for line in result['info']:
        if '\n' in line:
            st.text(line.split('\n')[0])
            st.write(line.split('\n', 1)[1] if '\n' in line else '')
        else:
            st.text(line)
    st.markdown('---')

    # --- Метрики модели ---
    st.subheader('· Метрики модели')
    m = result['metrics']
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric('Train MAE', f'{m["train_mae"]:.6f}')
    with col2:
        st.metric('Train RMSE', f'{m["train_rmse"]:.6f}')
    with col3:
        st.metric('Train R²', f'{m["train_r2"]:.4f}')
    with col4:
        st.metric('Test MAE', f'{m["test_mae"]:.6f}')
    with col5:
        st.metric('Test RMSE', f'{m["test_rmse"]:.6f}')
    with col6:
        st.metric('Test R²', f'{m["test_r2"]:.4f}')
    st.markdown('---')

    # --- Топ признаков ---
    st.subheader('· Топ-10 важных признаков')
    st.dataframe(result['feature_importance'].head(10), use_container_width=True, hide_index=True)
    st.markdown('---')

    # --- Визуализации ---
    st.subheader('· Визуализации')
    for fig in result['figures']:
        st.pyplot(fig)
        st.markdown('')

    # --- Таблица временного ряда ---
    st.subheader('· Временной ряд (test_period)')
    st.dataframe(result['time_series_df'], use_container_width=True, hide_index=True)
    st.markdown('---')

    # --- Остатки и тесты ---
    st.subheader('· Остатки и статистические тесты')
    ri = result['residuals_info']
    st.write(f"Размер test_period: {ri['n_test']} | Среднее остатков: {ri['mean']:.6f} | "
             f"Стандартное отклонение: {ri['std']:.6f}")
    if ri.get('lb_pvalue') is not None:
        st.write(f"Тест Люнга-Бокса: p-value = {ri['lb_pvalue']:.6f} | "
                 f"Автокорреляция: {'да' if ri['has_autocorr'] else 'нет'}")

    # --- Диагностика отклонений прогноза (PIT и CUSUMSQ) ---
    if result.get('diagnostics') is not None:
        st.markdown('---')
        st.subheader('· Диагностика отклонений прогноза (Forecast deviation diagnostics)')
        for line in result['diagnostics']['summary_lines']:
            st.write(line)
        st.caption(
            "PIT (KS): маленький p → возможная некалибровка распределения прогноза. "
            "CUSUMSQ: маленький p → нестабильность дисперсии/режима."
        )

    st.markdown('---')
    st.subheader('· Итоговая сводка')
    for line in result['summary']:
        st.write(line)


main()
