import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz 
from dateutil.relativedelta import relativedelta 
import numpy as np 

# --- Configuración Inicial y Constantes ---

st.set_page_config(page_title="Dashboard Climático Lab BD", layout="wide")

# Constantes definidas en los requisitos
API_BASE_URL = "http://localhost:8000"
AVAILABLE_CITIES = ["Barranquilla", "Bogota", "Bucaramanga", "Ibague", "Cali", "Cartagena", "Cucuta", "Medellin", "Leticia", "Pereira", "Santa Marta"]
CITY_COORDS = {
    "Barranquilla": {"lat": 11.0, "lon": -74.75},
    "Bogota": {"lat": 4.625, "lon": -74.125},
    "Bucaramanga": {"lat": 7.0, "lon": -73.125},
    "Ibague": {"lat": 4.375, "lon": -75.25},
    "Cali": {"lat": 3.5, "lon": -76.5},
    "Cartagena": {"lat": 10.25, "lon": -75.5},
    "Cucuta": {"lat": 8.0, "lon": -72.5},
    "Medellin": {"lat": 6.125, "lon": -75.75},
    "Leticia": {"lat": -4.25, "lon": -69.875},
    "Pereira": {"lat": 4.75, "lon": -75.75},
    "Santa Marta": {"lat": 11.125, "lon": -74.125}
}
WEATHER_VARIABLES_KEYS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "wind_speed_10m", "uv_index", "pressure_msl", "shortwave_radiation", "cloud_cover"]

VARIABLE_MAP_ES = {
    "temperature_2m": "Temperatura (°C)",
    "relative_humidity_2m": "Humedad Relativa (%)",
    "dew_point_2m": "Punto de Rocío (°C)",
    "precipitation": "Precipitación (mm)",
    "wind_speed_10m": "Velocidad Viento (m/s)",
    "uv_index": "Índice UV (0-11+)",
    "pressure_msl": "Presión Nivel Mar (hPa)",
    "shortwave_radiation": "Radiación Solar (W/m²)",
    "cloud_cover": "Cobertura Nubosa (%)"
}

THRESHOLDS = {
    "temperature_2m": {"Calor Extremo": 32, "Calor": 28, "Frío": 10}
    # Add other thresholds if needed
}
COLOMBIA_TZ = pytz.timezone('America/Bogota')

# --- Funciones Auxiliares (Llamadas API y Procesamiento) ---

@st.cache_data(ttl=300)
def fetch_api_data(endpoint: str, params: dict = None, request_desc: str = "datos"):
    """Realiza una llamada GET a la API y maneja errores comunes."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        try: error_detail = http_err.response.json().get("detail", http_err.response.text)
        except Exception: error_detail = http_err.response.text
        if status_code == 404: st.warning(f"⚠️ No se encontraron {request_desc} para: {endpoint} (Parámetros: {params}). (Error 404)")
        elif status_code == 422: st.warning(f"⚠️ Parámetros inválidos para {request_desc}: {endpoint} (Parámetros: {params}). Detalle: {error_detail} (Error 422)")
        elif status_code >= 500: st.error(f"❌ Error del servidor ({status_code}) al solicitar {request_desc}: {endpoint}. Detalle: {error_detail}")
        else: st.error(f"❌ Error HTTP ({status_code}) al cargar {request_desc} de {endpoint}: {error_detail}")
        return None
    except requests.exceptions.ConnectionError as conn_err: st.error(f"❌ Error de conexión API ({request_desc}) en {url}: {conn_err}"); return None
    except requests.exceptions.Timeout as timeout_err: st.error(f"❌ Tiempo de espera excedido para {request_desc} en {url}: {timeout_err}"); return None
    except requests.exceptions.RequestException as req_err: st.error(f"❌ Error inesperado de red ({request_desc}) en {url}: {req_err}"); return None

def process_data_for_plotting(data: list | dict | None, time_col: str | None = 'time', value_col: str | None = None, parse_dates: bool = True) -> pd.DataFrame:
    """Convierte la respuesta JSON de la API en un DataFrame de Pandas."""
    if data is None: return pd.DataFrame()
    if isinstance(data, dict):
         if all(isinstance(v, dict) for v in data.values()):
             df = pd.DataFrame.from_dict(data, orient='index').reset_index().rename(columns={'index': 'city'})
             if parse_dates and time_col and time_col in df.columns:
                  try: df[time_col] = pd.to_datetime(df[time_col], errors='coerce', infer_datetime_format=True)
                  except Exception: pass # Fail silently on date parse error here
             if value_col and value_col in df.columns: df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
             if parse_dates and time_col and time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]): df = df.dropna(subset=[time_col])
             return df
         else: return pd.DataFrame() # Non-processable dict
    if isinstance(data, list):
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        if parse_dates and time_col and time_col in df.columns:
            try:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce', infer_datetime_format=True)
                if pd.api.types.is_datetime64_any_dtype(df[time_col]): df = df.dropna(subset=[time_col])
            except Exception: pass # Fail silently
        if value_col and value_col in df.columns: df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        return df
    else: return pd.DataFrame() # Unexpected format

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convierte un DataFrame a CSV bytes."""
    return df.to_csv(index=False).encode('utf-8')

# --- Componentes del Dashboard ---

def display_trends_chart(city: str, start_date: datetime.date, end_date: datetime.date):
    """Muestra la sección de tendencias."""
    st.subheader(f"📈 Evolución del Clima en {city}")
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_trend_var_es = st.selectbox(
            "Selecciona la Variable",
            options=list(VARIABLE_MAP_ES.values()),
            index=0,
            key=f"trend_var_{city}"
        )
        trend_var = [k for k, v in VARIABLE_MAP_ES.items() if v == selected_trend_var_es][0]
    with col2:
        compare_yoy = st.checkbox("Comparar Año Anterior", key=f"trend_yoy_{city}", value=False, help="Superpone los datos del mismo período del año pasado.")

    st.caption(f"Observa cómo cambió la **{selected_trend_var_es}** en **{city}** entre las fechas seleccionadas, y opcionalmente compárala con el año anterior.")

    endpoint = f"/weather/{city}/trends"
    current_params = {"variable_name": trend_var, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
    prev_trends_df = pd.DataFrame()
    prev_start_date = None

    with st.spinner(f"Cargando datos de '{selected_trend_var_es}' para {city}..."):
        trends_data = fetch_api_data(endpoint, current_params, request_desc=f"tendencias actuales para {city}")
        trends_df = process_data_for_plotting(trends_data, time_col='time', value_col=trend_var)

    if compare_yoy and not trends_df.empty:
        try:
            prev_start_date = start_date - relativedelta(years=1)
            prev_end_date = end_date - relativedelta(years=1)
            prev_params = {"variable_name": trend_var, "start_date": prev_start_date.strftime('%Y-%m-%d'), "end_date": prev_end_date.strftime('%Y-%m-%d')}
            with st.spinner(f"Cargando datos del año anterior ({prev_start_date.year})..."):
                prev_trends_data = fetch_api_data(endpoint, prev_params, request_desc=f"tendencias año anterior para {city}")
                prev_trends_df = process_data_for_plotting(prev_trends_data, time_col='time', value_col=trend_var)
                if not prev_trends_df.empty and pd.api.types.is_datetime64_any_dtype(prev_trends_df['time']):
                     # Desplaza el eje de tiempo del año anterior para alinearlo con el año actual en el gráfico.
                     prev_trends_df['time_shifted'] = prev_trends_df['time'].apply(lambda d: d + relativedelta(years=1))
                else: prev_trends_df = pd.DataFrame()
        except Exception as e: st.error(f"Error obteniendo datos del año anterior: {e}"); prev_trends_df = pd.DataFrame()

    if not trends_df.empty and trend_var in trends_df.columns:
        with st.spinner("Generando gráfico de evolución..."):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trends_df['time'], y=trends_df[trend_var], mode='lines+markers', name=f'{selected_trend_var_es} ({start_date.year})', hovertemplate=f"<b>Fecha ({start_date.year})</b>: %{{x|%d-%b %H:%M}}<br><b>Valor</b>: %{{y}}<extra></extra>"))
            if compare_yoy and not prev_trends_df.empty and 'time_shifted' in prev_trends_df.columns and prev_start_date:
                 fig.add_trace(go.Scatter(x=prev_trends_df['time_shifted'], y=prev_trends_df[trend_var], mode='lines', name=f'{selected_trend_var_es} ({prev_start_date.year})', line=dict(dash='dash', color='grey'), opacity=0.7, hovertemplate=f"<b>Fecha ({prev_start_date.year}, Eje {start_date.year})</b>: %{{x|%d-%b %H:%M}}<br><b>Valor</b>: %{{y}}<extra></extra>"))
            if trend_var in THRESHOLDS:
                 for label, value in THRESHOLDS[trend_var].items(): fig.add_hline(y=value, line_dash="dash", line_color="red", opacity=0.6, annotation_text=label, annotation_position="bottom right")
            fig.update_layout(title=f"Evolución de {selected_trend_var_es} en {city}", xaxis_title="Fecha y Hora", yaxis_title=selected_trend_var_es, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
    elif trends_df.empty and trends_data is not None:
        st.info(f"No se encontraron datos de '{selected_trend_var_es}' para {city} en el período seleccionado.")

def display_precipitation_summary(city: str, start_date: datetime.date, end_date: datetime.date):
    """Muestra la sección de resumen de precipitación."""
    st.subheader(f"💧 Resumen de Lluvias en {city}")
    granularity = st.selectbox("Agrupar Lluvia por:", options=["daily", "weekly", "monthly"], index=0, format_func=lambda x: {"daily": "Día", "weekly": "Semana", "monthly": "Mes"}.get(x, x), key=f"precip_granularity_{city}")

    granularity_es_map = {"daily": "diario", "weekly": "semanal", "monthly": "mensual"}
    st.caption(f"Visualiza el total de lluvia acumulada ({granularity_es_map.get(granularity, '')}) en **{city}**. Barras altas indican más lluvia.")

    endpoint = f"/weather/{city}/precipitation/summary"
    params = {"granularity": granularity, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
    with st.spinner(f"Cargando resumen de lluvias por {granularity}..."):
        precip_data = fetch_api_data(endpoint, params, request_desc=f"resumen precipitación para {city}")
        precip_df = process_data_for_plotting(precip_data, time_col='period_start', value_col='total_precipitation')

    if not precip_df.empty and 'total_precipitation' in precip_df.columns:
        with st.spinner("Generando gráfico de lluvias..."):
            if 'period_start' in precip_df.columns and not pd.api.types.is_datetime64_any_dtype(precip_df['period_start']): precip_df['period_start'] = pd.to_datetime(precip_df['period_start'], errors='coerce')
            precip_df = precip_df.dropna(subset=['period_start', 'total_precipitation']).sort_values('period_start')
            if granularity == 'daily': precip_df['period_label'] = precip_df['period_start'].dt.strftime('%d-%b-%Y')
            elif granularity == 'weekly': precip_df['period_label'] = precip_df['period_start'].dt.strftime('Sem %U (%Y)')
            elif granularity == 'monthly': precip_df['period_label'] = precip_df['period_start'].dt.strftime('%b %Y')
            else: precip_df['period_label'] = precip_df['period_start'].astype(str)
            granularity_title = {"daily": "Diaria", "weekly": "Semanal", "monthly": "Mensual"}.get(granularity, granularity.title())
            fig = px.bar(precip_df, x='period_label', y='total_precipitation', title=f"Lluvia Acumulada {granularity_title} en {city}", labels={'period_label': f'Período ({granularity_title})', 'total_precipitation': 'Lluvia Total (mm)'})
            fig.update_traces(hovertemplate="<b>Período</b>: %{x}<br><b>Lluvia</b>: %{y} mm<extra></extra>")
            fig.update_layout(xaxis={'type': 'category'})
            st.plotly_chart(fig, use_container_width=True)
    elif precip_df.empty and precip_data is not None:
        st.info(f"No se encontraron datos de lluvia para {city} con la agrupación seleccionada.")

@st.cache_data(ttl=600)
def fetch_and_process_period_map_data(variable_key: str, start_date: datetime.date, end_date: datetime.date):
    """Obtiene y agrega datos (promedio/total) para el mapa de período."""
    city_data_agg = {}
    variable_es = VARIABLE_MAP_ES.get(variable_key, variable_key)
    aggregation_type = "Promedio"
    with st.spinner(f"Calculando {'totales' if variable_key == 'precipitation' else 'promedios'} de '{variable_es}' para el mapa ({start_date.strftime('%d-%b')} a {end_date.strftime('%d-%b-%Y')})..."):
        if variable_key == "precipitation":
            # La precipitación se suma en el período, no se promedia.
            aggregation_type = "Total Acumulado"
            city_totals = {city: 0.0 for city in AVAILABLE_CITIES}
            for city_name in AVAILABLE_CITIES:
                endpoint = f"/weather/{city_name}/precipitation/summary"
                params = {"granularity": 'daily', "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
                city_precip_data = fetch_api_data(endpoint, params, request_desc=f"precip. diaria para {city_name}")
                if city_precip_data and isinstance(city_precip_data, list):
                    try:
                        df_city_precip = pd.DataFrame(city_precip_data)
                        if 'total_precipitation' in df_city_precip.columns:
                            total = pd.to_numeric(df_city_precip['total_precipitation'], errors='coerce').sum()
                            city_totals[city_name] = total if pd.notna(total) else 0.0
                    except Exception: pass # Ignore errors for single city aggregation
            city_data_agg = city_totals
            if not any(v > 0 for v in city_totals.values()): city_data_agg = {} # Reset if no data found
        else: # Calculate average for other variables
            # Para otras variables, se calcula el promedio general del período.
            aggregation_type = "Promedio"
            cities_str = ",".join(AVAILABLE_CITIES)
            endpoint = "/weather/averages"; average_col_name = f"average_{variable_key}"
            params = {"variable_name": variable_key, "granularity": 'daily', "cities": cities_str, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
            averages_data = fetch_api_data(endpoint, params, request_desc=f"promedios diarios para mapa")
            if averages_data and isinstance(averages_data, list):
                try:
                    df_averages = pd.DataFrame(averages_data)
                    if average_col_name in df_averages.columns and 'city' in df_averages.columns:
                        df_averages[average_col_name] = pd.to_numeric(df_averages[average_col_name], errors='coerce')
                        # Agrupa por ciudad y calcula la media de los promedios diarios devueltos para obtener la media del período completo.
                        city_period_means = df_averages.groupby('city')[average_col_name].mean()
                        city_data_agg = city_period_means.dropna().to_dict()
                except Exception: pass # Ignore errors in aggregation

        if not city_data_agg: return None, aggregation_type
        try:
            map_df = pd.DataFrame(list(city_data_agg.items()), columns=['city', 'valor_mapa'])
            map_df['lat'] = map_df['city'].map(lambda city: CITY_COORDS.get(city, {}).get('lat'))
            map_df['lon'] = map_df['city'].map(lambda city: CITY_COORDS.get(city, {}).get('lon'))
            map_df['variable_nombre_es'] = variable_es
            map_df['periodo'] = f"{start_date.strftime('%d-%b')} a {end_date.strftime('%d-%b-%Y')}"
            map_df['tipo_agregacion'] = aggregation_type
            map_df = map_df.dropna(subset=['lat', 'lon', 'valor_mapa'])
            return map_df if not map_df.empty else None, aggregation_type
        except Exception: return None, aggregation_type

def display_period_summary_map_section(start_date: datetime.date, end_date: datetime.date):
    """
    Muestra el mapa de promedios/totales por período,
    utilizando las fechas globales proporcionadas.
    """
    st.subheader("🗺️ Resumen Climático en el Mapa por Período")

    selected_variable_es = st.selectbox(
        "Variable a Mapear",
        options=list(VARIABLE_MAP_ES.values()),
        index=0,
        key="map_period_var"
    )
    variable_key = [k for k, v in VARIABLE_MAP_ES.items() if v == selected_variable_es][0]

    is_precip = variable_key == 'precipitation'
    aggregation_type_text = "Total Acumulado" if is_precip else "Promedio"
    st.caption(f"Compara el valor **{aggregation_type_text.lower()}** de **{selected_variable_es}** entre ciudades para el período global seleccionado ({start_date.strftime('%d-%b-%Y')} a {end_date.strftime('%d-%b-%Y')}). Usa la leyenda de color para interpretar.")

    map_df, aggregation_type = fetch_and_process_period_map_data(variable_key, start_date, end_date)

    if map_df is not None and not map_df.empty:
        with st.spinner("Generando mapa resumen..."):
            try:
                vmin = map_df['valor_mapa'].min(); vmax = map_df['valor_mapa'].max()
                if pd.isna(vmin) or pd.isna(vmax): vmin, vmax = None, None
                elif vmin == vmax: vmin -= 1; vmax += 1 # Add buffer if min == max

                color_scale = px.colors.sequential.Viridis # Default
                if variable_key in ["temperature_2m", "dew_point_2m"]:
                    color_scale = px.colors.sequential.RdBu_r # Red-Blue reversed (Red hot)
                elif variable_key in ["precipitation", "relative_humidity_2m", "cloud_cover"]:
                    color_scale = px.colors.sequential.Blues # Blue tones for water/clouds
                elif variable_key in ["wind_speed_10m", "uv_index", "shortwave_radiation", "pressure_msl"]:
                    color_scale = px.colors.sequential.YlOrRd # Yellow-Orange-Red for intensity

                size_col = None
                if map_df['valor_mapa'].min() >= 0:
                     # Añade un tamaño base constante para evitar puntos de tamaño cero y escala según el valor.
                     base_size_offset = (vmax * 0.05 if vmax and pd.notna(vmax) and vmax > 0 else 1)
                     map_df['size_plot'] = map_df['valor_mapa'].fillna(0) + base_size_offset
                     size_col = 'size_plot'

                fig = px.scatter_mapbox(
                    map_df, lat="lat", lon="lon", color="valor_mapa",
                    size=size_col, size_max=30 if size_col else None,
                    hover_name="city",
                    hover_data={'variable_nombre_es': True, 'tipo_agregacion': True, 'valor_mapa': ":.2f", 'periodo': True, 'lat': False, 'lon': False, 'size_plot': False},
                    color_continuous_scale=color_scale,
                    range_color=[vmin, vmax] if vmin is not None else None,
                    mapbox_style="carto-positron", zoom=4.2, center={"lat": 4.57, "lon": -74.29},
                    title=f"Mapa Resumen: {selected_variable_es} ({aggregation_type})",
                    labels={"valor_mapa": f"{aggregation_type_text}"}
                )
                # El hovertemplate referencia los datos en hover_data por su índice (customdata[0], customdata[1], etc.).
                fig.update_traces(hovertemplate="<br>".join([
                    "<b>Ciudad:</b> %{hovertext}",
                    "<b>Variable:</b> %{customdata[0]}",
                    "<b>Tipo:</b> %{customdata[1]}",
                    f"<b>Valor:</b> %{{customdata[2]:.2f}}",
                    "<b>Período:</b> %{customdata[3]}<extra></extra>"
                ]))
                fig.update_layout(
                    coloraxis_colorbar=dict(
                        title=f"{aggregation_type_text}<br>{selected_variable_es.split('(')[0]}",
                        thicknessmode="pixels", thickness=15,
                        lenmode="fraction", len=0.75,
                        yanchor="middle", y=0.5
                    ),
                    margin={"r":10,"t":50,"l":10,"b":10}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generando mapa del período: {e}")
    else:
        st.info("No hay datos disponibles para mostrar en el mapa resumen con los parámetros seleccionados.")

def display_correlation_scatter(city: str, start_date: datetime.date, end_date: datetime.date):
    """Muestra la sección de correlación."""
    st.subheader(f"🔄 Relación entre Variables en {city}")
    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        selected_var_x_es = st.selectbox("Variable Eje X", options=list(VARIABLE_MAP_ES.values()), index=0, key=f"corr_var_x_{city}")
        corr_var_x = [k for k, v in VARIABLE_MAP_ES.items() if v == selected_var_x_es][0]
    with col2:
        available_y_vars_es = [v for k, v in VARIABLE_MAP_ES.items() if k != corr_var_x]
        default_y_val_es = VARIABLE_MAP_ES.get("relative_humidity_2m", available_y_vars_es[0])
        selected_var_y_es = st.selectbox("Variable Eje Y", options=available_y_vars_es, index=available_y_vars_es.index(default_y_val_es) if default_y_val_es in available_y_vars_es else 0, key=f"corr_var_y_{city}")
        corr_var_y = [k for k, v in VARIABLE_MAP_ES.items() if v == selected_var_y_es][0]
    with col3: show_trendline = st.checkbox("Mostrar Tendencia", key=f"corr_trend_{city}", value=True, help="Dibuja una línea mostrando la tendencia general.")

    st.caption(f"Explora si **{selected_var_x_es}** y **{selected_var_y_es}** se mueven juntas (correlación) en **{city}**. Usa el coeficiente abajo para medir la fuerza.")

    endpoint = f"/weather/{city}/correlation"
    params = {"variable_x": corr_var_x, "variable_y": corr_var_y, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
    with st.spinner(f"Cargando datos de correlación para {city}..."):
        correlation_data = fetch_api_data(endpoint, params, request_desc=f"correlación para {city}")
        correlation_df = process_data_for_plotting(correlation_data, time_col='time', parse_dates=True)

    corr_value = None
    if not correlation_df.empty and corr_var_x in correlation_df.columns and corr_var_y in correlation_df.columns:
         with st.spinner("Generando gráfico de correlación..."):
             correlation_df[corr_var_x] = pd.to_numeric(correlation_df[corr_var_x], errors='coerce')
             correlation_df[corr_var_y] = pd.to_numeric(correlation_df[corr_var_y], errors='coerce')
             plot_df = correlation_df.dropna(subset=[corr_var_x, corr_var_y])
             if not plot_df.empty and len(plot_df) > 1:
                fig = px.scatter(plot_df, x=corr_var_x, y=corr_var_y, title=f"Relación entre {selected_var_y_es} y {selected_var_x_es}", labels={corr_var_x: selected_var_x_es, corr_var_y: selected_var_y_es}, trendline="ols" if show_trendline else None, trendline_color_override='red', hover_data={'time': '|%d-%b %H:%M'})
                fig.update_traces(hovertemplate=f"<b>{selected_var_x_es}</b>: %{{x}}<br><b>{selected_var_y_es}</b>: %{{y}}<br><b>Fecha</b>: %{{customdata[0]}}<extra></extra>")
                st.plotly_chart(fig, use_container_width=True)
                try: corr_value = plot_df[corr_var_x].corr(plot_df[corr_var_y])
                except Exception: pass
             elif len(plot_df) <= 1: st.info("No hay suficientes datos para graficar/calcular correlación.")
             else: st.info("No hay datos válidos para graficar correlación.")
    elif correlation_df.empty and correlation_data is not None: st.info(f"No se encontraron datos de correlación para {city} en el período.")
    if corr_value is not None:
         corr_desc = "Positiva Fuerte" if corr_value > 0.7 else "Positiva Moderada" if corr_value > 0.4 else "Positiva Débil" if corr_value > 0.1 else "Negativa Fuerte" if corr_value < -0.7 else "Negativa Moderada" if corr_value < -0.4 else "Negativa Débil" if corr_value < -0.1 else "Muy Débil o Nula"
         st.metric(label=f"Coef. Correlación Lineal", value=f"{corr_value:.3f}", help=f"Interpretación: {corr_desc}")

def display_comparative_averages(start_date: datetime.date, end_date: datetime.date):
    """Muestra la sección de promedios comparativos."""
    st.subheader("🆚 Comparación de Promedios entre Ciudades")
    col1, col2, col3 = st.columns(3)
    with col1: comp_cities = st.multiselect("Ciudades a Comparar", options=AVAILABLE_CITIES, default=["Barranquilla", "Bogota", "Medellin"], key="comp_cities")
    with col2:
        selected_comp_var_es = st.selectbox("Variable a Comparar", options=list(VARIABLE_MAP_ES.values()), index=0, key="comp_var_es")
        comp_var = [k for k, v in VARIABLE_MAP_ES.items() if v == selected_comp_var_es][0]
    with col3: comp_granularity = st.selectbox("Agrupar Promedios por", options=["hourly", "daily", "weekly", "monthly"], index=0, format_func=lambda x: {"hourly":"Hora","daily":"Día","weekly":"Día Semana","monthly":"Mes"}.get(x,x), key="comp_granularity")
    if not comp_cities: st.warning("Por favor, selecciona al menos una ciudad."); return

    granularity_es_avg_map = {"hourly": "hora", "daily": "día", "weekly": "día de la semana", "monthly": "mes"}
    granularity_es_avg = granularity_es_avg_map.get(comp_granularity, comp_granularity)
    st.caption(f"Compara el promedio de **{selected_comp_var_es}** por **{granularity_es_avg}** entre las ciudades seleccionadas.")

    endpoint = "/weather/averages"
    params = {"cities": ",".join(comp_cities), "variable_name": comp_var, "granularity": comp_granularity, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
    with st.spinner(f"Cargando promedios comparativos por {comp_granularity}..."):
        averages_data = fetch_api_data(endpoint, params, request_desc="promedios comparativos")
        period_key_map = {"hourly": "hour", "daily": "period_start", "weekly": "day_of_week_iso", "monthly": "month"}
        period_col = period_key_map.get(comp_granularity, "period_key")
        average_col = f"average_{comp_var}"
        averages_df = process_data_for_plotting(averages_data, time_col=None, value_col=average_col, parse_dates=False)

    if not averages_df.empty and period_col in averages_df.columns and average_col in averages_df.columns:
        with st.spinner("Generando gráfico comparativo..."):
            averages_df = averages_df.dropna(subset=[average_col])
            x_label = f"Período ({granularity_es_avg.title()})"; y_label = f"Promedio {selected_comp_var_es}"
            title = f"Comparación Promedio de {selected_comp_var_es} por {granularity_es_avg.title()}"; period_col_display = period_col; xaxis_opts = {}
            try:
                if comp_granularity == "hourly": averages_df[period_col] = pd.to_numeric(averages_df[period_col], errors='coerce').astype(int); averages_df = averages_df.sort_values(by=[period_col, 'city']); x_label = "Hora del día (0-23)"
                elif comp_granularity == "daily": averages_df[period_col] = pd.to_datetime(averages_df[period_col], errors='coerce'); averages_df = averages_df.sort_values(by=[period_col, 'city']); x_label = "Fecha"
                elif comp_granularity == "weekly":
                    # Mapea el número ISO del día de la semana (1=Lun) a su nombre en español.
                    averages_df[period_col] = pd.to_numeric(averages_df[period_col], errors='coerce').astype(int); day_map = {1: 'Lun', 2: 'Mar', 3: 'Mié', 4: 'Jue', 5: 'Vie', 6: 'Sáb', 7: 'Dom'}
                    averages_df['day_name'] = averages_df[period_col].map(day_map); averages_df = averages_df.sort_values(by=[period_col, 'city']); period_col_display = 'day_name'; x_label = "Día de la Semana"; xaxis_opts = {'type': 'category', 'categoryorder':'array', 'categoryarray':['Lun','Mar','Mié','Jue','Vie','Sáb','Dom']}
                elif comp_granularity == "monthly":
                    # Mapea el número del mes (1=Ene) a su nombre abreviado en español.
                    averages_df[period_col] = pd.to_numeric(averages_df[period_col], errors='coerce').astype(int); month_map = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
                    averages_df['month_name'] = averages_df[period_col].map(month_map); averages_df = averages_df.sort_values(by=[period_col, 'city']); period_col_display = 'month_name'; x_label = "Mes"; xaxis_opts = {'type': 'category', 'categoryorder':'array', 'categoryarray':['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']}
                else: averages_df = averages_df.sort_values(by=[period_col, 'city'])
            except Exception as e: st.error(f"Error preparando datos comparativos: {e}"); return

            if not averages_df.empty:
                 color_discrete_map = {city: color for city, color in zip(averages_df['city'].unique(), px.colors.qualitative.Plotly)}
                 # Usa gráficos de línea para granularidad horaria/diaria (continua) y barras para semanal/mensual (categórica).
                 if comp_granularity in ["hourly", "daily"]: fig = px.line(averages_df, x=period_col_display, y=average_col, color='city', title=title, labels={period_col_display: x_label, average_col: y_label, 'city': 'Ciudad'}, markers=True, color_discrete_map=color_discrete_map)
                 elif comp_granularity in ["weekly", "monthly"]: fig = px.bar(averages_df, x=period_col_display, y=average_col, color='city', title=title, labels={period_col_display: x_label, average_col: y_label, 'city': 'Ciudad'}, barmode='group', color_discrete_map=color_discrete_map)
                 else: fig = None
                 if fig:
                     fig.update_traces(hovertemplate=f"<b>Ciudad</b>: %{{customdata[0]}}<br><b>{x_label}</b>: %{{x}}<br><b>{y_label}</b>: %{{y:.2f}}<extra></extra>")
                     fig.update_layout(xaxis=xaxis_opts, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                     st.plotly_chart(fig, use_container_width=True)
            else: st.info("No hay datos válidos para la comparación.")
    elif averages_df.empty and averages_data is not None: st.info(f"No se encontraron promedios comparativos para {', '.join(comp_cities)}.")

def display_distribution_chart(city: str, start_date: datetime.date, end_date: datetime.date):
    """Muestra histograma y box plot para una variable."""
    st.subheader(f"📊 Variabilidad del Clima en {city}")
    selected_dist_var_es = st.selectbox("Variable a Analizar", options=list(VARIABLE_MAP_ES.values()), index=0, key=f"dist_var_es_{city}")
    dist_var = [k for k, v in VARIABLE_MAP_ES.items() if v == selected_dist_var_es][0]

    st.caption(f"Analiza qué tan variable fue la **{selected_dist_var_es}** en **{city}**. El histograma muestra valores comunes, el diagrama de caja resume la dispersión.")

    endpoint = f"/weather/{city}/trends"
    params = {"variable_name": dist_var, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
    with st.spinner(f"Cargando datos de '{selected_dist_var_es}' para análisis en {city}..."):
        dist_data = fetch_api_data(endpoint, params, request_desc=f"datos para distribución en {city}")
        dist_df = process_data_for_plotting(dist_data, time_col='time', value_col=dist_var)

    if not dist_df.empty and dist_var in dist_df.columns:
         plot_df = dist_df.dropna(subset=[dist_var])
         if not plot_df.empty:
             with st.spinner("Generando gráficos de distribución..."):
                 col1, col2 = st.columns(2)
                 with col1:
                     st.markdown("###### Frecuencia (Histograma)")
                     fig_hist = px.histogram(plot_df, x=dist_var, title=f"Histograma: {selected_dist_var_es}", labels={dist_var: selected_dist_var_es}, nbins=30)
                     fig_hist.update_layout(bargap=0.1); st.plotly_chart(fig_hist, use_container_width=True)
                 with col2:
                     st.markdown("###### Dispersión (Diagrama Caja)")
                     fig_box = px.box(plot_df, y=dist_var, title=f"Diagrama Caja: {selected_dist_var_es}", labels={dist_var: selected_dist_var_es}, points='outliers')
                     st.plotly_chart(fig_box, use_container_width=True)
         else: st.info(f"No hay valores válidos de '{selected_dist_var_es}' para analizar.")
    elif dist_df.empty and dist_data is not None: st.info(f"No se encontraron datos para analizar la distribución de '{selected_dist_var_es}'.")

def display_full_history(city: str):
    """Muestra la tabla del historial completo en un expander."""
    st.subheader(f"📚 Tabla de Datos Históricos Detallados para {city}")
    with st.expander("Ver/Ocultar Tabla Completa", expanded=False):
        st.caption(f"Consulta todos los registros climáticos históricos disponibles para **{city}**")

        endpoint = f"/weather/{city}"
        with st.spinner(f"Cargando historial completo para {city}..."):
            history_data = fetch_api_data(endpoint, request_desc=f"historial completo para {city}")
            history_df = process_data_for_plotting(history_data, time_col='time', parse_dates=True)

        if not history_df.empty:
            with st.spinner("Preparando tabla de historial..."):
                history_df_display = history_df.copy()
                if 'time' in history_df_display.columns and pd.api.types.is_datetime64_any_dtype(history_df_display['time']):
                    history_df_display = history_df_display.sort_values('time', ascending=False)
                    history_df_display['time'] = history_df_display['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                display_df = history_df_display.rename(columns=VARIABLE_MAP_ES).rename(columns={'time': 'Fecha y Hora'})
                st.info(f"Mostrando las últimas {min(100, len(display_df))} de {len(display_df)} entradas.")
                st.dataframe(display_df.head(100), use_container_width=True, height=300)
                csv_data = convert_df_to_csv(history_df)
                st.download_button(label="Descargar Historial Completo (CSV)", data=csv_data, file_name=f"{city}_historial_climatico_{datetime.now().strftime('%Y%m%d')}.csv", mime='text/csv', key=f"download_csv_{city}")
        elif history_df.empty and history_data is not None: st.info(f"No hay datos históricos disponibles para {city}.")

# --- Aplicación Principal Streamlit (Scrolling Layout) ---

st.title("Análisis Climático de Colombia")

st.markdown("""
Este proyecto se enfoca en el análisis y visualización de datos meteorológicos de 10 ciudades principales de Colombia, además de Leticia, incluida por interés en su particular comportamiento climático. Para la adquisición de datos, recurrimos a la API gratuita de Open-Meteo, extrayendo el historial disponible desde el 2 de febrero de 2022 para las variables: `time`, `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `precipitation`, `wind_speed_10m`, `uv_index`, `pressure_msl`, `shortwave_radiation` y `cloud_cover`. Inicialmente, descargamos manualmente la información para cada ciudad desde el sitio web de Open-Meteo y la consolidamos en un único archivo CSV.

Posteriormente, diseñamos la estructura de la base de datos en PostgreSQL e importamos los datos desde el archivo CSV unificado. El siguiente paso fue desarrollar el servidor backend utilizando Python y FastAPI, estableciendo la conexión con la base de datos y definiendo los endpoints necesarios para las consultas. Dentro de cada endpoint, se implementaron las consultas SQL específicas para recuperar la información requerida. Para la interfaz de usuario, empleamos Streamlit, aprovechando su facilidad para construir el frontend y generar gráficos interactivos con Plotly. Finalmente, se implementó un script en Python que interactúa con la API de Open-Meteo para recopilar datos actualizados e insertarlos en la base de datos, asegurando así que el dashboard refleje la información más reciente.

La comprensión detallada del clima en Colombia es crucial para diversos sectores. Permite optimizar la planificación agrícola, gestionar recursos hídricos, prepararse para eventos climáticos extremos, diseñar infraestructura resiliente y proteger la rica biodiversidad del país. Este análisis proporciona información valiosa para la toma de decisiones informadas que pueden mitigar riesgos y fomentar el desarrollo sostenible.

**Desarrollado por:**
- Sebastian Ibañez ([GitHub](https://github.com/sebdavid3))
- Daniel Cruzado ([GitHub](https://github.com/AlexDanii))
""")
st.markdown("---")

st.sidebar.header("Controles Globales")
selected_city = st.sidebar.selectbox(
    "1. Selecciona Ciudad Principal", options=AVAILABLE_CITIES, index=AVAILABLE_CITIES.index("Bogota"),
    help="Ciudad usada para gráficos individuales."
)
st.sidebar.subheader("2. Selecciona Rango de Fechas")
today = datetime.now(COLOMBIA_TZ).date()
default_global_start = today - timedelta(days=29); default_global_end = today
global_start_date = st.sidebar.date_input("Fecha Inicio", value=default_global_start, max_value=default_global_end, key="global_start", help="Fecha inicial (inclusive).")
global_end_date = st.sidebar.date_input("Fecha Fin", value=default_global_end, min_value=global_start_date, max_value=today, key="global_end", help="Fecha final (inclusive).")
st.sidebar.markdown("---")
st.sidebar.info(f"API simulada en: `{API_BASE_URL}`")
st.sidebar.caption("Proyecto Bases de Datos")
st.sidebar.caption("Creado por: Sebastian Ibañez & Daniel Cruzado")

st.markdown(f"### Análisis para: **{selected_city}** | Período: **{global_start_date.strftime('%d-%b-%Y')}** a **{global_end_date.strftime('%d-%b-%Y')}**")
st.markdown("---")

display_trends_chart(selected_city, global_start_date, global_end_date)
st.divider()

display_precipitation_summary(selected_city, global_start_date, global_end_date)
st.divider()

display_period_summary_map_section(global_start_date, global_end_date)
st.divider()

display_correlation_scatter(selected_city, global_start_date, global_end_date)
st.divider()

display_comparative_averages(global_start_date, global_end_date)
st.divider()

display_distribution_chart(selected_city, global_start_date, global_end_date)
st.divider()

display_full_history(selected_city)

st.markdown("---")
st.caption("Fin del Dashboard.")