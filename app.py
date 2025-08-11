import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz 
from dateutil.relativedelta import relativedelta 
import subprocess
import os
import sys 

# --- Configuración Inicial y Constantes ---

st.set_page_config(page_title="Colombia Weather Dashboard", layout="wide")

# Constantes definidas en los requisitos
def _get_api_base_url() -> str:
    # Prefer Streamlit secrets if present (Streamlit Cloud)
    try:
        if "API_BASE_URL" in st.secrets:
            return st.secrets["API_BASE_URL"].rstrip("/")
    except Exception:
        pass
    # Fallback to environment variable
    env_val = os.getenv("API_BASE_URL")
    if env_val:
        return env_val.rstrip("/")
    # Local default
    return "http://localhost:8000"

API_BASE_URL = _get_api_base_url()
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

VARIABLE_MAP_ES = {
    "temperature_2m": "Temperature (°C)",
    "relative_humidity_2m": "Relative Humidity (%)",
    "dew_point_2m": "Dew Point (°C)",
    "precipitation": "Precipitation (mm)",
    "wind_speed_10m": "Wind Speed (m/s)",
    "uv_index": "UV Index (0-11+)",
    "pressure_msl": "Sea Level Pressure (hPa)",
    "shortwave_radiation": "Shortwave Radiation (W/m²)",
    "cloud_cover": "Cloud Cover (%)"
}

THRESHOLDS = {
    "temperature_2m": {"Extreme Heat": 32, "Hot": 28, "Cold": 10}
    # Add other thresholds if needed
}
COLOMBIA_TZ = pytz.timezone('America/Bogota')

# --- Funciones Auxiliares (Llamadas API y Procesamiento) ---

@st.cache_data(ttl=300)
def fetch_api_data(endpoint: str, params: dict = None, request_desc: str = "data"):
    """Perform a GET call to the API and handle common errors."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        try:
            error_detail = http_err.response.json().get("detail", http_err.response.text)
        except Exception:
            error_detail = http_err.response.text
        if status_code == 404:
            st.warning(f"No {request_desc} found for: {endpoint} (params: {params}). (404)")
        elif status_code == 422:
            st.warning(f"Invalid parameters for {request_desc}: {endpoint} (params: {params}). Details: {error_detail} (422)")
        elif status_code >= 500:
            st.error(f"Server error ({status_code}) requesting {request_desc}: {endpoint}. Details: {error_detail}")
        else:
            st.error(f"HTTP error ({status_code}) loading {request_desc} from {endpoint}: {error_detail}")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"API connection error ({request_desc}) at {url}: {conn_err}")
        return None
    except requests.exceptions.Timeout as timeout_err:
        st.error(f"Timeout for {request_desc} at {url}: {timeout_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        st.error(f"Unexpected network error ({request_desc}) at {url}: {req_err}")
        return None

def process_data_for_plotting(
    data: list | dict | None,
    time_col: str | None = 'time',
    value_col: str | None = None,
    parse_dates: bool = True
) -> pd.DataFrame:
    """Convert API JSON response into a Pandas DataFrame safely."""
    if data is None:
        return pd.DataFrame()

    if isinstance(data, dict):
        # Dict of dicts keyed by city
        if all(isinstance(v, dict) for v in data.values()):
            df = (
                pd.DataFrame.from_dict(data, orient='index')
                .reset_index()
                .rename(columns={'index': 'city'})
            )
            if parse_dates and time_col and time_col in df.columns:
                try:
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                except Exception:
                    pass
            if value_col and value_col in df.columns:
                df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            if parse_dates and time_col and time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df = df.dropna(subset=[time_col])
            return df
        # Non-processable dict
        return pd.DataFrame()

    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if parse_dates and time_col and time_col in df.columns:
            try:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                    df = df.dropna(subset=[time_col])
            except Exception:
                pass
        if value_col and value_col in df.columns:
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        return df

    # Unexpected format
    return pd.DataFrame()

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode('utf-8')

def get_last_update_info():
    """Get information about the last data update from the API."""
    try:
        # Usar el endpoint de estado de la API
        endpoint = "/update-weather/status"
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            status_map = {
                "updated": ("Updated", "green"),
                "recent": ("Recent", "orange"), 
                "outdated": ("Outdated", "red"),
                "no_data": ("No data", "gray")
            }
            
            status_info = data.get("status", "unknown")
            status_text, status_color = status_map.get(status_info, ("Unknown", "gray"))
            
            last_update = data.get("last_update")
            if last_update:
                # Convert ISO string to datetime and format
                try:
                    dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                    # Convert to Colombia time
                    colombia_tz = pytz.timezone('America/Bogota')
                    if dt.tzinfo is None:
                        dt = colombia_tz.localize(dt)
                    else:
                        dt = dt.astimezone(colombia_tz)
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    formatted_time = "N/A"
            else:
                formatted_time = "N/A"
            
            return status_text, formatted_time, status_color
    except Exception as e:
        print(f"Error getting status: {e}")
    
    return "Unknown", "N/A", "gray"

def is_running_locally():
    """Detect if the application is running locally."""
    try:
    # Check if we are in a local development environment
        import socket
        hostname = socket.gethostname()
        
    # Local environment indicators
        local_indicators = [
            "localhost" in API_BASE_URL.lower(),
            "127.0.0.1" in API_BASE_URL,
            ":8000" in API_BASE_URL,
            hostname.lower() in ["localhost", "127.0.0.1"],
            os.path.exists(os.path.join(os.path.dirname(__file__), "scripts", "update_weather.py"))
        ]
        
        return any(local_indicators)
    except:
        return False

def update_weather_data_local():
    """Run the local weather update script via subprocess."""
    try:
        # Path to the update script
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "update_weather.py")
        
        # Verify that the script exists
        if not os.path.exists(script_path):
            return False, f"Script not found at: {script_path}"
        
        # Execute the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos de timeout
        )
        
        if result.returncode == 0:
            return True, "Weather data updated successfully"
        else:
            return False, f"Error running script: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "The update process exceeded the time limit (5 minutes)"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def update_weather_data_remote():
    """Trigger data update via the remote API."""
    try:
        # Call the update endpoint
        endpoint = "/update-weather"
        response = requests.post(f"{API_BASE_URL}{endpoint}", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return True, data.get("message", "Update started in background")
        else:
            return False, f"HTTP error {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        return False, "Timeout while triggering remote update"
    except Exception as e:
        return False, f"Error connecting to API: {str(e)}"

def update_weather_data():
    """Run the weather update using the appropriate method based on the environment."""
    if is_running_locally():
        return update_weather_data_local()
    else:
        return update_weather_data_remote()

# Trigger data update automatically on page load (no manual buttons)
try:
    _ok, _msg = update_weather_data()
    # Optionally clear cached API responses so fresh data is fetched
    st.cache_data.clear()
except Exception:
    pass

# --- Componentes del Dashboard ---

def display_trends_chart(city: str, start_date: datetime.date, end_date: datetime.date):
    """Show the trends section."""
    st.subheader(f"Weather Trends in {city}")
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_trend_var_es = st.selectbox(
            "Select Variable",
            options=list(VARIABLE_MAP_ES.values()),
            index=0,
            key=f"trend_var_{city}"
        )
        trend_var = [k for k, v in VARIABLE_MAP_ES.items() if v == selected_trend_var_es][0]
    with col2:
        compare_yoy = st.checkbox("Compare Previous Year", key=f"trend_yoy_{city}", value=False, help="Overlay data from the same period last year.")

    st.caption(f"See how **{selected_trend_var_es}** changed in **{city}** over the selected period, optionally compared with last year.")

    endpoint = f"/weather/{city}/trends"
    current_params = {"variable_name": trend_var, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
    prev_trends_df = pd.DataFrame()
    prev_start_date = None

    with st.spinner(f"Loading '{selected_trend_var_es}' data for {city}..."):
        trends_data = fetch_api_data(endpoint, current_params, request_desc=f"current trends for {city}")
        trends_df = process_data_for_plotting(trends_data, time_col='time', value_col=trend_var)

    if compare_yoy and not trends_df.empty:
        try:
            prev_start_date = start_date - relativedelta(years=1)
            prev_end_date = end_date - relativedelta(years=1)
            prev_params = {"variable_name": trend_var, "start_date": prev_start_date.strftime('%Y-%m-%d'), "end_date": prev_end_date.strftime('%Y-%m-%d')}
            with st.spinner(f"Loading previous year data ({prev_start_date.year})..."):
                prev_trends_data = fetch_api_data(endpoint, prev_params, request_desc=f"previous year trends for {city}")
                prev_trends_df = process_data_for_plotting(prev_trends_data, time_col='time', value_col=trend_var)
                if not prev_trends_df.empty and pd.api.types.is_datetime64_any_dtype(prev_trends_df['time']):
                     # Desplaza el eje de tiempo del año anterior para alinearlo con el año actual en el gráfico.
                     prev_trends_df['time_shifted'] = prev_trends_df['time'].apply(lambda d: d + relativedelta(years=1))
                else: prev_trends_df = pd.DataFrame()
        except Exception as e: st.error(f"Error getting previous year data: {e}"); prev_trends_df = pd.DataFrame()

    if not trends_df.empty and trend_var in trends_df.columns:
        with st.spinner("Building trend chart..."):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trends_df['time'], y=trends_df[trend_var], mode='lines+markers', name=f'{selected_trend_var_es} ({start_date.year})', hovertemplate=f"<b>Date ({start_date.year})</b>: %{{x|%d-%b %H:%M}}<br><b>Value</b>: %{{y}}<extra></extra>"))
            if compare_yoy and not prev_trends_df.empty and 'time_shifted' in prev_trends_df.columns and prev_start_date:
                 fig.add_trace(go.Scatter(x=prev_trends_df['time_shifted'], y=prev_trends_df[trend_var], mode='lines', name=f'{selected_trend_var_es} ({prev_start_date.year})', line=dict(dash='dash', color='grey'), opacity=0.7, hovertemplate=f"<b>Date ({prev_start_date.year}, aligned to {start_date.year})</b>: %{{x|%d-%b %H:%M}}<br><b>Value</b>: %{{y}}<extra></extra>"))
            if trend_var in THRESHOLDS:
                 for label, value in THRESHOLDS[trend_var].items(): fig.add_hline(y=value, line_dash="dash", line_color="red", opacity=0.6, annotation_text=label, annotation_position="bottom right")
            fig.update_layout(title=f"Trend of {selected_trend_var_es} in {city}", xaxis_title="Date & Time", yaxis_title=selected_trend_var_es, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
    elif trends_df.empty and trends_data is not None:
        st.info(f"No '{selected_trend_var_es}' data found for {city} in the selected period.")

def display_precipitation_summary(city: str, start_date: datetime.date, end_date: datetime.date):
    """Show the precipitation summary section."""
    st.subheader(f"Rainfall Summary in {city}")
    granularity = st.selectbox("Group Rainfall by:", options=["daily", "weekly", "monthly"], index=0, key=f"precip_granularity_{city}")

    st.caption(f"View the total accumulated rainfall ({granularity}) in **{city}**. Taller bars indicate more rain.")

    endpoint = f"/weather/{city}/precipitation/summary"
    params = {"granularity": granularity, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
    with st.spinner(f"Loading rainfall summary by {granularity}..."):
        precip_data = fetch_api_data(endpoint, params, request_desc=f"rainfall summary for {city}")
        precip_df = process_data_for_plotting(precip_data, time_col='period_start', value_col='total_precipitation')

    if not precip_df.empty and 'total_precipitation' in precip_df.columns:
        with st.spinner("Building rainfall chart..."):
            if 'period_start' in precip_df.columns and not pd.api.types.is_datetime64_any_dtype(precip_df['period_start']): precip_df['period_start'] = pd.to_datetime(precip_df['period_start'], errors='coerce')
            precip_df = precip_df.dropna(subset=['period_start', 'total_precipitation']).sort_values('period_start')
            if granularity == 'daily': precip_df['period_label'] = precip_df['period_start'].dt.strftime('%d-%b-%Y')
            elif granularity == 'weekly': precip_df['period_label'] = precip_df['period_start'].dt.strftime('Week %U (%Y)')
            elif granularity == 'monthly': precip_df['period_label'] = precip_df['period_start'].dt.strftime('%b %Y')
            else: precip_df['period_label'] = precip_df['period_start'].astype(str)
            granularity_title = {"daily": "Daily", "weekly": "Weekly", "monthly": "Monthly"}.get(granularity, granularity.title())
            fig = px.bar(precip_df, x='period_label', y='total_precipitation', title=f"Accumulated Rainfall ({granularity_title}) in {city}", labels={'period_label': f'Period ({granularity_title})', 'total_precipitation': 'Total Rain (mm)'} )
            fig.update_traces(hovertemplate="<b>Period</b>: %{x}<br><b>Rain</b>: %{y} mm<extra></extra>")
            fig.update_layout(xaxis={'type': 'category'})
            st.plotly_chart(fig, use_container_width=True)
    elif precip_df.empty and precip_data is not None:
        st.info(f"No rainfall data found for {city} with the selected grouping.")

@st.cache_data(ttl=600)
def fetch_and_process_period_map_data(variable_key: str, start_date: datetime.date, end_date: datetime.date):
    """Fetch and aggregate data (average/total) for the period map."""
    city_data_agg = {}
    variable_es = VARIABLE_MAP_ES.get(variable_key, variable_key)
    aggregation_type = "Average"
    with st.spinner(f"Calculating {'totals' if variable_key == 'precipitation' else 'averages'} of '{variable_es}' for the map ({start_date.strftime('%d-%b')} to {end_date.strftime('%d-%b-%Y')})..."):
        if variable_key == "precipitation":
            # For precipitation we sum over the period, not average.
            aggregation_type = "Total"
            city_totals = {city: 0.0 for city in AVAILABLE_CITIES}
            for city_name in AVAILABLE_CITIES:
                endpoint = f"/weather/{city_name}/precipitation/summary"
                params = {"granularity": 'daily', "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
                city_precip_data = fetch_api_data(endpoint, params, request_desc=f"daily precip for {city_name}")
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
            # For other variables, compute the overall period average.
            aggregation_type = "Average"
            cities_str = ",".join(AVAILABLE_CITIES)
            endpoint = "/weather/averages"; average_col_name = f"average_{variable_key}"
            params = {"variable_name": variable_key, "granularity": 'daily', "cities": cities_str, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
            averages_data = fetch_api_data(endpoint, params, request_desc=f"daily averages for map")
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
            map_df['periodo'] = f"{start_date.strftime('%d-%b')} to {end_date.strftime('%d-%b-%Y')}"
            map_df['tipo_agregacion'] = aggregation_type
            map_df = map_df.dropna(subset=['lat', 'lon', 'valor_mapa'])
            return map_df if not map_df.empty else None, aggregation_type
        except Exception: return None, aggregation_type

def display_period_summary_map_section(start_date: datetime.date, end_date: datetime.date):
    """Show the map of averages/totals over the selected period using global dates provided."""
    st.subheader("Weather Summary Map for Period")

    selected_variable_es = st.selectbox(
        "Variable to Map",
        options=list(VARIABLE_MAP_ES.values()),
        index=0,
        key="map_period_var"
    )
    variable_key = [k for k, v in VARIABLE_MAP_ES.items() if v == selected_variable_es][0]

    is_precip = variable_key == 'precipitation'
    aggregation_type_text = "Total" if is_precip else "Average"
    st.caption(f"Compare the **{aggregation_type_text.lower()}** of **{selected_variable_es}** across cities for the selected period ({start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}). Use the color legend to interpret.")

    map_df, aggregation_type = fetch_and_process_period_map_data(variable_key, start_date, end_date)

    if map_df is not None and not map_df.empty:
        with st.spinner("Building summary map..."):
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
            title=f"Summary Map: {selected_variable_es} ({aggregation_type})",
            labels={"valor_mapa": f"{aggregation_type_text}"}
                )
                # El hovertemplate referencia los datos en hover_data por su índice (customdata[0], customdata[1], etc.).
                fig.update_traces(hovertemplate="<br>".join([
            "<b>City:</b> %{hovertext}",
            "<b>Variable:</b> %{customdata[0]}",
            "<b>Type:</b> %{customdata[1]}",
            f"<b>Value:</b> %{{customdata[2]:.2f}}",
            "<b>Period:</b> %{customdata[3]}<extra></extra>"
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
                st.error(f"Error building period map: {e}")
    else:
        st.info("No data available to show on the summary map with the selected parameters.")

def display_correlation_scatter(city: str, start_date: datetime.date, end_date: datetime.date):
    """Show the correlation section."""
    st.subheader(f"Relationship Between Variables in {city}")
    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        selected_var_x_es = st.selectbox("X-axis Variable", options=list(VARIABLE_MAP_ES.values()), index=0, key=f"corr_var_x_{city}")
        corr_var_x = [k for k, v in VARIABLE_MAP_ES.items() if v == selected_var_x_es][0]
    with col2:
        available_y_vars_es = [v for k, v in VARIABLE_MAP_ES.items() if k != corr_var_x]
        default_y_val_es = VARIABLE_MAP_ES.get("relative_humidity_2m", available_y_vars_es[0])
        selected_var_y_es = st.selectbox("Y-axis Variable", options=available_y_vars_es, index=available_y_vars_es.index(default_y_val_es) if default_y_val_es in available_y_vars_es else 0, key=f"corr_var_y_{city}")
        corr_var_y = [k for k, v in VARIABLE_MAP_ES.items() if v == selected_var_y_es][0]
    with col3: show_trendline = st.checkbox("Show Trendline", key=f"corr_trend_{city}", value=True, help="Draw a line showing overall trend.")

    st.caption(f"Explore whether **{selected_var_x_es}** and **{selected_var_y_es}** move together (correlation) in **{city}**. Use the coefficient below to measure strength.")

    endpoint = f"/weather/{city}/correlation"
    params = {"variable_x": corr_var_x, "variable_y": corr_var_y, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
    with st.spinner(f"Loading correlation data for {city}..."):
        correlation_data = fetch_api_data(endpoint, params, request_desc=f"correlation for {city}")
        correlation_df = process_data_for_plotting(correlation_data, time_col='time', parse_dates=True)

    corr_value = None
    if not correlation_df.empty and corr_var_x in correlation_df.columns and corr_var_y in correlation_df.columns:
        with st.spinner("Building correlation chart..."):
            correlation_df[corr_var_x] = pd.to_numeric(correlation_df[corr_var_x], errors='coerce')
            correlation_df[corr_var_y] = pd.to_numeric(correlation_df[corr_var_y], errors='coerce')
            plot_df = correlation_df.dropna(subset=[corr_var_x, corr_var_y])
            if not plot_df.empty and len(plot_df) > 1:
                fig = px.scatter(
                    plot_df,
                    x=corr_var_x,
                    y=corr_var_y,
                    title=f"Relationship between {selected_var_y_es} and {selected_var_x_es}",
                    labels={corr_var_x: selected_var_x_es, corr_var_y: selected_var_y_es},
                    trendline="ols" if show_trendline else None,
                    trendline_color_override='red',
                    hover_data={'time': '|%d-%b %H:%M'}
                )
                fig.update_traces(hovertemplate=f"<b>{selected_var_x_es}</b>: %{{x}}<br><b>{selected_var_y_es}</b>: %{{y}}<br><b>Date</b>: %{{customdata[0]}}<extra></extra>")
                st.plotly_chart(fig, use_container_width=True)
                try:
                    corr_value = plot_df[corr_var_x].corr(plot_df[corr_var_y])
                except Exception:
                    pass
            elif len(plot_df) <= 1:
                st.info("Not enough data to plot/calculate correlation.")
            else:
                st.info("No valid data to plot correlation.")
    elif correlation_df.empty and correlation_data is not None:
        st.info(f"No correlation data found for {city} in the period.")
    if corr_value is not None:
        corr_desc = (
            "Strong Positive" if corr_value > 0.7 else
            "Moderate Positive" if corr_value > 0.4 else
            "Weak Positive" if corr_value > 0.1 else
            "Strong Negative" if corr_value < -0.7 else
            "Moderate Negative" if corr_value < -0.4 else
            "Weak Negative" if corr_value < -0.1 else
            "Very Weak or None"
        )
        st.metric(label="Linear Correlation Coefficient", value=f"{corr_value:.3f}", help=f"Interpretation: {corr_desc}")

def display_comparative_averages(start_date: datetime.date, end_date: datetime.date):
    """Show the comparative averages section."""
    st.subheader("Average Comparison Across Cities")
    col1, col2, col3 = st.columns(3)
    with col1: comp_cities = st.multiselect("Cities to Compare", options=AVAILABLE_CITIES, default=["Barranquilla", "Bogota", "Medellin"], key="comp_cities")
    with col2:
        selected_comp_var_es = st.selectbox("Variable to Compare", options=list(VARIABLE_MAP_ES.values()), index=0, key="comp_var_es")
        comp_var = [k for k, v in VARIABLE_MAP_ES.items() if v == selected_comp_var_es][0]
    with col3: comp_granularity = st.selectbox("Group Averages by", options=["hourly", "daily", "weekly", "monthly"], index=0, key="comp_granularity")
    if not comp_cities: st.warning("Please select at least one city."); return

    granularity_es_avg = {"hourly": "hour", "daily": "day", "weekly": "day of week", "monthly": "month"}.get(comp_granularity, comp_granularity)
    st.caption(f"Compare the average **{selected_comp_var_es}** by **{granularity_es_avg}** across the selected cities.")

    endpoint = "/weather/averages"
    params = {"cities": ",".join(comp_cities), "variable_name": comp_var, "granularity": comp_granularity, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
    with st.spinner(f"Loading comparative averages by {comp_granularity}..."):
        averages_data = fetch_api_data(endpoint, params, request_desc="comparative averages")
        period_key_map = {"hourly": "hour", "daily": "period_start", "weekly": "day_of_week_iso", "monthly": "month"}
        period_col = period_key_map.get(comp_granularity, "period_key")
        average_col = f"average_{comp_var}"
        averages_df = process_data_for_plotting(averages_data, time_col=None, value_col=average_col, parse_dates=False)

    if not averages_df.empty and period_col in averages_df.columns and average_col in averages_df.columns:
        with st.spinner("Building comparison chart..."):
            averages_df = averages_df.dropna(subset=[average_col])
            x_label = f"Period ({granularity_es_avg.title()})"; y_label = f"Average {selected_comp_var_es}"
            title = f"Average {selected_comp_var_es} by {granularity_es_avg.title()}"; period_col_display = period_col; xaxis_opts = {}
            try:
                if comp_granularity == "hourly": averages_df[period_col] = pd.to_numeric(averages_df[period_col], errors='coerce').astype(int); averages_df = averages_df.sort_values(by=[period_col, 'city']); x_label = "Hour of day (0-23)"
                elif comp_granularity == "daily": averages_df[period_col] = pd.to_datetime(averages_df[period_col], errors='coerce'); averages_df = averages_df.sort_values(by=[period_col, 'city']); x_label = "Date"
                elif comp_granularity == "weekly":
                    # Map ISO day of week (1=Mon) to English abbreviations.
                    averages_df[period_col] = pd.to_numeric(averages_df[period_col], errors='coerce').astype(int); day_map = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
                    averages_df['day_name'] = averages_df[period_col].map(day_map); averages_df = averages_df.sort_values(by=[period_col, 'city']); period_col_display = 'day_name'; x_label = "Day of Week"; xaxis_opts = {'type': 'category', 'categoryorder':'array', 'categoryarray':['Mon','Tue','Wed','Thu','Fri','Sat','Sun']}
                elif comp_granularity == "monthly":
                    # Map month number to English abbreviations.
                    averages_df[period_col] = pd.to_numeric(averages_df[period_col], errors='coerce').astype(int); month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
                    averages_df['month_name'] = averages_df[period_col].map(month_map); averages_df = averages_df.sort_values(by=[period_col, 'city']); period_col_display = 'month_name'; x_label = "Month"; xaxis_opts = {'type': 'category', 'categoryorder':'array', 'categoryarray':['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']}
                else: averages_df = averages_df.sort_values(by=[period_col, 'city'])
            except Exception as e: st.error(f"Error preparing comparative data: {e}"); return

            if not averages_df.empty:
                 color_discrete_map = {city: color for city, color in zip(averages_df['city'].unique(), px.colors.qualitative.Plotly)}
                 # Usa gráficos de línea para granularidad horaria/diaria (continua) y barras para semanal/mensual (categórica).
                 if comp_granularity in ["hourly", "daily"]: fig = px.line(averages_df, x=period_col_display, y=average_col, color='city', title=title, labels={period_col_display: x_label, average_col: y_label, 'city': 'City'}, markers=True, color_discrete_map=color_discrete_map)
                 elif comp_granularity in ["weekly", "monthly"]: fig = px.bar(averages_df, x=period_col_display, y=average_col, color='city', title=title, labels={period_col_display: x_label, average_col: y_label, 'city': 'City'}, barmode='group', color_discrete_map=color_discrete_map)
                 else: fig = None
                 if fig:
                     fig.update_traces(hovertemplate=f"<b>City</b>: %{{customdata[0]}}<br><b>{x_label}</b>: %{{x}}<br><b>{y_label}</b>: %{{y:.2f}}<extra></extra>")
                     fig.update_layout(xaxis=xaxis_opts, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                     st.plotly_chart(fig, use_container_width=True)
            else: st.info("No valid data for comparison.")
    elif averages_df.empty and averages_data is not None: st.info(f"No comparative averages found for {', '.join(comp_cities)}.")

def display_distribution_chart(city: str, start_date: datetime.date, end_date: datetime.date):
    """Show histogram and box plot for a variable."""
    st.subheader(f"Weather Variability in {city}")
    selected_dist_var_es = st.selectbox("Variable to Analyze", options=list(VARIABLE_MAP_ES.values()), index=0, key=f"dist_var_es_{city}")
    dist_var = [k for k, v in VARIABLE_MAP_ES.items() if v == selected_dist_var_es][0]

    st.caption(f"Analyze how variable **{selected_dist_var_es}** was in **{city}**. The histogram shows common values; the box plot summarizes spread.")

    endpoint = f"/weather/{city}/trends"
    params = {"variable_name": dist_var, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
    with st.spinner(f"Loading '{selected_dist_var_es}' data for analysis in {city}..."):
        dist_data = fetch_api_data(endpoint, params, request_desc=f"distribution data in {city}")
        dist_df = process_data_for_plotting(dist_data, time_col='time', value_col=dist_var)

    if not dist_df.empty and dist_var in dist_df.columns:
        plot_df = dist_df.dropna(subset=[dist_var])
        if not plot_df.empty:
            with st.spinner("Building distribution charts..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("###### Frequency (Histogram)")
                    fig_hist = px.histogram(plot_df, x=dist_var, title=f"Histogram: {selected_dist_var_es}", labels={dist_var: selected_dist_var_es}, nbins=30)
                    fig_hist.update_layout(bargap=0.1)
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col2:
                    st.markdown("###### Spread (Box Plot)")
                    fig_box = px.box(plot_df, y=dist_var, title=f"Box Plot: {selected_dist_var_es}", labels={dist_var: selected_dist_var_es}, points='outliers')
                    st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info(f"No valid values of '{selected_dist_var_es}' to analyze.")
    elif dist_df.empty and dist_data is not None:
        st.info(f"No data found to analyze the distribution of '{selected_dist_var_es}'.")

def display_full_history(city: str):
    """Show the full historical data table in an expander."""
    st.subheader(f"Detailed Historical Data Table for {city}")
    with st.expander("Show/Hide Full Table", expanded=False):
        st.caption(f"Browse all available historical weather records for **{city}**")

        endpoint = f"/weather/{city}"
        with st.spinner(f"Loading full history for {city}..."):
            history_data = fetch_api_data(endpoint, request_desc=f"full history for {city}")
            history_df = process_data_for_plotting(history_data, time_col='time', parse_dates=True)

        if not history_df.empty:
            with st.spinner("Preparing history table..."):
                history_df_display = history_df.copy()
                if 'time' in history_df_display.columns and pd.api.types.is_datetime64_any_dtype(history_df_display['time']):
                    history_df_display = history_df_display.sort_values('time', ascending=False)
                    history_df_display['time'] = history_df_display['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                display_df = history_df_display.rename(columns=VARIABLE_MAP_ES).rename(columns={'time': 'Date & Time'})
                st.info(f"Showing the last {min(100, len(display_df))} of {len(display_df)} rows.")
                st.dataframe(display_df.head(100), use_container_width=True, height=300)
                csv_data = convert_df_to_csv(history_df)
                st.download_button(label="Download Full History (CSV)", data=csv_data, file_name=f"{city}_weather_history_{datetime.now().strftime('%Y%m%d')}.csv", mime='text/csv', key=f"download_csv_{city}")
        elif history_df.empty and history_data is not None: st.info(f"No historical data available for {city}.")

# --- Aplicación Principal Streamlit (Scrolling Layout) ---

st.title("Colombia Climate Analysis")

st.markdown("""
This project analyzes and visualizes weather data for major Colombian cities plus Leticia. We use the free Open-Meteo API to retrieve historical data since February 2, 2022 for: `time`, `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `precipitation`, `wind_speed_10m`, `uv_index`, `pressure_msl`, `shortwave_radiation`, and `cloud_cover`.

We designed a PostgreSQL database and imported the data, then built a FastAPI backend to serve the data and a Streamlit frontend using Plotly for interactive visualizations. A Python script updates the database by pulling fresh data from Open-Meteo so the dashboard reflects recent conditions.

Developed by:
- Sebastian Ibañez ([GitHub](https://github.com/sebdavid3))
- Daniel Cruzado ([GitHub](https://github.com/AlexDanii))
""")
st.markdown("---")

st.sidebar.header("Global Controls")

# Environment information
is_local = is_running_locally()
env_info = "Local" if is_local else "Cloud"
st.sidebar.caption(f"Environment: {env_info}")

# Data status
status, last_update, status_color = get_last_update_info()
st.sidebar.markdown(f"**Status:** {status}")
st.sidebar.caption(f"Last update: {last_update}")
st.sidebar.caption("Open-Meteo API")

st.sidebar.markdown("---")

selected_city = st.sidebar.selectbox(
    "1. Select Main City", options=AVAILABLE_CITIES, index=AVAILABLE_CITIES.index("Bogota"),
    help="City used for individual charts."
)
st.sidebar.subheader("2. Select Date Range")
today = datetime.now(COLOMBIA_TZ).date()
default_global_start = today - timedelta(days=29); default_global_end = today
global_start_date = st.sidebar.date_input("Start Date", value=default_global_start, max_value=default_global_end, key="global_start", help="Start date (inclusive).")
global_end_date = st.sidebar.date_input("End Date", value=default_global_end, min_value=global_start_date, max_value=today, key="global_end", help="End date (inclusive).")
st.sidebar.markdown("---")
st.sidebar.info(f"API base URL: `{API_BASE_URL}`")
st.sidebar.caption("Database Project")
st.sidebar.caption("Created by: Sebastian Ibañez & Daniel Cruzado")

st.markdown(f"### Analysis for: **{selected_city}** | Period: **{global_start_date.strftime('%d-%b-%Y')}** to **{global_end_date.strftime('%d-%b-%Y')}**")
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

# Data update info section removed per request

st.markdown("---")
st.caption("End of Dashboard.")