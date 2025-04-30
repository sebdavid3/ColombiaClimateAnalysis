import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime

# Configuración de la página
st.set_page_config(layout="wide", page_title="Dashboard Climático Colombia")
API_URL = "http://127.0.0.1:8000"

# Coordenadas de todas las ciudades
CITY_COORDS = {
    "Barranquilla": {"coords": [10.9639, -74.7964], "color": "#FF5733"},
    "Bogota": {"coords": [4.7110, -74.0721], "color": "#3386FF"},
    "Bucaramanga": {"coords": [7.1139, -73.1198], "color": "#33FF57"},
    "Cali": {"coords": [3.4516, -76.5320], "color": "#F033FF"},
    "Cartagena": {"coords": [10.3910, -75.4794], "color": "#FF33A8"},
    "Cucuta": {"coords": [7.8939, -72.5078], "color": "#33FFF6"},
    "Ibague": {"coords": [4.4447, -75.2422], "color": "#FFD733"},
    "Leticia": {"coords": [-4.2153, -69.9406], "color": "#8C33FF"},
    "Medellin": {"coords": [6.2442, -75.5812], "color": "#33FFBD"},
    "Pereira": {"coords": [4.8143, -75.6946], "color": "#FF33F6"},
    "Santa_Marta": {"coords": [11.2408, -74.1990], "color": "#33A8FF"}
}

# Función para obtener datos
@st.cache_data
def load_all_data():
    all_data = []
    for city in CITY_COORDS.keys():
        try:
            response = requests.get(f"{API_URL}/weather/{city}", timeout=3)
            if response.status_code == 200:
                df = pd.DataFrame(response.json())
                if not df.empty:
                    df['time'] = pd.to_datetime(df['time'])
                    df['Ciudad'] = city
                    df['latitude'] = CITY_COORDS[city]["coords"][0]
                    df['longitude'] = CITY_COORDS[city]["coords"][1]
                    all_data.append(df)
        except:
            continue
    return pd.concat(all_data) if all_data else pd.DataFrame()

# Función para crear el mapa
def create_map(df, selected_city=None):
    m = folium.Map(location=[6.0, -73.0], zoom_start=5)
    if df.empty:
        return m
    
    last_records = df.sort_values('time').groupby('Ciudad').last().reset_index()
    
    for _, row in last_records.iterrows():
        city = row['Ciudad']
        if city in CITY_COORDS:
            is_selected = selected_city and city == selected_city
            marker_color = CITY_COORDS[city]["color"]
            
            popup_html = f"""
            <div style="width: 200px">
                <h4>{city.replace('_', ' ')}</h4>
                <p><b>Temp:</b> {row.get('temperature_2m', 'N/A')}°C</p>
                <p><b>Humedad:</b> {row.get('relative_humidity_2m', 'N/A')}%</p>
                <p><b>Viento:</b> {row.get('wind_speed_10m', 'N/A')} km/h</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=10 if is_selected else 7,
                popup=popup_html,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.8,
                weight=2 if is_selected else 1
            ).add_to(m)
    return m

# --- Interfaz ---
st.title("🌦 Dashboard Climático Colombia")

# Cargar datos
df_all = load_all_data()
available_cities = sorted(df_all['Ciudad'].unique().tolist()) if not df_all.empty else list(CITY_COORDS.keys())

# Sección superior: Dos columnas
col1, col2 = st.columns([0.5, 0.5])

with col1:
    st.subheader("📈 Tendencias Climáticas")
    selected_city = st.selectbox("Seleccionar ciudad", available_cities)
    
    if not df_all.empty:
        df_city = df_all[df_all['Ciudad'] == selected_city]
        
        variable = st.selectbox(
            "Variable a visualizar",
            ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
            format_func=lambda x: {
                "temperature_2m": "Temperatura (°C)",
                "relative_humidity_2m": "Humedad Relativa (%)",
                "wind_speed_10m": "Velocidad del Viento (km/h)"
            }[x]
        )
        
        fig_line = px.line(
            df_city,
            x="time",
            y=variable,
            title=f"Tendencia de {variable.replace('_', ' ').title()} en {selected_city.replace('_', ' ')}",
            labels={"time": "Fecha", variable: variable}
        )
        st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.subheader("🗺️ Mapa de Ciudades")
    if not df_all.empty:
        mapa = create_map(df_all, selected_city)
        st_folium(mapa, width=700, height=500)

# --- Nueva sección: Gráfico de Radar ---
st.subheader("📊 Comparación Multivariable entre Ciudades")

if not df_all.empty:
    # Obtener últimos registros por ciudad
    last_records = df_all.sort_values('time').groupby('Ciudad').last().reset_index()
    
    # Seleccionar ciudades a comparar
    selected_cities = st.multiselect(
        "Selecciona ciudades para comparar",
        options=available_cities,
        default=available_cities[:3] if len(available_cities) >= 3 else available_cities
    )
    
    if selected_cities:
        # Variables a incluir en el radar
        radar_vars = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "uv_index"]
        available_vars = [v for v in radar_vars if v in df_all.columns]
        
        # Preparar datos para el radar
        radar_data = []
        for city in selected_cities:
            city_data = last_records[last_records['Ciudad'] == city].iloc[0]
            values = [city_data.get(var, 0) for var in available_vars]
            radar_data.append(
                go.Scatterpolar(
                    r=values,
                    theta=[var.replace('_', ' ').title() for var in available_vars],
                    fill='toself',
                    name=city.replace('_', ' '),
                    line_color=CITY_COORDS[city]["color"]
                )
            )
        
        # Crear gráfico de radar
        fig_radar = go.Figure(data=radar_data)
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(last_records[available_vars].max().max(), 10) if available_vars else [0, 100]]
            )),
            showlegend=True,
            title="Comparación de Variables Climáticas",
            height=600
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("Selecciona al menos una ciudad para comparar")
else:
    st.warning("No hay datos disponibles para generar la comparación")

# Panel de estado
with st.expander("🔍 Estado de los datos"):
    if not df_all.empty:
        st.success(f"Datos cargados para {len(available_cities)} ciudades")
        st.dataframe(last_records[['Ciudad', 'time'] + available_vars].sort_values('Ciudad'))
    else:
        st.error("No se encontraron datos")