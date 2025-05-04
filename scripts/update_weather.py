import asyncio
import asyncpg
import httpx
import logging
from datetime import datetime, timedelta, timezone, date, time as dt_time
from typing import List, Dict, Any, Optional, Tuple
import sys
import os
from dotenv import load_dotenv
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
try:
    from zoneinfo import ZoneInfo # Python 3.9+
except ImportError:
    # Fallback para Python < 3.9 (requiere pip install backports.zoneinfo o pytz)
    logging.error("zoneinfo module not found. Install backports.zoneinfo or use Python 3.9+.")
    logging.error("Attempting fallback with pytz (pip install pytz might be needed).")
    try:
        from pytz import timezone as ZoneInfo
    except ImportError:
        logging.critical("Neither zoneinfo nor pytz found. Cannot proceed with timezone conversions.")
        sys.exit(1) # Salir si no hay manejo de zona horaria disponible.

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stdout)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(dotenv_path): logging.info(f"Loading .env from: {dotenv_path}"); load_dotenv(dotenv_path=dotenv_path)
else: logging.warning(f".env not found at {dotenv_path}.")
DEFAULT_DB_URL = "postgresql://user:password@host:port/database"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_PAST_DAYS = 7; ARCHIVE_DELAY_DAYS = 5; DEFAULT_FETCH_DAYS_IF_EMPTY = 7
CITIES_COORDS = { "Barranquilla": {"latitude": 11.0, "longitude": -74.75}, "Bogota": {"latitude": 4.625, "longitude": -74.125}, "Bucaramanga": {"latitude": 7.0, "longitude": -73.125}, "Ibague": {"latitude": 4.375, "longitude": -75.25}, "Cali": {"latitude": 3.5, "longitude": -76.5}, "Cartagena": {"latitude": 10.25, "longitude": -75.5}, "Cucuta": {"latitude": 8.0, "longitude": -72.5}, "Medellin": {"latitude": 6.125, "longitude": -75.75}, "Leticia": {"latitude": -4.25, "longitude": -69.875}, "Pereira": {"latitude": 4.75, "longitude": -75.75}, "Santa Marta": {"latitude": 11.125, "longitude": -74.125}, }
OPEN_METEO_VARIABLES = [ "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "wind_speed_10m", "uv_index", "pressure_msl", "shortwave_radiation", "cloud_cover", ]

# --- Definir Zona Horaria de Colombia ---
try:
    BOGOTA_TZ = ZoneInfo("America/Bogota")
except Exception as tz_error:
     logging.critical(f"Could not load Timezone 'America/Bogota'. Error: {tz_error}")
     sys.exit(1)

# --- Database Functions ---
async def get_latest_timestamp(pool: asyncpg.Pool, city: str) -> Optional[datetime]:
    """
    Obtiene el timestamp más reciente de la base de datos para una ciudad dada.
    Retorna un datetime naive que representa la hora local de Bogotá.
    """
    conn=None; latest_time=None
    try:
        async with pool.acquire() as conn:
            # La columna "time" en la BD se almacena como TIMESTAMP WITHOUT TIME ZONE, representando la hora local de Bogotá.
            query = 'SELECT MAX("time") FROM weather_data WHERE lower(city) = lower($1)'
            latest_time = await conn.fetchval(query, city)
            if latest_time:
                if latest_time.tzinfo is not None:
                    # Situación inesperada: El valor de la BD tiene timezone, forzar a naive.
                    logging.warning(f"[{city}] DB timestamp (naive column) was tz-aware. Forcing naive.")
                    return latest_time.replace(tzinfo=None)
                return latest_time # Ya es naive, representa la hora de Bogotá.
            logging.info(f"[{city}] No previous timestamp found.") ; return None
    except Exception as e: logging.error(f"[{city}] Error fetching latest timestamp: {e}", exc_info=True); return None

async def insert_weather_data(pool: asyncpg.Pool, city:str, records: List[Tuple]):
    """
    Inserta registros en la base de datos. Espera datetimes naive representando la hora de Bogotá.
    """
    if not records: logging.info(f"[{city}] No new records to insert."); return
    conn=None
    try:
        async with pool.acquire() as conn:
            columns = ["time", "city"] + OPEN_METEO_VARIABLES
            # Asegura que el nombre de columna 'time' se cite correctamente.
            column_names = ', '.join(f'"{c}"' if c == "time" else c for c in columns)
            placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
            # Ignora registros duplicados basados en la clave primaria (city, time).
            query = f'INSERT INTO weather_data ({column_names}) VALUES ({placeholders}) ON CONFLICT (city, "time") DO NOTHING;'
            status = await conn.executemany(query, records)
            logging.info(f"[{city}] Attempted insert/ignore for {len(records)} records. (Status: {status})")
    except asyncpg.exceptions.UndefinedTableError: logging.critical("Table 'weather_data' missing."); raise
    except Exception as e: logging.error(f"[{city}] Error inserting data: {e}", exc_info=True); (records and logging.error(f"[{city}] First record example: {records[0]}"))

# --- Open-Meteo API Function ---
async def fetch_open_meteo_data(client: httpx.AsyncClient, api_url: str, city: str, coords: Dict[str, float], variables: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None, past_days: Optional[int] = None, forecast_days: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Solicita datos a la API de Open-Meteo, siempre en timezone UTC."""
    params = {"latitude": coords["latitude"], "longitude": coords["longitude"], "hourly": ",".join(variables), "timezone": "UTC"}
    if past_days is not None: params["past_days"] = past_days
    if forecast_days is not None: params["forecast_days"] = forecast_days
    if start_date: params["start_date"] = start_date
    if end_date: params["end_date"] = end_date
    log_params = params.copy(); log_params["hourly"] = "..."; req_log_url = f"{api_url}?{'&'.join(f'{k}={v}' for k, v in log_params.items())}"
    try:
        response = await client.get(api_url, params=params, timeout=60.0)
        response.raise_for_status(); logging.info(f"[{city}] Fetched OK from {api_url}.")
        return response.json()
    except httpx.TimeoutException: logging.error(f"[{city}] Timeout. URL: {req_log_url}"); return None
    except httpx.RequestError as e: logging.error(f"[{city}] Request error: {e}. URL: {req_log_url}"); return None
    except httpx.HTTPStatusError as e: logging.error(f"[{city}] Status error: {e.response.status_code}. URL: {req_log_url}\nResponse: {e.response.text[:500]}"); return None
    except Exception as e: logging.error(f"[{city}] Unexpected fetch error: {e}", exc_info=True); return None


# --- Parsing Function ---
def parse_open_meteo_response(
    response_json: Dict[str, Any],
    city: str,
    # Último timestamp de la BD (naive, representando hora de Bogotá). Se usa como límite inferior del filtro.
    start_filtering_after_local_naive: Optional[datetime],
    # Hora UTC de inicio del script. Se usa para calcular el límite superior del filtro.
    current_script_time_utc: datetime
) -> Optional[List[Tuple]]:
    """
    Procesa la respuesta JSON de la API. Convierte tiempos UTC a hora local de Bogotá (naive),
    filtra registros ya existentes o futuros, redondea valores numéricos y formatea para inserción en BD.
    """
    try:
        if not response_json or "hourly" not in response_json: return None
        hourly_data = response_json["hourly"]
        if "time" not in hourly_data or not hourly_data["time"]: return None
        missing_vars = [var for var in OPEN_METEO_VARIABLES if var not in hourly_data]
        if missing_vars: logging.error(f"[{city}] Missing vars: {', '.join(missing_vars)}"); return None

        time_array = hourly_data["time"]; num_records = len(time_array)
        records_to_insert = []; parse_errors = 0; filter_count = 0
        variable_arrays = {var: hourly_data[var] for var in OPEN_METEO_VARIABLES}

        # --- Calcular límites de filtro (ambos como tiempo naive de Bogotá) ---
        # Límite inferior: El último registro existente en la BD (o el mínimo posible si no hay datos).
        lower_filter_bound_local_naive = start_filtering_after_local_naive if start_filtering_after_local_naive else datetime.min

        # Límite superior: La hora de inicio del script, convertida a Bogotá, truncada a la hora en punto y hecha naive.
        # Esto asegura que no insertemos datos parciales de la hora actual o datos futuros.
        current_script_time_bogota_aware = current_script_time_utc.astimezone(BOGOTA_TZ)
        upper_filter_bound_bogota_aware = current_script_time_bogota_aware.replace(minute=0, second=0, microsecond=0)
        upper_filter_bound_local_naive = upper_filter_bound_bogota_aware.replace(tzinfo=None)

        logging.info(f"[{city}] Parsing {num_records} records. Filtering > {lower_filter_bound_local_naive.isoformat()} AND <= {upper_filter_bound_local_naive.isoformat()} (Bogota Naive Time).")
        TWO_PLACES = Decimal("0.01")

        for i in range(num_records):
            try:
                # 1. Parsear el string ISO de la API (UTC) a un datetime aware (UTC).
                time_dt_aware_utc = datetime.fromisoformat(time_array[i]).replace(tzinfo=timezone.utc)
                # 2. Convertir a datetime aware (Bogotá).
                time_dt_aware_bogota = time_dt_aware_utc.astimezone(BOGOTA_TZ)
                # 3. Convertir a datetime naive (Bogotá) para filtrar y almacenar en BD.
                time_dt_naive_bogota = time_dt_aware_bogota.replace(tzinfo=None)
            except (ValueError, TypeError): parse_errors += 1; continue

            # --- Filtro temporal (usando tiempo naive de Bogotá) ---
            # Se incluyen solo los registros estrictamente posteriores al último guardado y
            # estrictamente anteriores o iguales a la hora de inicio del script (truncada).
            if not (lower_filter_bound_local_naive < time_dt_naive_bogota <= upper_filter_bound_local_naive):
                filter_count += 1
                continue

            # Construir tupla para inserción. Usar el tiempo naive de Bogotá.
            record_values = [time_dt_naive_bogota, city]
            try:
                for var in OPEN_METEO_VARIABLES:
                    value = variable_arrays[var][i]; processed_value = value
                    # Manejo específico para valores None (ej. uv_index puede ser None).
                    if value is not None:
                        try:
                             # Intentar convertir a Decimal y redondear a dos decimales.
                             processed_value = Decimal(str(value)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                        except (InvalidOperation, TypeError, ValueError):
                             # Si falla la conversión (ej. NaN), almacenar NULL.
                             logging.warning(f"[{city}] Could not process value '{value}' for '{var}'. Storing NULL.")
                             processed_value = None
                    # Si el valor original era None, processed_value permanece None.
                    record_values.append(processed_value)

            except IndexError: parse_errors += 1; continue
            records_to_insert.append(tuple(record_values))

        if parse_errors > 0: logging.warning(f"[{city}] Encountered {parse_errors} parsing errors.")
        logging.info(f"[{city}] Original: {num_records}. Filtered out: {filter_count}. Parsed new (Decimal, Bogota Naive): {len(records_to_insert)}.")
        return records_to_insert if records_to_insert else None
    except Exception as e: logging.error(f"[{city}] Error parsing response: {e}", exc_info=True); return None

# --- Main Orchestration Functions ---

async def update_one_city(
    pool: asyncpg.Pool, client: httpx.AsyncClient, city: str, coords: Dict[str, float],
    script_start_utc: datetime # Hora UTC de inicio del script, usada consistentemente para el filtrado.
):
    """
    Orquesta la actualización para una ciudad: obtiene el último dato, calcula rangos de fechas,
    solicita datos a las APIs (Forecast y Archive si es necesario), procesa la respuesta
    y almacena los nuevos registros. Usa internamente tiempos naive locales (Bogotá) para filtros
    y almacenamiento, y tiempos UTC para las llamadas a la API.
    """
    logging.info(f"--- [{city}] Processing started ---")
    now_utc = datetime.now(timezone.utc) # Hora UTC actual para cálculos de rangos de API.
    today_utc = now_utc.date()
    all_new_records = []
    try:
        # 1. Obtener el último timestamp almacenado (naive, representando hora de Bogotá).
        latest_timestamp_local_naive = await get_latest_timestamp(pool, city)

        # 2. Determinar el rango de tiempo necesario.
        start_filtering_after_local_naive : Optional[datetime] # Límite inferior del filtro (naive Bogota)
        first_needed_local_naive : Optional[datetime]         # Primer instante local (naive Bogota) que falta
        first_needed_utc : datetime                           # Primer instante UTC que corresponde a lo que falta

        if latest_timestamp_local_naive:
            start_filtering_after_local_naive = latest_timestamp_local_naive
            first_needed_local_naive = latest_timestamp_local_naive + timedelta(microseconds=1)
            # Asumir que el timestamp naive representa Bogotá, convertirlo a UTC para calcular
            # la fecha de inicio necesaria para la API de forma segura.
            temp_aware_bogota = latest_timestamp_local_naive.replace(tzinfo=BOGOTA_TZ)
            first_needed_utc = temp_aware_bogota.astimezone(timezone.utc)
        else:
            # Si no hay datos, obtener datos desde hace DEFAULT_FETCH_DAYS_IF_EMPTY días.
            default_start_date_calc = today_utc - timedelta(days=DEFAULT_FETCH_DAYS_IF_EMPTY)
            start_filtering_after_local_naive = datetime.combine(default_start_date_calc, dt_time.min) - timedelta(microseconds=1)
            first_needed_local_naive = start_filtering_after_local_naive + timedelta(microseconds=1)
            # El primer UTC necesario es el inicio del rango por defecto.
            first_needed_utc = datetime.combine(default_start_date_calc, dt_time.min, tzinfo=timezone.utc)

        logging.info(f"[{city}] Last DB local naive ts: {latest_timestamp_local_naive.isoformat() if latest_timestamp_local_naive else 'None'}. Filtering after: {start_filtering_after_local_naive.isoformat()}")

        # 3. Obtener datos recientes usando Forecast API (fechas UTC).
        # Se piden datos desde FORECAST_PAST_DAYS hasta hoy (UTC).
        forecast_api_start_date_utc = today_utc - timedelta(days=FORECAST_PAST_DAYS - 1)
        forecast_api_end_date_utc = today_utc
        logging.info(f"[{city}] Fetching recent data (Forecast API, start={forecast_api_start_date_utc}, end={forecast_api_end_date_utc}).")
        forecast_response = await fetch_open_meteo_data(
            client, FORECAST_API_URL, city, coords, OPEN_METEO_VARIABLES,
            start_date=forecast_api_start_date_utc.isoformat(),
            end_date=forecast_api_end_date_utc.isoformat()
        )
        if forecast_response:
            # Pasar el límite inferior (naive Bogota) y la hora de inicio del script (UTC).
            recent_records = parse_open_meteo_response(
                forecast_response, city, start_filtering_after_local_naive, script_start_utc
            )
            if recent_records: all_new_records.extend(recent_records); logging.info(f"[{city}] Found {len(recent_records)} new from Forecast (time-limited).")
        else: logging.warning(f"[{city}] No data received from Forecast API.")

        # 4. Obtener datos históricos faltantes si es necesario (Archive API, fechas UTC).
        # La API de archivo tiene un retraso (ARCHIVE_DELAY_DAYS).
        archive_latest_available_utc = now_utc - timedelta(days=ARCHIVE_DELAY_DAYS)
        # Verificar si el primer dato que necesitamos (UTC) es anterior al último dato disponible en el archivo (UTC).
        if first_needed_utc < archive_latest_available_utc:
            logging.info(f"[{city}] Older gap potentially exists (needed from UTC {first_needed_utc.isoformat()}, archive has up to UTC {archive_latest_available_utc.isoformat()}).")
            archive_start_date_utc = first_needed_utc.date()
            # Pedir datos hasta el último día completo disponible en el archivo.
            archive_end_date_utc = archive_latest_available_utc.date()
            if archive_start_date_utc <= archive_end_date_utc:
                logging.info(f"[{city}] Fetching older gap (Archive API, {archive_start_date_utc} to {archive_end_date_utc}).")
                archive_response = await fetch_open_meteo_data(
                    client, ARCHIVE_API_URL, city, coords, OPEN_METEO_VARIABLES,
                    start_date=archive_start_date_utc.isoformat(),
                    end_date=archive_end_date_utc.isoformat()
                )
                if archive_response:
                    # Pasar el límite inferior (naive Bogota) y la hora de inicio del script (UTC).
                    archive_records = parse_open_meteo_response(
                        archive_response, city, start_filtering_after_local_naive, script_start_utc
                    )
                    if archive_records: all_new_records.extend(archive_records); logging.info(f"[{city}] Found {len(archive_records)} older from Archive (time-limited).")
            # else: logging.debug(f"[{city}] Archive start date {archive_start_date_utc} > end date {archive_end_date_utc}. Skipping archive fetch.")
        else:
             logging.info(f"[{city}] No older data gap detected within Archive API's range.")

        # 5. Eliminar duplicados (si se obtuvieron de ambas APIs) e insertar.
        # Los registros ya tienen timestamps naive de Bogotá.
        if all_new_records:
            # Usar un diccionario para eliminar duplicados por (ciudad, timestamp) y luego ordenar por timestamp.
            unique_records = sorted(list({record[:2]: record for record in all_new_records}.values()), key=lambda r: r[0])
            logging.info(f"[{city}] Total unique new records to insert (time-limited, Bogota Naive): {len(unique_records)}.")
            await insert_weather_data(pool, city, unique_records)
        else:
            logging.info(f"[{city}] No new records found after all filtering.")
    except Exception as city_error:
        logging.error(f"!!! [{city}] FAILED processing: {city_error} !!!", exc_info=True)
        raise # Propagar la excepción para que asyncio.gather la capture.

async def update_all_cities():
    """
    Función principal que coordina la actualización para todas las ciudades en paralelo.
    Establece la conexión a la BD y crea tareas asyncio para cada ciudad.
    """
    # Capturar la hora UTC una sola vez al inicio para consistencia en el filtrado superior.
    script_start_utc = datetime.now(timezone.utc)
    logging.info(f"--- Starting script at {script_start_utc.isoformat()} ---")
    pool = None; total_processed = 0; total_failed = 0
    if "user:password@host:port/database" in DATABASE_URL: logging.critical("FATAL: Update DATABASE_URL."); return
    try:
        try:
            pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5, command_timeout=60)
            # Verificar conexión inicial.
            async with pool.acquire() as conn: await conn.fetchval("SELECT 1")
            logging.info("DB pool established.")
        except Exception as db_err: logging.critical(f"Failed DB pool creation: {db_err}", exc_info=True); return

        async with httpx.AsyncClient() as client:
            tasks = [
                # Pasar la misma hora de inicio UTC a todas las tareas.
                asyncio.create_task(update_one_city(pool, client, city, coords, script_start_utc))
                for city, coords in CITIES_COORDS.items()
            ]
            # Esperar a que todas las tareas terminen y recoger resultados/excepciones.
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Resumen de resultados.
            for i, res in enumerate(results):
                 city_name = list(CITIES_COORDS.keys())[i]
                 if isinstance(res, Exception): total_failed += 1; logging.error(f"--- Summary: Task [{city_name}] Error. ---")
                 else: total_processed += 1; logging.info(f"--- Summary: Task [{city_name}] OK. ---")
    except Exception as e: logging.critical(f"Unrecoverable error: {e}", exc_info=True)
    finally:
        if pool: await pool.close(); logging.info("DB pool closed.")
    script_end_time = datetime.now(timezone.utc)
    logging.info(f"--- Script finished at {script_end_time.isoformat()} (Duration: {script_end_time - script_start_utc}) ---")
    logging.info(f"--- Cities attempted: {len(CITIES_COORDS)}. Succeeded: {total_processed}. Failed: {total_failed} ---")

# --- Script Execution ---
if __name__ == "__main__":
    if "user:password@host:port/database" in DATABASE_URL: logging.warning("!"*60+"\n! UPDATE DATABASE_URL !\n"+"!"*60)

    asyncio.run(update_all_cities())