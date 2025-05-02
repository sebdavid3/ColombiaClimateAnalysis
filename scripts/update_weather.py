

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
    # Fallback para Python < 3.9 (requiere pip install pytz)
    # from pytz import timezone as ZoneInfo
    logging.error("zoneinfo module not found. Install backports.zoneinfo or use Python 3.9+.")
    logging.error("Attempting fallback with pytz (pip install pytz might be needed).")
    try:
        from pytz import timezone as ZoneInfo
    except ImportError:
        logging.critical("Neither zoneinfo nor pytz found. Cannot proceed with timezone conversions.")
        sys.exit(1) # Salir si no hay manejo de zona horaria

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
    Fetches the latest timestamp (naive, representing Bogota time) for a given city.
    """
    conn=None; latest_time=None
    try:
        async with pool.acquire() as conn:
            query = 'SELECT MAX("time") FROM weather_data WHERE lower(city) = lower($1)'
            latest_time = await conn.fetchval(query, city) # Should be naive from DB
            if latest_time:
                if latest_time.tzinfo is not None:
                    # Log unexpected case but return naive anyway
                    logging.warning(f"[{city}] DB timestamp (naive column) was tz-aware. Forcing naive.")
                    return latest_time.replace(tzinfo=None)
                return latest_time # Already naive, representing Bogota time
            logging.info(f"[{city}] No previous timestamp found.") ; return None
    except Exception as e: logging.error(f"[{city}] Error fetching latest timestamp: {e}", exc_info=True); return None

async def insert_weather_data(pool: asyncpg.Pool, city:str, records: List[Tuple]):
    """Inserts records with naive datetimes (representing Bogota time)."""
    # (Sin cambios, sigue esperando naive datetimes)
    if not records: logging.info(f"[{city}] No new records to insert."); return
    conn=None
    try:
        async with pool.acquire() as conn:
            columns = ["time", "city"] + OPEN_METEO_VARIABLES
            column_names = ', '.join(f'"{c}"' if c == "time" else c for c in columns)
            placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
            query = f'INSERT INTO weather_data ({column_names}) VALUES ({placeholders}) ON CONFLICT (city, "time") DO NOTHING;'
            status = await conn.executemany(query, records)
            logging.info(f"[{city}] Attempted insert/ignore for {len(records)} records. (Status: {status})")
    except asyncpg.exceptions.UndefinedTableError: logging.critical("Table 'weather_data' missing."); raise
    except Exception as e: logging.error(f"[{city}] Error inserting data: {e}", exc_info=True); (records and logging.error(f"[{city}] First record example: {records[0]}"))

# --- Open-Meteo API Function (sin cambios) ---
async def fetch_open_meteo_data(client: httpx.AsyncClient, api_url: str, city: str, coords: Dict[str, float], variables: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None, past_days: Optional[int] = None, forecast_days: Optional[int] = None) -> Optional[Dict[str, Any]]:
    # (Sin cambios, sigue pidiendo UTC a la API)
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


# --- Parsing Function (MODIFICADA para convertir a Bogota y luego naive) ---
def parse_open_meteo_response(
    response_json: Dict[str, Any],
    city: str,
    # Este es el último timestamp de la BD (naive, representando Bogota)
    start_filtering_after_local_naive: Optional[datetime],
    # Esta es la hora UTC actual del script
    current_script_time_utc: datetime
) -> Optional[List[Tuple]]:
    """
    Parses API UTC data, converts to Bogota time, makes naive, filters,
    rounds numerics, and formats for DB insertion.
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

        # --- Calcular límites de filtro (ambos como naive Bogota time) ---
        lower_filter_bound_local_naive = start_filtering_after_local_naive if start_filtering_after_local_naive else datetime.min

        # Convertir hora actual UTC a Bogota Aware, truncar, y hacer naive
        current_script_time_bogota_aware = current_script_time_utc.astimezone(BOGOTA_TZ)
        upper_filter_bound_bogota_aware = current_script_time_bogota_aware.replace(minute=0, second=0, microsecond=0)
        upper_filter_bound_local_naive = upper_filter_bound_bogota_aware.replace(tzinfo=None)

        logging.info(f"[{city}] Parsing {num_records} records. Filtering > {lower_filter_bound_local_naive.isoformat()} AND < {upper_filter_bound_local_naive.isoformat()} (Bogota Naive Time).")
        TWO_PLACES = Decimal("0.01")

        for i in range(num_records):
            try:
                # 1. Parsear API UTC string a Aware UTC Datetime
                time_dt_aware_utc = datetime.fromisoformat(time_array[i]).replace(tzinfo=timezone.utc)
                # 2. Convertir a Aware Bogota Datetime
                time_dt_aware_bogota = time_dt_aware_utc.astimezone(BOGOTA_TZ)
                # 3. Convertir a Naive Bogota Datetime (para filtro y almacenamiento)
                time_dt_naive_bogota = time_dt_aware_bogota.replace(tzinfo=None)
            except (ValueError, TypeError): parse_errors += 1; continue

            # --- DOBLE FILTRO (usando naive Bogota time) ---
            if not (lower_filter_bound_local_naive < time_dt_naive_bogota <= upper_filter_bound_local_naive):
                filter_count += 1
                continue

            # Construir tupla con redondeo Decimal y Naive Bogota Time
            record_values = [time_dt_naive_bogota, city] # <-- Usar naive Bogota time
            try:
                for var in OPEN_METEO_VARIABLES:
                    value = variable_arrays[var][i]; processed_value = value
                    # Corrección clave para UV Index None: no intentar convertir None a Decimal
                    if value is not None:
                        try:
                             # Intentar convertir a Decimal y redondear
                             processed_value = Decimal(str(value)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
                        except (InvalidOperation, TypeError, ValueError):
                             # Si la conversión falla (ej. NaN o tipo inesperado), guardar NULL
                             logging.warning(f"[{city}] Could not process value '{value}' for '{var}'. Storing NULL.")
                             processed_value = None
                    # Si el valor original ya era None, processed_value sigue siendo None
                    record_values.append(processed_value)

            except IndexError: parse_errors += 1; continue
            records_to_insert.append(tuple(record_values))

        if parse_errors > 0: logging.warning(f"[{city}] Encountered {parse_errors} parsing errors.")
        logging.info(f"[{city}] Original: {num_records}. Filtered out: {filter_count}. Parsed new (Decimal, Bogota Naive): {len(records_to_insert)}.")
        return records_to_insert if records_to_insert else None
    except Exception as e: logging.error(f"[{city}] Error parsing response: {e}", exc_info=True); return None

# --- Main Orchestration Functions (MODIFICADA para usar/pasar tiempos correctos) ---

async def update_one_city(
    pool: asyncpg.Pool, client: httpx.AsyncClient, city: str, coords: Dict[str, float],
    script_start_utc: datetime # Recibe la hora UTC de inicio del script
):
    """Handles update logic using local naive times internally."""
    logging.info(f"--- [{city}] Processing started ---")
    now_utc = datetime.now(timezone.utc) # Hora UTC actual (para cálculos de API)
    today_utc = now_utc.date()
    all_new_records = []
    try:
        # 1. Get latest timestamp (naive, representa Bogota time)
        latest_timestamp_local_naive = await get_latest_timestamp(pool, city)

        # 2. Determinar el filtro inferior (naive Bogota time)
        start_filtering_after_local_naive : Optional[datetime]
        first_needed_local_naive : Optional[datetime] # El primer momento local que falta
        if latest_timestamp_local_naive:
            start_filtering_after_local_naive = latest_timestamp_local_naive
            first_needed_local_naive = latest_timestamp_local_naive + timedelta(microseconds=1)
            # Para la API necesitamos UTC, asumimos que el naive local representa Bogota
            # y lo convertimos a UTC para calcular la fecha de inicio UTC de forma segura
            temp_aware_bogota = latest_timestamp_local_naive.replace(tzinfo=BOGOTA_TZ)
            first_needed_utc = temp_aware_bogota.astimezone(timezone.utc) + timedelta(microseconds=1)
        else:
            # Si no hay datos, el filtro inferior es muy antiguo
            default_start_date_calc = today_utc - timedelta(days=DEFAULT_FETCH_DAYS_IF_EMPTY)
            start_filtering_after_local_naive = datetime.combine(default_start_date_calc, dt_time.min) - timedelta(microseconds=1)
            first_needed_local_naive = start_filtering_after_local_naive + timedelta(microseconds=1)
            # El primer UTC necesario es el inicio del rango por defecto
            first_needed_utc = datetime.combine(default_start_date_calc, dt_time.min, tzinfo=timezone.utc)

        logging.info(f"[{city}] Last DB local naive ts: {latest_timestamp_local_naive.isoformat() if latest_timestamp_local_naive else 'None'}. Filtering after: {start_filtering_after_local_naive.isoformat()}")

        # 3. Fetch Recent Data (Forecast API usando fechas UTC)
        forecast_api_start_date_utc = today_utc - timedelta(days=FORECAST_PAST_DAYS - 1)
        forecast_api_end_date_utc = today_utc # Pedir hasta hoy
        logging.info(f"[{city}] Fetching recent data (Forecast API, start={forecast_api_start_date_utc}, end={forecast_api_end_date_utc}).")
        forecast_response = await fetch_open_meteo_data(
            client, FORECAST_API_URL, city, coords, OPEN_METEO_VARIABLES,
            start_date=forecast_api_start_date_utc.isoformat(),
            end_date=forecast_api_end_date_utc.isoformat()
        )
        if forecast_response:
            # Pasar el filtro INFERIOR (naive Bogota) y la hora ACTUAL (UTC)
            recent_records = parse_open_meteo_response(
                forecast_response, city, start_filtering_after_local_naive, script_start_utc
            )
            if recent_records: all_new_records.extend(recent_records); logging.info(f"[{city}] Found {len(recent_records)} new from Forecast (time-limited).")
        else: logging.warning(f"[{city}] No data received from Forecast API.")

        # 4. Conditionally Fetch Older Gap (Archive API usando fechas UTC)
        # El límite del archivo es un punto en UTC, lo comparamos con el primer UTC necesario
        archive_latest_available_utc = now_utc - timedelta(days=ARCHIVE_DELAY_DAYS)
        if first_needed_utc < archive_latest_available_utc:
            logging.info(f"[{city}] Older gap potentially exists (needed from UTC {first_needed_utc.isoformat()}, archive has up to UTC {archive_latest_available_utc.isoformat()}).")
            archive_start_date_utc = first_needed_utc.date()
            archive_end_date_utc = archive_latest_available_utc.date()
            if archive_start_date_utc <= archive_end_date_utc:
                logging.info(f"[{city}] Fetching older gap (Archive API, {archive_start_date_utc} to {archive_end_date_utc}).")
                archive_response = await fetch_open_meteo_data(
                    client, ARCHIVE_API_URL, city, coords, OPEN_METEO_VARIABLES,
                    start_date=archive_start_date_utc.isoformat(),
                    end_date=archive_end_date_utc.isoformat()
                )
                if archive_response:
                    # Pasar filtro INFERIOR (naive Bogota) y hora ACTUAL (UTC)
                    archive_records = parse_open_meteo_response(
                        archive_response, city, start_filtering_after_local_naive, script_start_utc
                    )
                    if archive_records: all_new_records.extend(archive_records); logging.info(f"[{city}] Found {len(archive_records)} older from Archive (time-limited).")
            # else: logging.warning(f"[{city}] Archive start > end date. Skipping.")
        else:
             logging.info(f"[{city}] No older data gap detected within Archive API's range.")

        # 5. Deduplicate and Insert (contiene naive Bogota times)
        if all_new_records:
            unique_records = sorted(list({record[:2]: record for record in all_new_records}.values()), key=lambda r: r[0])
            logging.info(f"[{city}] Total unique new records to insert (time-limited, Bogota Naive): {len(unique_records)}.")
            await insert_weather_data(pool, city, unique_records)
        else:
            logging.info(f"[{city}] No new records found after all filtering.")
    except Exception as city_error:
        logging.error(f"!!! [{city}] FAILED processing: {city_error} !!!", exc_info=True)
        raise

async def update_all_cities():
    """Main function orchestrating updates, passing script start UTC."""
    script_start_utc = datetime.now(timezone.utc) # Captura UTC al inicio
    logging.info(f"--- Starting script at {script_start_utc.isoformat()} ---")
    pool = None; total_processed = 0; total_failed = 0
    if "user:password@host:port/database" in DATABASE_URL: logging.critical("FATAL: Update DATABASE_URL."); return
    try:
        try:
            pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5, command_timeout=60)
            async with pool.acquire() as conn: await conn.fetchval("SELECT 1")
            logging.info("DB pool established.")
        except Exception as db_err: logging.critical(f"Failed DB pool creation: {db_err}", exc_info=True); return

        async with httpx.AsyncClient() as client:
            tasks = [
                # Pasar la misma hora UTC de inicio a todas las tareas
                asyncio.create_task(update_one_city(pool, client, city, coords, script_start_utc))
                for city, coords in CITIES_COORDS.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # (Resumen de logs sin cambios)
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

