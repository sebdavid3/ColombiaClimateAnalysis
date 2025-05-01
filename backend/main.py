import asyncpg
from fastapi import FastAPI, Depends, HTTPException, status, Query
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any, Optional, Set
from datetime import datetime, date, timedelta
import traceback

from .config import settings
from .schemas.weather import WeatherRecord


db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando aplicación y conexión a BD...")
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=10) 
        async with db_pool.acquire() as connection:
             await connection.fetchval("SELECT 1")
        print("Conexión a la base de datos establecida exitosamente.")
    except Exception as e:
        print(f"!! Error: {e} !!")
        db_pool = None 
    yield 

    print("Cerrando conexión a la base de datos...")
    if db_pool:
        await db_pool.close()
        print("Conexión a la base de datos cerrada.")

app = FastAPI(
    title="Api Clima Colombia",
    description="Api para obtener el clima de Colombia conectada a una base de datos",
    version="1.0.0",
    lifespan=lifespan 
)


async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    if not db_pool:
        print("ERROR")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio no disponible: El pool de la base de datos no está inicializado."
        )

    conn = None
    try:
        # 1. Adquirir conexión
        conn = await db_pool.acquire()

        # 2. Entregar la conexión a la función de ruta
        yield conn

    except Exception as e:
        print(f"!!!!!!!! ERROR en get_db_connection (acquire/yield): {e} !!!!!!!!")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor al obtener la conexión DB."
        ) from e
    finally:
        # 3. Liberar la conexión SIEMPRE, si se adquirió
        if conn:
            try:
                await db_pool.release(conn)
            except Exception as e:
                print(f"!!!!!!!! ERROR CRÍTICO en get_db_connection (release): {e} !!!!!!!!")


# --- Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "¡Bienvenido a la API del Clima de Colombia!"}

@app.get("/db-test")
async def test_db_connection(conn: asyncpg.Connection = Depends(get_db_connection)):
    """
    Endpoint para verificar que podemos obtener y usar una conexión a la BD.
    """
    try:
        result = await conn.fetchval("SELECT version();")
        return {"db_version": result}
    except Exception as e:
        print(f"!!!!!!!! ERROR durante la consulta en /db-test: {e} !!!!!!!!")
        raise HTTPException(status_code=500, detail=f"Error interactuando con la base de datos: {e}")


#aqui iba el codigo de las pruebas
    
# --- Constants for Validation ---
ALLOWED_WEATHER_VARIABLES: Set[str] = {
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "wind_speed_10m", "uv_index", "pressure_msl",
    "shortwave_radiation", "cloud_cover"
}
ALLOWED_PRECIP_GRANULARITY: Set[str] = {"daily", "weekly", "monthly"}
ALLOWED_AVG_GRANULARITY: Set[str] = {"hourly", "daily", "weekly", "monthly"}

# --- Helper Functions ---

def validate_variable_name(variable_name: str):
    """Raises HTTPException if variable name is not allowed."""
    if variable_name not in ALLOWED_WEATHER_VARIABLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid variable name '{variable_name}'. Allowed variables are: {', '.join(ALLOWED_WEATHER_VARIABLES)}"
        )

def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parses YYYY-MM-DD string to date object or raises HTTPException."""
    if date_str is None:
        return None
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid date format '{date_str}'. Use 'YYYY-MM-DD'."
        )

def parse_cities_list(cities_str: str) -> List[str]:
    """Parses comma-separated city string into a list of lowercase, stripped names."""
    if not cities_str:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cities parameter cannot be empty."
        )
    cities = [city.strip().lower() for city in cities_str.split(',') if city.strip()]
    if not cities:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid city names provided in 'cities' parameter after parsing."
        )
    return cities

# --- New Endpoints ---

@app.get("/weather/{city_name}/trends", response_model=List[Dict[str, Any]])
async def get_city_trends(
    city_name: str,
    variable_name: str = Query(..., description=f"Weather variable to plot. Allowed: {', '.join(ALLOWED_WEATHER_VARIABLES)}"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Get time series data for a specific variable and city, optionally filtered by date.
    """
    validate_variable_name(variable_name)
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)

    # Build query safely - variable name is validated, so direct inclusion is okay here.
    query_parts = [f"SELECT time, {variable_name}"] # Select time and the validated variable
    query_parts.append("FROM weather_data")
    query_parts.append("WHERE lower(city) = lower($1)")

    params = [city_name]
    param_index = 2

    if start_dt:
        query_parts.append(f"AND time >= ${param_index}")
        params.append(start_dt)
        param_index += 1
    if end_dt:
        # Add one day to end_dt to include the whole day
        inclusive_end_dt = end_dt + timedelta(days=1)
        query_parts.append(f"AND time < ${param_index}")
        params.append(inclusive_end_dt)
        param_index += 1

    query_parts.append("ORDER BY time ASC")
    query = " ".join(query_parts) + ";"

    try:
        print(f"Executing Trends Query: {query} with params: {params}")
        records = await conn.fetch(query, *params)

        if not records:
            # Check if the city exists at all, maybe return empty list instead of 404?
            # For now, assume no records for the filters means "not found" for this request.
            # Could add a separate check if needed.
             raise HTTPException(
                 status_code=status.HTTP_404_NOT_FOUND,
                 detail=f"No trend data found for city '{city_name}' with variable '{variable_name}' and specified date range."
             )

        # Convert asyncpg Records to list of dictionaries
        result_list = [{ "time": r['time'], variable_name: r[variable_name] } for r in records]
        return result_list

    except HTTPException:
        raise # Re-raise specific HTTP exceptions (like 404)
    except Exception as e:
        print(f"!!!!!!!! ERROR during trends query for {city_name}: {e} !!!!!!!!")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error retrieving trend data for {city_name}."
        )


@app.get("/weather/{city_name}/precipitation/summary", response_model=List[Dict[str, Any]])
async def get_precipitation_summary(
    city_name: str,
    granularity: str = Query(..., description=f"Time granularity. Allowed: {', '.join(ALLOWED_PRECIP_GRANULARITY)}"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Get summarized precipitation totals (daily, weekly, monthly) for a city.
    """
    if granularity not in ALLOWED_PRECIP_GRANULARITY:
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid granularity '{granularity}'. Allowed values: {', '.join(ALLOWED_PRECIP_GRANULARITY)}"
        )

    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)

    # Map granularity to SQL DATE_TRUNC argument
    sql_granularity_map = {'daily': 'day', 'weekly': 'week', 'monthly': 'month'}
    sql_granularity = sql_granularity_map[granularity]

    query_parts = [
        f"SELECT DATE_TRUNC('{sql_granularity}', time) AS period_start,",
        "SUM(precipitation) AS total_precipitation",
        "FROM weather_data",
        "WHERE lower(city) = lower($1) AND precipitation IS NOT NULL" # Ensure we only sum actual values
    ]
    params = [city_name]
    param_index = 2

    if start_dt:
        query_parts.append(f"AND time >= ${param_index}")
        params.append(start_dt)
        param_index += 1
    if end_dt:
        inclusive_end_dt = end_dt + timedelta(days=1)
        query_parts.append(f"AND time < ${param_index}")
        params.append(inclusive_end_dt)
        param_index += 1

    query_parts.append("GROUP BY period_start")
    query_parts.append("ORDER BY period_start ASC")
    query = " ".join(query_parts) + ";"

    try:
        print(f"Executing Precipitation Summary Query: {query} with params: {params}")
        records = await conn.fetch(query, *params)

        if not records:
             raise HTTPException(
                 status_code=status.HTTP_404_NOT_FOUND,
                 detail=f"No precipitation summary data found for city '{city_name}' with granularity '{granularity}' and specified date range."
             )

        # Convert records, handle potential None if SUM is over zero rows (though WHERE clause helps)
        result_list = [
            {
                "period_start": r['period_start'],
                "total_precipitation": r['total_precipitation'] if r['total_precipitation'] is not None else 0.0
            }
            for r in records
        ]
        return result_list

    except HTTPException:
        raise
    except Exception as e:
        print(f"!!!!!!!! ERROR during precipitation summary query for {city_name}: {e} !!!!!!!!")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error retrieving precipitation summary for {city_name}."
        )


@app.get("/weather/current", response_model=Dict[str, Dict[str, Any]])
async def get_current_weather_for_map(
    variable_name: str = Query(..., description=f"Weather variable to display. Allowed: {', '.join(ALLOWED_WEATHER_VARIABLES)}"),
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Get the most recent value for a specific variable for all distinct cities.
    """
    validate_variable_name(variable_name)

    # Use DISTINCT ON (city) to get the latest record per city
    # Order by city, then time descending, DISTINCT ON picks the first row (which is the latest)
    query = f"""
        SELECT DISTINCT ON (lower(city))
               city,
               time,
               {variable_name} AS value
        FROM weather_data
        WHERE {variable_name} IS NOT NULL
        ORDER BY lower(city), time DESC;
    """
    # Note: WHERE {variable_name} IS NOT NULL ensures we get actual values

    try:
        print(f"Executing Current Weather Query: {query}")
        records = await conn.fetch(query)

        if not records:
            # This likely means the table is empty or the variable has no non-null values
            return {} # Return empty dict, not 404

        # Format into the desired dictionary structure
        result_dict = {
            r['city']: {
                "variable": variable_name,
                "value": r['value'],
                "timestamp": r['time']
            }
            for r in records
        }
        return result_dict

    except asyncpg.exceptions.UndefinedColumnError:
         # This happens if the variable_name was invalid despite the initial check (e.g., DB schema changed)
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Database column '{variable_name}' does not exist."
        )
    except Exception as e:
        print(f"!!!!!!!! ERROR during current weather query: {e} !!!!!!!!")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error retrieving current weather data."
        )


@app.get("/weather/{city_name}/correlation", response_model=List[Dict[str, Any]])
async def get_variable_correlation(
    city_name: str,
    variable_x: str = Query(..., description=f"Variable for X-axis. Allowed: {', '.join(ALLOWED_WEATHER_VARIABLES)}"),
    variable_y: str = Query(..., description=f"Variable for Y-axis. Allowed: {', '.join(ALLOWED_WEATHER_VARIABLES)}"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Get pairs of values for two variables in a city, for scatter plot correlation.
    """
    validate_variable_name(variable_x)
    validate_variable_name(variable_y)
    if variable_x == variable_y:
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="variable_x and variable_y cannot be the same."
        )

    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)

    # Build query safely - variables are validated
    query_parts = [f"SELECT time, {variable_x}, {variable_y}"]
    query_parts.append("FROM weather_data")
    query_parts.append(f"WHERE lower(city) = lower($1) AND {variable_x} IS NOT NULL AND {variable_y} IS NOT NULL")

    params = [city_name]
    param_index = 2

    if start_dt:
        query_parts.append(f"AND time >= ${param_index}")
        params.append(start_dt)
        param_index += 1
    if end_dt:
        inclusive_end_dt = end_dt + timedelta(days=1)
        query_parts.append(f"AND time < ${param_index}")
        params.append(inclusive_end_dt)
        param_index += 1

    # Order by time usually makes sense for scatter plots too, but not strictly required
    query_parts.append("ORDER BY time ASC")
    query = " ".join(query_parts) + ";"

    try:
        print(f"Executing Correlation Query: {query} with params: {params}")
        records = await conn.fetch(query, *params)

        if not records:
             raise HTTPException(
                 status_code=status.HTTP_404_NOT_FOUND,
                 detail=f"No data found for city '{city_name}' with variables '{variable_x}' and '{variable_y}' for the specified date range."
             )

        # Convert records to list of dictionaries
        result_list = [
            {"time": r['time'], variable_x: r[variable_x], variable_y: r[variable_y]}
            for r in records
        ]
        return result_list

    except HTTPException:
        raise
    except asyncpg.exceptions.UndefinedColumnError:
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"One or both database columns ('{variable_x}', '{variable_y}') do not exist."
        )
    except Exception as e:
        print(f"!!!!!!!! ERROR during correlation query for {city_name}: {e} !!!!!!!!")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error retrieving correlation data for {city_name}."
        )


@app.get("/weather/averages", response_model=List[Dict[str, Any]])
async def get_comparative_averages(
    variable_name: str = Query(..., description=f"Variable to average. Allowed: {', '.join(ALLOWED_WEATHER_VARIABLES)}"),
    granularity: str = Query(..., description=f"Time granularity for averaging. Allowed: {', '.join(ALLOWED_AVG_GRANULARITY)}"),
    cities: str = Query(..., description="Comma-separated list of city names (e.g., Bogota,Medellin)"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Calculate average values for a variable across multiple cities, grouped by time period.
    """
    validate_variable_name(variable_name)
    if granularity not in ALLOWED_AVG_GRANULARITY:
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid granularity '{granularity}'. Allowed values: {', '.join(ALLOWED_AVG_GRANULARITY)}"
        )

    cities_list = parse_cities_list(cities)
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)

    # Determine grouping expression and period identifier based on granularity
    period_alias = "period_value" # Use a consistent alias
    if granularity == 'hourly':
        grouping_expression = "EXTRACT(hour FROM time)"
        period_name = "hour"
    elif granularity == 'daily':
        grouping_expression = "DATE_TRUNC('day', time)"
        period_name = "period_start"
    elif granularity == 'weekly':
        # ISO day of week (1=Mon, 7=Sun)
        grouping_expression = "EXTRACT(isodow FROM time)"
        period_name = "day_of_week_iso"
    elif granularity == 'monthly':
        # Month number (1=Jan, 12=Dec)
        grouping_expression = "EXTRACT(month FROM time)"
        period_name = "month"
    else:
        # Should not happen due to validation, but safeguard
         raise HTTPException(status_code=500, detail="Internal mapping error for granularity.")

    avg_col_alias = f"average_{variable_name}"

    query_parts = [
        f"SELECT {grouping_expression} AS {period_alias},",
        "city,",
        f"AVG({variable_name}) AS {avg_col_alias}",
        "FROM weather_data",
        # Use ANY for efficient list comparison. $1 needs to be a list/tuple passed to fetch.
        # Convert input cities to lowercase for the comparison array as well.
        "WHERE lower(city) = ANY($1::text[])" # $1 will be the list of lowercase city names
    ]
    params: List[Any] = [cities_list] # Pass the list directly
    param_index = 2

    # Add optional date filters before aggregation
    if start_dt:
        query_parts.append(f"AND time >= ${param_index}")
        params.append(start_dt)
        param_index += 1
    if end_dt:
        inclusive_end_dt = end_dt + timedelta(days=1)
        query_parts.append(f"AND time < ${param_index}")
        params.append(inclusive_end_dt)
        param_index += 1

    # Ensure we only average non-null values
    query_parts.append(f"AND {variable_name} IS NOT NULL")

    # Group by the calculated period and city
    query_parts.append(f"GROUP BY {period_alias}, city")
    # Order for consistent output
    query_parts.append(f"ORDER BY city, {period_alias} ASC")
    query = " ".join(query_parts) + ";"

        # --- INICIO DEPURACIÓN EXTRA ---
    print("--- DEBUG: Final query string ---")
    # Imprime la query final rodeada de delimitadores para ver espacios/saltos de línea
    print(f"QUERY_START>>>{query}<<<QUERY_END")
    # --- FIN DEPURACIÓN EXTRA ---

    try:
        print(f"Executing Averages Query: {query} with params: {params}")
        records = await conn.fetch(query, *params)

        if not records:
             raise HTTPException(
                 status_code=status.HTTP_404_NOT_FOUND,
                 detail=f"No average data found for variable '{variable_name}' in cities '{', '.join(cities_list)}' with granularity '{granularity}' for the specified date range."
             )

        # Convert records, adjusting the period key name
        result_list = []
        for r in records:
            record_dict = {
                period_name: r[period_alias], # Use the descriptive name based on granularity
                "city": r['city'],
                avg_col_alias: r[avg_col_alias] if r[avg_col_alias] is not None else None # AVG can return NULL
            }
            result_list.append(record_dict)

        return result_list

    except HTTPException:
        raise
    except asyncpg.exceptions.UndefinedColumnError:
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Database column '{variable_name}' does not exist."
        )
    except Exception as e:
        print(f"!!!!!!!! ERROR during averages query: {e} !!!!!!!!")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error retrieving comparative average data."
        )
    
@app.get("/weather/{city_name}", response_model=list[WeatherRecord])
async def get_city_weather(
    city_name: str,
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Obtiene los registros climáticos históricos para una ciudad específica.
    """
    print(f"--- Iniciando consulta para: {city_name} ---")
    formatted_city_name = city_name.capitalize()
    query = """
        SELECT id, time, temperature_2m, relative_humidity_2m, dew_point_2m,
               precipitation, wind_speed_10m, uv_index, pressure_msl,
               shortwave_radiation, cloud_cover, city
        FROM weather_data
        WHERE lower(city) = lower($1)
        ORDER BY time DESC;
    """
    try:
        print(f"Ejecutando fetch para {formatted_city_name}...")
        records = await conn.fetch(query, formatted_city_name)
        print(f"Fetch completado. Registros encontrados: {len(records)}")

        if not records:
            print(f"No se encontraron datos para {city_name}, devolviendo 404.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontraron datos para la ciudad: {city_name}"
            )

        result_list = [dict(record) for record in records]

        print(f"Devolviendo {len(result_list)} registros convertidos a dict para {city_name}.")
        return result_list

    except HTTPException:
        raise
    except Exception as e:
        print(f"!!!!!!!! ERROR durante la consulta de clima para {city_name}: {e} !!!!!!!!")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor al obtener datos para {city_name}."
        )