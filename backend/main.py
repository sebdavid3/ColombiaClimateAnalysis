import asyncpg
from fastapi import FastAPI, Depends, HTTPException, status, Query
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any, Optional, Set
from datetime import datetime, date, timedelta
import traceback

from .config import settings
from .schemas.weather import WeatherRecord


db_pool = None


# Gestiona el ciclo de vida de la aplicación, estableciendo y cerrando el pool de conexiones a la base de datos.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando aplicación y conexión a BD...")
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=10)
        # Verifica la conexión inicial al adquirir y ejecutar un comando simple.
        async with db_pool.acquire() as connection:
             await connection.fetchval("SELECT 1")
        print("Conexión a la base de datos establecida exitosamente.")
    except Exception as e:
        print(f"!! Error al inicializar el pool de la base de datos: {e} !!")
        db_pool = None # Asegura que el pool no se use si la inicialización falla.
    yield # Permite que la aplicación se ejecute

    print("Cerrando conexión a la base de datos...")
    if db_pool:
        await db_pool.close()
        print("Conexión a la base de datos cerrada.")

app = FastAPI(
    title="Api Clima Colombia",
    description="Api para obtener el clima de Colombia conectada a una base de datos",
    version="1.0.0",
    lifespan=lifespan # Registra la función lifespan para gestionar recursos.
)

# Dependencia de FastAPI para obtener una conexión de base de datos del pool.
# Gestiona la adquisición y liberación de conexiones para cada solicitud que la necesite.
async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    if not db_pool:
        # Si el pool no está inicializado, el servicio no está disponible.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio no disponible: El pool de la base de datos no está inicializado."
        )

    conn = None
    try:
        # Adquiere una conexión del pool.
        conn = await db_pool.acquire()
        # Entrega la conexión a la función de ruta que la solicitó.
        yield conn
    except Exception as e:
        print(f"!!!!!!!! ERROR en get_db_connection (acquire/yield): {e} !!!!!!!!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor al obtener la conexión DB."
        ) from e
    finally:
        # Asegura que la conexión se libere de vuelta al pool, incluso si ocurren errores.
        if conn:
            try:
                await db_pool.release(conn)
            except Exception as e:
                # Un error aquí es crítico, ya que podría indicar un problema con el pool.
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


# --- Constantes para Validación ---
ALLOWED_WEATHER_VARIABLES: Set[str] = {
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "wind_speed_10m", "uv_index", "pressure_msl",
    "shortwave_radiation", "cloud_cover"
}
ALLOWED_PRECIP_GRANULARITY: Set[str] = {"daily", "weekly", "monthly"}
ALLOWED_AVG_GRANULARITY: Set[str] = {"hourly", "daily", "weekly", "monthly"}

# --- Funciones Auxiliares ---

def validate_variable_name(variable_name: str):
    """Lanza HTTPException si el nombre de la variable no es permitido."""
    if variable_name not in ALLOWED_WEATHER_VARIABLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Nombre de variable inválido '{variable_name}'. Variables permitidas: {', '.join(ALLOWED_WEATHER_VARIABLES)}"
        )

def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Convierte una cadena YYYY-MM-DD a objeto date o lanza HTTPException."""
    if date_str is None:
        return None
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Formato de fecha inválido '{date_str}'. Use 'YYYY-MM-DD'."
        )

def parse_cities_list(cities_str: str) -> List[str]:
    """Convierte una cadena de ciudades separadas por comas en una lista de nombres en minúsculas y sin espacios extra."""
    if not cities_str:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El parámetro 'cities' no puede estar vacío."
        )
    cities = [city.strip().lower() for city in cities_str.split(',') if city.strip()]
    if not cities:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No se proporcionaron nombres de ciudad válidos en el parámetro 'cities' después del análisis."
        )
    return cities

# --- Nuevos Endpoints ---

@app.get("/weather/{city_name}/trends", response_model=List[Dict[str, Any]])
async def get_city_trends(
    city_name: str,
    variable_name: str = Query(..., description=f"Variable climática a graficar. Permitidas: {', '.join(ALLOWED_WEATHER_VARIABLES)}"),
    start_date: Optional[str] = Query(None, description="Filtro de fecha de inicio (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filtro de fecha de fin (YYYY-MM-DD)"),
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Obtiene datos de series temporales para una variable y ciudad específicas, opcionalmente filtrados por fecha.
    """
    validate_variable_name(variable_name)
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)

    # Construye la consulta de forma segura. La inclusión directa de 'variable_name' es segura
    # porque ha sido validada contra la lista ALLOWED_WEATHER_VARIABLES.
    query_parts = [f"SELECT time, {variable_name}"]
    query_parts.append("FROM weather_data")
    query_parts.append("WHERE lower(city) = lower($1)")

    params = [city_name]
    param_index = 2

    if start_dt:
        query_parts.append(f"AND time >= ${param_index}")
        params.append(start_dt)
        param_index += 1
    if end_dt:
        # Se añade un día a end_dt para que la consulta incluya todo el día final (hasta las 23:59:59...).
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
             raise HTTPException(
                 status_code=status.HTTP_404_NOT_FOUND,
                 detail=f"No se encontraron datos de tendencias para la ciudad '{city_name}' con la variable '{variable_name}' y el rango de fechas especificado."
             )

        # Convierte los registros de asyncpg a una lista de diccionarios para la respuesta JSON.
        result_list = [{ "time": r['time'], variable_name: r[variable_name] } for r in records]
        return result_list

    except HTTPException:
        raise # Re-lanza excepciones HTTP específicas (como 404).
    except Exception as e:
        print(f"!!!!!!!! ERROR during trends query for {city_name}: {e} !!!!!!!!")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor al recuperar datos de tendencias para {city_name}."
        )


@app.get("/weather/{city_name}/precipitation/summary", response_model=List[Dict[str, Any]])
async def get_precipitation_summary(
    city_name: str,
    granularity: str = Query(..., description=f"Granularidad temporal. Permitidas: {', '.join(ALLOWED_PRECIP_GRANULARITY)}"),
    start_date: Optional[str] = Query(None, description="Filtro de fecha de inicio (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filtro de fecha de fin (YYYY-MM-DD)"),
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Obtiene totales de precipitación acumulados (diario, semanal, mensual) para una ciudad.
    """
    if granularity not in ALLOWED_PRECIP_GRANULARITY:
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Granularidad inválida '{granularity}'. Valores permitidos: {', '.join(ALLOWED_PRECIP_GRANULARITY)}"
        )

    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)

    # Mapea la granularidad solicitada al argumento SQL correspondiente para DATE_TRUNC.
    sql_granularity_map = {'daily': 'day', 'weekly': 'week', 'monthly': 'month'}
    sql_granularity = sql_granularity_map[granularity]

    query_parts = [
        # Trunca la fecha/hora al inicio del período (día, semana, mes) según la granularidad.
        f"SELECT DATE_TRUNC('{sql_granularity}', time) AS period_start,",
        # Suma la precipitación dentro de cada período truncado.
        "SUM(precipitation) AS total_precipitation",
        "FROM weather_data",
        # Filtra por ciudad y excluye registros sin datos de precipitación para evitar sumar NULLs.
        "WHERE lower(city) = lower($1) AND precipitation IS NOT NULL"
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

    # Agrupa los resultados por el inicio del período calculado para obtener la suma por período.
    query_parts.append("GROUP BY period_start")
    query_parts.append("ORDER BY period_start ASC")
    query = " ".join(query_parts) + ";"

    try:
        print(f"Executing Precipitation Summary Query: {query} with params: {params}")
        records = await conn.fetch(query, *params)

        if not records:
             raise HTTPException(
                 status_code=status.HTTP_404_NOT_FOUND,
                 detail=f"No se encontraron datos resumidos de precipitación para la ciudad '{city_name}' con granularidad '{granularity}' y el rango de fechas especificado."
             )

        # Convierte los registros, manejando el caso donde SUM podría devolver NULL si no hay filas (aunque WHERE lo previene).
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
            detail=f"Error interno del servidor al recuperar el resumen de precipitación para {city_name}."
        )


@app.get("/weather/current", response_model=Dict[str, Dict[str, Any]])
async def get_current_weather_for_map(
    variable_name: str = Query(..., description=f"Variable climática a mostrar. Permitidas: {', '.join(ALLOWED_WEATHER_VARIABLES)}"),
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Obtiene el valor más reciente para una variable específica para todas las ciudades distintas.
    Ideal para visualizar el estado actual en un mapa.
    """
    validate_variable_name(variable_name)

    # Utiliza DISTINCT ON (lower(city)) junto con ORDER BY para obtener eficientemente
    # el registro más reciente (según 'time DESC') para cada ciudad única.
    # Es una característica útil de PostgreSQL para este tipo de consulta "último por grupo".
    query = f"""
        SELECT DISTINCT ON (lower(city))
               city,
               time,
               {variable_name} AS value
        FROM weather_data
        WHERE {variable_name} IS NOT NULL -- Asegura que solo se consideren valores no nulos para la variable.
        ORDER BY lower(city), time DESC;
    """

    try:
        print(f"Executing Current Weather Query: {query}")
        records = await conn.fetch(query)

        if not records:
            # Esto podría significar que la tabla está vacía o la variable no tiene valores no nulos.
            # Devolver un diccionario vacío es apropiado en este caso.
            return {}

        # Formatea los resultados en la estructura de diccionario anidado deseada: {ciudad: {datos}}.
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
         # Ocurre si variable_name no existe en la tabla (a pesar de la validación inicial, p.ej., si el esquema cambia).
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"La columna de base de datos '{variable_name}' no existe."
        )
    except Exception as e:
        print(f"!!!!!!!! ERROR during current weather query: {e} !!!!!!!!")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor al recuperar los datos climáticos actuales."
        )


@app.get("/weather/{city_name}/correlation", response_model=List[Dict[str, Any]])
async def get_variable_correlation(
    city_name: str,
    variable_x: str = Query(..., description=f"Variable para el eje X. Permitidas: {', '.join(ALLOWED_WEATHER_VARIABLES)}"),
    variable_y: str = Query(..., description=f"Variable para el eje Y. Permitidas: {', '.join(ALLOWED_WEATHER_VARIABLES)}"),
    start_date: Optional[str] = Query(None, description="Filtro de fecha de inicio (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filtro de fecha de fin (YYYY-MM-DD)"),
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Obtiene pares de valores para dos variables en una ciudad, útil para gráficos de dispersión y análisis de correlación.
    """
    validate_variable_name(variable_x)
    validate_variable_name(variable_y)
    if variable_x == variable_y:
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="variable_x y variable_y no pueden ser la misma."
        )

    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)

    # Construye la consulta seleccionando las dos variables validadas.
    query_parts = [f"SELECT time, {variable_x}, {variable_y}"]
    query_parts.append("FROM weather_data")
    # Asegura que ambos valores existan (no sean NULL) para que el par sea útil para correlación.
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

    query_parts.append("ORDER BY time ASC")
    query = " ".join(query_parts) + ";"

    try:
        print(f"Executing Correlation Query: {query} with params: {params}")
        records = await conn.fetch(query, *params)

        if not records:
             raise HTTPException(
                 status_code=status.HTTP_404_NOT_FOUND,
                 detail=f"No se encontraron datos para la ciudad '{city_name}' con las variables '{variable_x}' y '{variable_y}' para el rango de fechas especificado."
             )

        # Convierte los registros a una lista de diccionarios.
        result_list = [
            {"time": r['time'], variable_x: r[variable_x], variable_y: r[variable_y]}
            for r in records
        ]
        return result_list

    except HTTPException:
        raise
    except asyncpg.exceptions.UndefinedColumnError:
         # Ocurre si una de las columnas de variable no existe.
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Una o ambas columnas de base de datos ('{variable_x}', '{variable_y}') no existen."
        )
    except Exception as e:
        print(f"!!!!!!!! ERROR during correlation query for {city_name}: {e} !!!!!!!!")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor al recuperar datos de correlación para {city_name}."
        )


@app.get("/weather/averages", response_model=List[Dict[str, Any]])
async def get_comparative_averages(
    variable_name: str = Query(..., description=f"Variable a promediar. Permitidas: {', '.join(ALLOWED_WEATHER_VARIABLES)}"),
    granularity: str = Query(..., description=f"Granularidad temporal para promediar. Permitidas: {', '.join(ALLOWED_AVG_GRANULARITY)}"),
    cities: str = Query(..., description="Lista de nombres de ciudades separadas por comas (ej., Bogota,Medellin)"),
    start_date: Optional[str] = Query(None, description="Filtro de fecha de inicio (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filtro de fecha de fin (YYYY-MM-DD)"),
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Calcula valores promedio para una variable en múltiples ciudades, agrupados por período de tiempo (hora, día, día de la semana, mes).
    """
    validate_variable_name(variable_name)
    if granularity not in ALLOWED_AVG_GRANULARITY:
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Granularidad inválida '{granularity}'. Valores permitidos: {', '.join(ALLOWED_AVG_GRANULARITY)}"
        )

    cities_list = parse_cities_list(cities)
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)

    # Determina la expresión SQL para agrupar y el nombre del período basado en la granularidad.
    period_alias = "period_value" # Alias consistente para el valor del período en la consulta SQL.
    if granularity == 'hourly':
        # Extrae la hora del día (0-23).
        grouping_expression = "EXTRACT(hour FROM time)"
        period_name = "hour" # Nombre descriptivo para la clave en la respuesta JSON.
    elif granularity == 'daily':
        # Trunca la fecha al inicio del día.
        grouping_expression = "DATE_TRUNC('day', time)"
        period_name = "period_start"
    elif granularity == 'weekly':
        # Extrae el día de la semana según ISO (1=Lunes, 7=Domingo).
        grouping_expression = "EXTRACT(isodow FROM time)"
        period_name = "day_of_week_iso"
    elif granularity == 'monthly':
        # Extrae el número del mes (1=Enero, 12=Diciembre).
        grouping_expression = "EXTRACT(month FROM time)"
        period_name = "month"
    else:
         # Salvaguarda, aunque la validación previa debería prevenir esto.
         raise HTTPException(status_code=500, detail="Error interno de mapeo para granularidad.")

    avg_col_alias = f"average_{variable_name}"

    query_parts = [
        f"SELECT {grouping_expression} AS {period_alias},",
        "city,",
        # Calcula el promedio de la variable especificada.
        f"AVG({variable_name}) AS {avg_col_alias}",
        "FROM weather_data",
        # Filtra por las ciudades solicitadas usando ANY con un array de texto.
        # Es eficiente para comparar contra una lista de valores. El casting a text[] es importante.
        # Se compara en minúsculas para coincidir con `cities_list`.
        "WHERE lower(city) = ANY($1::text[])" # $1 será la lista de ciudades en minúsculas.
    ]
    params: List[Any] = [cities_list] # El primer parámetro es la lista de ciudades.
    param_index = 2 # Los siguientes parámetros empiezan desde $2.

    # Aplica filtros de fecha antes de la agregación.
    if start_dt:
        query_parts.append(f"AND time >= ${param_index}")
        params.append(start_dt)
        param_index += 1
    if end_dt:
        inclusive_end_dt = end_dt + timedelta(days=1)
        query_parts.append(f"AND time < ${param_index}")
        params.append(inclusive_end_dt)
        param_index += 1

    # Asegura que solo se promedien valores no nulos.
    query_parts.append(f"AND {variable_name} IS NOT NULL")

    # Agrupa por el período calculado y por ciudad para obtener promedios separados.
    query_parts.append(f"GROUP BY {period_alias}, city")
    # Ordena para una salida consistente.
    query_parts.append(f"ORDER BY city, {period_alias} ASC")
    query = " ".join(query_parts) + ";"


    try:
        print(f"Executing Averages Query: {query} with params: {params}")
        records = await conn.fetch(query, *params)

        if not records:
             raise HTTPException(
                 status_code=status.HTTP_404_NOT_FOUND,
                 detail=f"No se encontraron datos promedio para la variable '{variable_name}' en las ciudades '{', '.join(cities_list)}' con granularidad '{granularity}' para el rango de fechas especificado."
             )

        # Convierte los registros, usando el nombre de período descriptivo (hour, day_of_week_iso, etc.).
        result_list = []
        for r in records:
            record_dict = {
                period_name: r[period_alias], # Usa el nombre de clave descriptivo.
                "city": r['city'],
                # AVG puede devolver NULL si no hay filas (o si todos los valores son NULL).
                avg_col_alias: r[avg_col_alias] if r[avg_col_alias] is not None else None
            }
            result_list.append(record_dict)

        return result_list

    except HTTPException:
        raise
    except asyncpg.exceptions.UndefinedColumnError:
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"La columna de base de datos '{variable_name}' no existe."
        )
    except Exception as e:
        print(f"!!!!!!!! ERROR during averages query: {e} !!!!!!!!")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor al recuperar datos promedio comparativos."
        )

@app.get("/weather/{city_name}", response_model=list[WeatherRecord])
async def get_city_weather(
    city_name: str,
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Obtiene todos los registros climáticos históricos para una ciudad específica, ordenados por fecha descendente.
    """
    print(f"--- Iniciando consulta para: {city_name} ---")
    # Aunque la consulta usa lower(), capitalizar aquí podría ser para consistencia en logs.
    formatted_city_name = city_name.capitalize()
    query = """
        SELECT id, time, temperature_2m, relative_humidity_2m, dew_point_2m,
               precipitation, wind_speed_10m, uv_index, pressure_msl,
               shortwave_radiation, cloud_cover, city
        FROM weather_data
        WHERE lower(city) = lower($1) -- Comparación insensible a mayúsculas/minúsculas.
        ORDER BY time DESC;
    """
    try:
        print(f"Ejecutando fetch para {formatted_city_name}...")
        records = await conn.fetch(query, formatted_city_name) # Pasa el nombre como parámetro.
        print(f"Fetch completado. Registros encontrados: {len(records)}")

        if not records:
            print(f"No se encontraron datos para {city_name}, devolviendo 404.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontraron datos para la ciudad: {city_name}"
            )
        # Convierte los registros de asyncpg (que son similares a tuplas con acceso por nombre)
        # a una lista de diccionarios, compatible con el response_model Pydantic.
        result_list = [dict(record) for record in records]

        print(f"Devolviendo {len(result_list)} registros convertidos a dict para {city_name}.")
        return result_list

    except HTTPException:
        # Re-lanza la excepción HTTP 404 si se generó explícitamente.
        raise
    except Exception as e:
        # Captura cualquier otro error durante la ejecución de la consulta o procesamiento.
        print(f"!!!!!!!! ERROR durante la consulta de clima para {city_name}: {e} !!!!!!!!")
        traceback.print_exc() # Imprime el traceback completo para depuración.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor al obtener datos para {city_name}."
        )