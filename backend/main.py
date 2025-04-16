import asyncpg
from fastapi import FastAPI, Depends, HTTPException, status
from contextlib import asynccontextmanager
from typing import AsyncGenerator
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
    