from pydantic_settings import BaseSettings, SettingsConfigDict
import os

# Construye la ruta al archivo .env asumiendo que está dos niveles por encima del directorio de este script.
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')


class Settings(BaseSettings):
    """
    Define la configuración de la aplicación cargada desde variables de entorno o un archivo .env.
    """
    database_url: str

    # Configuración para Pydantic: especifica el archivo .env, su codificación y permite variables extra.
    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding='utf-8', extra='ignore')

# Instancia única de la configuración, disponible para ser importada en otros módulos.
settings = Settings()