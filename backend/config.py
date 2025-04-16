from pydantic_settings import BaseSettings, SettingsConfigDict
import os

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')


class Settings(BaseSettings):

    database_url: str 

    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding='utf-8', extra='ignore')

settings = Settings()