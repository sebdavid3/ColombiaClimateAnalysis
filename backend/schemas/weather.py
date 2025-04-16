from pydantic import BaseModel
from datetime import datetime

class WeatherRecord(BaseModel):
    id: int
    time: datetime
    temperature_2m: float | None = None
    relative_humidity_2m: float | None = None
    dew_point_2m: float | None = None
    precipitation: float | None = None
    wind_speed_10m: float | None = None
    uv_index: float | None = None
    pressure_msl: float | None = None
    shortwave_radiation: float | None = None
    cloud_cover: float | None = None
    city: str

    class Config:
        from_attributes = True # Permite crear el modelo desde atributos de objeto (como registros de BD)
