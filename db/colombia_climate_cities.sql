CREATE TABLE weather_data (
    id SERIAL PRIMARY KEY,
    time TIMESTAMP,
    temperature_2m DECIMAL,
    relative_humidity_2m DECIMAL,
    dew_point_2m DECIMAL,
    precipitation DECIMAL,
    wind_speed_10m DECIMAL,
    uv_index DECIMAL,
    pressure_msl DECIMAL,
    shortwave_radiation DECIMAL,
    cloud_cover DECIMAL,
    city VARCHAR(255)
);