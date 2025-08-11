
## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

*   Python 3.9 or higher installed.
*   `pip` and `venv` (usually included with Python).
*   A running and accessible PostgreSQL server.
*   Git (optional, for cloning the repository).

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/sebdavid3/ColombiaClimateAnalysis.git
    cd ColombiaClimateAnalysis
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows:
    # .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure the database connection:**
    *   Create a `.env` file in the project root.
    *   Add the connection URL for your PostgreSQL database. This is used by both the backend API and the `update_weather.py` script:
        ```dotenv
        # .env
        DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<db_name>
        ```
    *   **Important:** Ensure `.env` is listed in your `.gitignore`.

5.  **Prepare the Database:**
    *   Ensure the database specified in `.env` exists.
*   You can use the following SQL file as a reference or execute it directly:  
    👉 [Download SQL script from OneDrive](https://uninorte-my.sharepoint.com/personal/sdibanez_uninorte_edu_co/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsdibanez%5Funinorte%5Fedu%5Fco%2FDocuments%2Fcolombia%5Fclima%2Esql&parent=%2Fpersonal%2Fsdibanez%5Funinorte%5Fedu%5Fco%2FDocuments&ga=1)

*   Alternatively, you can create the table manually using the following structure (adjust data types as needed):

        ```sql
        CREATE TABLE IF NOT EXISTS weather_data
        (
            id integer NOT NULL DEFAULT nextval('weather_data_id_seq'::regclass),
            "time" timestamp without time zone,
            temperature_2m numeric,
            relative_humidity_2m numeric,
            dew_point_2m numeric,
            precipitation numeric,
            wind_speed_10m numeric,
            uv_index numeric,
            pressure_msl numeric,
            shortwave_radiation numeric,
            cloud_cover numeric,
            city character varying(255) COLLATE pg_catalog."default",
            CONSTRAINT weather_data_pkey PRIMARY KEY (id),
            CONSTRAINT weather_data_city_time_key UNIQUE (city, "time")
        )

        ```
    *   **Initial Data Load:**
        *   Place your raw city CSV files (e.g., `Bogota.csv`, `Medellin.csv`) into a `Data/raw data/` directory (create it if it doesn't exist).
        *   Run the `combine_data.py` script to merge them into `Data/clean data/combined_cities.csv`:
            ```bash
            python scripts/combine_data.py
            ```
        *   Import the data from `Data/clean data/combined_cities.csv` into your `weather_data` table. You can use tools like `psql`'s `\copy` command, DBeaver, pgAdmin, or a custom script. Example using `psql`:
            ```sql
            -- Make sure column names in CSV match table columns (or specify mapping)
            -- Check CSV header: 'record_id', 'time', 'temperature_2m', ..., 'Ciudad' (maps to 'city')
            \copy weather_data(record_id, time, temperature_2m, relative_humidity_2m, dew_point_2m, precipitation, wind_speed_10m, uv_index, pressure_msl, shortwave_radiation, cloud_cover, city) FROM 'C:/path/to/your/project/Data/clean data/combined_cities.csv' WITH (FORMAT CSV, HEADER);
            ```
            *(Adjust the path and column mapping as needed based on your CSV and table structure)*.

## ▶️ Running the Application

You need to run the backend API and the frontend dashboard separately.

1.  **Run the Backend API:**
    Open a terminal, activate the virtual environment, and run Uvicorn:
    ```bash
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload`: Automatically reloads the server on code changes.
    *   `--host 0.0.0.0`: Makes the API accessible from your network (use `127.0.0.1` for local access only).
    *   `--port 8000`: Specifies the port (Streamlit app expects this by default).
    The API documentation will be available at `http://localhost:8000/docs`.

2.  **Run the Frontend Dashboard:**
    Open *another* terminal, activate the virtual environment, and run Streamlit:
    ```bash
    streamlit run app.py
    ```
    Streamlit will typically open the dashboard automatically in your web browser.

### Updating Data

The dashboard triggers a data update automatically on page load. It also queries the backend status endpoint to display data freshness. No manual update controls are shown in the UI.

If you need to update manually outside the app, run the script:

To fetch the latest weather data from Open-Meteo manually, run the update script:

```bash
python scripts/update_weather.py
```
This script will:
*   Check the latest timestamp for each city in the database.
*   Fetch new data from Open-Meteo's Forecast and Archive APIs since the last entry.
*   Process and insert the new records, avoiding duplicates.
*   Handles timezone conversions (API uses UTC, database stores local time).

You can schedule this script to run periodically (e.g., using cron on Linux/macOS or Task Scheduler on Windows) to keep the data up-to-date.

### ☁️ Cloud Deployment with Auto-Update

For production deployment in the cloud, this project includes several deployment configurations and automatic update mechanisms:

**Supported Platforms:**
- Railway (includes free PostgreSQL)
- Render.com (good free tier)
- Heroku (classic option)
- AWS/GCP/Azure (Docker)

**Auto-Update Options:**
- GitHub Actions (recommended - runs every 30 minutes)
- Platform cron jobs (Railway/Render scheduler)
- External cron services (EasyCron, cron-job.org)

**Key Features for Cloud:**
- 🌤️ **Smart Environment Detection**: Automatically switches between local and remote update methods
- 📊 **Real-time Status API**: Monitor data freshness via `/update-weather/status`
- 🔄 **Remote Update Trigger**: Update data via API endpoint `/update-weather`
- 🏠/☁️ **Environment Indicator**: Dashboard shows if running locally or in cloud

**Quick Cloud Setup:**
1. Choose a platform (Railway recommended for beginners)
2. Connect your GitHub repository
3. Add PostgreSQL database
4. Configure environment variables (`DATABASE_URL`)
5. Enable GitHub Actions for auto-updates
6. Deploy!

📖 **Complete deployment guide:** See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed step-by-step instructions.

### Configuration Files Included:
- `Procfile` - Railway/Heroku deployment
- `railway.toml` - Railway-specific config
- `render.yaml` - Render.com multi-service setup
- `Dockerfile` + `start.sh` - Container deployment
- `.github/workflows/update-weather.yml` - GitHub Actions scheduler

## 📄 API Endpoints

The backend API provides the following endpoints (base URL: `http://localhost:8000`):

*   `GET /`: Welcome message.
*   `GET /db-test`: Tests the database connection.
*   `GET /weather/{city_name}`: Gets all historical records for a specific city.
*   `GET /weather/{city_name}/trends`: Gets time series data for a specific variable and city, with optional date filtering. (Used for trend charts).
    *   Params: `variable_name`, `start_date`, `end_date`
*   `GET /weather/{city_name}/precipitation/summary`: Gets aggregated precipitation totals (daily, weekly, monthly) for a city. (Used for precipitation charts).
    *   Params: `granularity`, `start_date`, `end_date`
*   `GET /weather/current`: Gets the most recent value of a variable for all cities. (Potentially useful for a live map, though the dashboard uses averages/totals over a period).
    *   Params: `variable_name`
*   `GET /weather/{city_name}/correlation`: Gets paired values for two variables for correlation analysis. (Used for scatter plots).
    *   Params: `variable_x`, `variable_y`, `start_date`, `end_date`
*   `GET /weather/averages`: Calculates average values for a variable across multiple cities, grouped by time period (hourly, daily, weekly, monthly). (Used for comparative charts and period map).
    *   Params: `variable_name`, `granularity`, `cities` (comma-separated), `start_date`, `end_date`

*(Refer to `http://localhost:8000/docs` while the API is running for detailed interactive documentation.)*


## 👨‍💻 Developers

*   Sebastian Ibañez ([GitHub](https://github.com/sebdavid3))
*   Daniel Cruzado ([GitHub](https://github.com/AlexDanii))