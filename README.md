# Colombian Weather Data 

> RESTful API built with FastAPI to provide historical and current weather data for various Colombian cities, optimized for powering dynamic dashboards.

This API connects to a PostgreSQL database (using `asyncpg` for asynchronous operations) to query and serve detailed meteorological information.

## ✨ Key Features

* **Historical Data:** Access time series of weather variables (temperature, humidity, wind, etc.) for specific cities.
* **Data Aggregation:** Get sums or averages of variables (like precipitation) grouped by day, week, or month.
* **Current Data:** Query the latest available weather record for all monitored cities, ideal for maps.
* **Correlation Analysis:** Retrieve data for two variables simultaneously for correlation analysis over a given period.
* **Cross-City Comparisons:** Calculate and compare hourly, daily (day of the week), or monthly averages of variables across multiple cities.
* **Dynamic Filtering:** Filter data by city, date range, specific variable, and time granularity.
* **Asynchronous:** Fully asynchronous using FastAPI and `asyncpg` for high performance.

## 🛠️ Tech Stack

* **Backend Framework:** [FastAPI](https://fastapi.tiangolo.com/)
* **Database:** PostgreSQL
* **Database Driver:** [asyncpg](https://github.com/MagicStack/asyncpg)
* **Language:** Python 3.9+
* **Data Validation:** Pydantic
* **ASGI Server:** [Uvicorn](https://www.uvicorn.org/)
* **Configuration Management:** Environment variables (e.g., using `python-dotenv`)

## 🚀 Getting Started

Follow these steps to get the API running in your local environment.

### Prerequisites

* Python 3.9 or higher installed.
* `pip` and `venv` (usually included with Python).
* A running and accessible PostgreSQL server.
* Git (optional, for cloning the repository).

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <https://github.com/sebdavid3/ColombiaClimateAnalysis.git>
    cd <DIRECTORY_NAME>
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
    * Create a `.env` file in the project root (either the `backend` directory or the main one).
    * Add the connection URL for your PostgreSQL database:
        ```dotenv
        # .env
        DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<db_name>
        ```
    * **Important:** Make sure the `.env` file is listed in your `.gitignore` to avoid committing sensitive credentials.

5.  **Prepare the Database:**
    * Ensure the `weather_data` table (or whatever it's named) exists in your database with the expected structure (columns `time`, `city`, `temperature_2m`, etc.).
    * (Optional) If using migration tools like Alembic, run the migrations.

## ▶️ Running the Application

Once installed and configured, run the API using Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000