import os
import pandas as pd

# Define directorios de entrada y salida
raw_data_dir = os.path.join('Data', 'raw data')
clean_data_dir = os.path.join('Data', 'clean data')
combined_file_path = os.path.join(clean_data_dir, 'combined_cities.csv')

# Asegura que el directorio de salida exista
os.makedirs(clean_data_dir, exist_ok=True)

all_dataframes = []

print("Iniciando proceso de combinación y limpieza...")

# Itera sobre cada archivo en el directorio de datos brutos
for filename in os.listdir(raw_data_dir):
    if filename.endswith(".csv"):
        # Extrae el nombre de la ciudad del nombre del archivo
        city_name = filename.replace('.csv', '')
        file_path = os.path.join(raw_data_dir, filename)
        
        try:
            # Lee el CSV, ignorando líneas mal formadas que podrían causar errores de análisis.
            df = pd.read_csv(file_path, on_bad_lines='skip') 
            
            # Verifica la existencia y procesa la columna 'time'
            if 'time' in df.columns:
                 # Convierte a datetime; los valores no válidos se transforman en NaT (Not a Time).
                 df['time'] = pd.to_datetime(df['time'], errors='coerce')
                 # Elimina filas donde la conversión de 'time' falló (resultando en NaT).
                 df.dropna(subset=['time'], inplace=True)
            else:
                 # Advierte si la columna esencial 'time' no se encuentra
                 print(f"Advertencia: Columna 'time' no encontrada en {filename}")
                 continue # Salta al siguiente archivo

            # Añade una columna con el nombre de la ciudad
            df['Ciudad'] = city_name
            all_dataframes.append(df)
            print(f"Procesado: {filename}")
            
        except Exception as e:
            # Captura y reporta errores durante el procesamiento del archivo
            print(f"Error procesando {filename}: {e}")

# Procede solo si se cargaron DataFrames
if all_dataframes:
    print("\nConcatenando DataFrames...")
    # Combina todos los DataFrames individuales en uno solo
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Filas antes de eliminar duplicados: {len(combined_df)}")
    # Asegura que 'time' sea tipo datetime antes de eliminar duplicados
    combined_df['time'] = pd.to_datetime(combined_df['time'])
    # Elimina registros duplicados basados en la combinación única de tiempo y ciudad.
    combined_df.drop_duplicates(subset=['time', 'Ciudad'], keep='first', inplace=True)
    print(f"Filas después de eliminar duplicados: {len(combined_df)}")

    # Reconstruye el índice después de eliminar filas
    combined_df.reset_index(drop=True, inplace=True) 
    # Inserta una columna de ID único al principio del DataFrame
    combined_df.insert(0, 'record_id', combined_df.index + 1)
    print("Columna 'record_id' añadida.")

    print(f"Guardando archivo combinado en: {combined_file_path}")
    # Guarda el DataFrame combinado en un nuevo archivo CSV
    combined_df.to_csv(combined_file_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print("\nProceso completado.")
else:
    print("No se encontraron archivos CSV válidos para procesar.")