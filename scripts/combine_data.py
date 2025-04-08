import os
import pandas as pd

raw_data_dir = os.path.join('Data', 'raw data')
clean_data_dir = os.path.join('Data', 'clean data')
combined_file_path = os.path.join(clean_data_dir, 'combined_cities.csv')

os.makedirs(clean_data_dir, exist_ok=True)

all_dataframes = []

print("Iniciando proceso de combinación y limpieza...")

for filename in os.listdir(raw_data_dir):
    if filename.endswith(".csv"):
        city_name = filename.replace('.csv', '')
        file_path = os.path.join(raw_data_dir, filename)
        
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip') 
            
            if 'time' in df.columns:
                 df['time'] = pd.to_datetime(df['time'], errors='coerce')
                 df.dropna(subset=['time'], inplace=True)
            else:
                 print(f"Advertencia: Columna 'time' no encontrada en {filename}")
                 continue 

            df['Ciudad'] = city_name
            all_dataframes.append(df)
            print(f"Procesado: {filename}")
            
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

if all_dataframes:
    print("\nConcatenando DataFrames...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Filas antes de eliminar duplicados: {len(combined_df)}")
    combined_df['time'] = pd.to_datetime(combined_df['time'])
    combined_df.drop_duplicates(subset=['time', 'Ciudad'], keep='first', inplace=True)
    print(f"Filas después de eliminar duplicados: {len(combined_df)}")

    combined_df.reset_index(drop=True, inplace=True) 
    combined_df.insert(0, 'record_id', combined_df.index + 1)
    print("Columna 'record_id' añadida.")

    print(f"Guardando archivo combinado en: {combined_file_path}")
    combined_df.to_csv(combined_file_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print("\nProceso completado.")
else:
    print("No se encontraron archivos CSV válidos para procesar.") 