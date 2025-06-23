

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')


def data_clean(data_path, outlier_limit=2700, columns_to_remove=None):
    """
    Función de limpieza y preprocesamiento de datos de transacciones de tarjeta de crédito.

    Args:
        data_path (str): Ruta al archivo CSV con datos crudos
        outlier_limit (float): Límite superior para filtrar outliers en 'amt'
        columns_to_remove (list): Lista de columnas a eliminar

    Returns:
        pd.DataFrame: DataFrame limpio y preprocesado listo para entrenamiento
    """

    if columns_to_remove is None:
        columns_to_remove = [
            'Unnamed: 0', 'first', 'last', 'street', 'city', 'state', 'zip',
            'trans_num', 'unix_time', 'merch_zipcode'
        ]

    print(f"Datos cargados exitosamente desde: {data_path}")

    try:
        df = pd.read_csv(data_path)
        print(f"Forma original de los datos: {df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")

    initial_count = len(df)
    df = df[df['amt'] <= outlier_limit]
    removed_count = initial_count - len(df)
    print(
        f"Se eliminaron {removed_count} transacciones con 'amt' superior a {outlier_limit}.")
    print(f"Forma después de filtrar por 'amt': {df.shape}")

    existing_columns_to_remove = [
        col for col in columns_to_remove if col in df.columns]
    if existing_columns_to_remove:
        df = df.drop(columns=existing_columns_to_remove)
        print(f"Columnas eliminadas: {', '.join(existing_columns_to_remove)}")
        print(f"Forma después de eliminar columnas: {df.shape}")

    print("Procesando columnas de fecha...")
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(
            df['trans_date_trans_time'])

        df['trans_year'] = df['trans_date_trans_time'].dt.year
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        df['trans_day'] = df['trans_date_trans_time'].dt.day
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_minute'] = df['trans_date_trans_time'].dt.minute
        df['trans_second'] = df['trans_date_trans_time'].dt.second
        df['trans_weekday'] = df['trans_date_trans_time'].dt.dayofweek

        df['trans_season'] = df['trans_month'].apply(lambda x:
                                                     1 if x in [12, 1, 2] else
                                                     2 if x in [3, 4, 5] else
                                                     3 if x in [6, 7, 8] else
                                                     4)

        df = df.drop('trans_date_trans_time', axis=1)

    print("Columnas de fecha procesadas.")

    print("Calculando la edad del titular de la tarjeta...")
    if 'dob' in df.columns and 'trans_year' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'])
        df['card_holder_age'] = df['trans_year'] - df['dob'].dt.year
        df = df.drop('dob', axis=1)
    print("Edad del titular calculada.")

    print("Calculando la distancia geográfica de la transacción...")
    initial_len = len(df)

    def calculate_distance(row):
        try:
            if pd.isna(row['lat']) or pd.isna(row['long']) or pd.isna(row['merch_lat']) or pd.isna(row['merch_long']):
                return np.nan
            customer_location = (row['lat'], row['long'])
            merchant_location = (row['merch_lat'], row['merch_long'])
            return geodesic(customer_location, merchant_location).kilometers
        except:
            return np.nan

    if all(col in df.columns for col in ['lat', 'long', 'merch_lat', 'merch_long']):
        df['distance'] = df.apply(calculate_distance, axis=1)

        df = df.dropna(subset=['distance'])
        removed_distance = initial_len - len(df)
        print(
            f"Se eliminaron {removed_distance} filas donde no se pudo calcular la distancia.")

    print("Distancia geográfica calculada.")
    print(f"Forma después de calcular distancia y manejar nulos: {df.shape}")

    print("Codificando columnas categóricas...")
    categorical_columns = df.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        if col != 'is_fraud':
            df[col] = pd.Categorical(df[col]).codes

    if 'is_fraud' in df.columns:
        df['is_fraud'] = df['is_fraud'].astype(int)

    print("Columnas categóricas codificadas.")
    print(
        f"Forma final del DataFrame después de la limpieza y procesamiento: {df.shape}")

    return df


def clean_and_train(data_path, outlier_limit=2700, columns_to_remove=None, train_params=None):
    """
    Pipeline completo: limpiar datos y entrenar modelo.
    """
    from AIlibrary import FraudDetectionLibrary

    cleaned_data = data_clean(data_path, outlier_limit, columns_to_remove)

    detector = FraudDetectionLibrary()
    results = detector.train_model(cleaned_data, train_params=train_params)

    return detector, results, cleaned_data


if __name__ == "__main__":

    print("🧹 DEMO: Limpieza de datos")
    print("="*40)

    try:
        cleaned_df = data_clean('/content/credit_card_transactions.csv')
        print("\n✅ Limpieza completada exitosamente!")
        print(f"📊 Dataset final: {cleaned_df.shape}")
        print(f"📋 Columnas: {list(cleaned_df.columns)}")

        if 'is_fraud' in cleaned_df.columns:
            print(
                f"📈 Distribución de fraude: {cleaned_df['is_fraud'].value_counts().to_dict()}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Ajusta la ruta del archivo según tu configuración")
