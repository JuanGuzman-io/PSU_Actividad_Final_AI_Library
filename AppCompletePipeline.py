from AIlibrary import data_clean, FraudDetectionLibrary
import pandas as pd


def complete_pipeline_demo():
    """
    Demostración del pipeline completo desde datos crudos hasta predicciones.
    """

    print("🔄 PIPELINE COMPLETO: DATOS CRUDOS → MODELO → PREDICCIONES")
    print("="*60)

    #
    print("\n🧹 PASO 1: Limpieza de datos crudos")
    print("-"*40)

    try:

        data_path = 'credit_card_transactions.csv'
        outlier_limit = 2700
        columns_to_remove = [
            'Unnamed: 0', 'first', 'last', 'street', 'city', 'state', 'zip',
            'trans_num', 'unix_time', 'merch_zipcode'
        ]

        cleaned_df = data_clean(data_path, outlier_limit, columns_to_remove)

        print(f"✅ Limpieza completada: {cleaned_df.shape}")
        print(f"📋 Columnas finales: {list(cleaned_df.columns)}")

        if 'is_fraud' in cleaned_df.columns:
            fraud_dist = cleaned_df['is_fraud'].value_counts()
            print(f"📊 Distribución de fraude: {dict(fraud_dist)}")
        elif 'fraud' in cleaned_df.columns:

            cleaned_df = cleaned_df.rename(columns={'fraud': 'is_fraud'})
            fraud_dist = cleaned_df['is_fraud'].value_counts()
            print(f"📊 Distribución de fraude: {dict(fraud_dist)}")
        elif 'isFraud' in cleaned_df.columns:

            cleaned_df = cleaned_df.rename(columns={'isFraud': 'is_fraud'})
            fraud_dist = cleaned_df['is_fraud'].value_counts()
            print(f"📊 Distribución de fraude: {dict(fraud_dist)}")
        else:
            print("⚠️ No se encontró columna de fraude. Columnas disponibles:")
            print(list(cleaned_df.columns))
            return None

    except FileNotFoundError:
        print("❌ Archivo crudo no encontrado. Usando datos ya procesados...")
        cleaned_df = pd.read_csv('data.csv')
        print(f"✅ Datos cargados: {cleaned_df.shape}")

    print("\n🤖 PASO 2: Entrenamiento del modelo")
    print("-"*40)

    detector = FraudDetectionLibrary()

    train_params = {
        'algorithm': 'random_forest',
        'n_estimators': 100,
        'max_depth': 10,
        'test_size': 0.2,
        'use_smote': True,
        'cv_folds': 5,
        'random_state': 42
    }

    if cleaned_df is None:
        print("❌ No se puede continuar sin columna de fraude")
        return None

    results = detector.train_model(cleaned_df, train_params=train_params)

    print("\n💾 PASO 3: Persistencia del modelo")
    print("-"*40)

    model_name = "fraud_detector_pipeline"
    detector.save_model(model_name)

    print("\n📂 PASO 4: Carga y uso del modelo")
    print("-"*40)

    new_detector = FraudDetectionLibrary()
    new_detector.load_model(model_name)

    print("\n🔍 PASO 5: Predicciones en tiempo real")
    print("-"*40)

    sample_data = cleaned_df.drop(
        'is_fraud', axis=1).sample(100, random_state=42)

    prediction_results = new_detector.test_model(sample_data)

    print(f"📊 Resultados de predicción:")
    print(
        f"   - Total transacciones analizadas: {prediction_results['total_transactions']}")
    print(f"   - Fraudes detectados: {prediction_results['fraud_count']}")
    print(
        f"   - Tasa de fraude: {prediction_results['fraud_percentage']:.2f}%")

    print("\n🚨 PASO 6: Análisis de alto riesgo")
    print("-"*40)

    high_risk_threshold = 0.8
    fraud_probs = prediction_results['fraud_probability']
    high_risk_cases = fraud_probs > high_risk_threshold

    if high_risk_cases.sum() > 0:
        print(
            f"⚠️  Casos de alto riesgo (>{high_risk_threshold:.0%}): {high_risk_cases.sum()}")
        high_risk_data = sample_data[high_risk_cases]

        if len(high_risk_data) > 0:
            print("\n📋 Características promedio de casos de alto riesgo:")
            numeric_cols = ['amt', 'trans_hour', 'card_holder_age', 'distance']
            available_cols = [
                col for col in numeric_cols if col in high_risk_data.columns]

            for col in available_cols:
                avg_val = high_risk_data[col].mean()
                print(f"   - {col}: {avg_val:.2f}")
    else:
        print("✅ No se detectaron casos de muy alto riesgo en esta muestra")

    print("\n🎉 ¡PIPELINE COMPLETO EJECUTADO EXITOSAMENTE!")
    print("="*60)

    return {
        'cleaned_data': cleaned_df,
        'model': detector,
        'training_results': results,
        'prediction_results': prediction_results
    }


def simple_usage_example():
    """
    Ejemplo de uso simple para usuarios rápidos.
    """

    print("\n🚀 EJEMPLO DE USO SIMPLE")
    print("="*30)

    print("\n📁 Opción A: Desde datos crudos")
    print("```python")
    print("from AIlibrary import data_clean, FraudDetectionLibrary")
    print("")
    print("# 1. Limpiar datos")
    print("df = data_clean('datos_crudos.csv')")
    print("")
    print("# 2. Entrenar modelo")
    print("detector = FraudDetectionLibrary()")
    print("results = detector.train_model(df)")
    print("")
    print("# 3. Guardar modelo")
    print("detector.save_model('mi_modelo')")
    print("")
    print("# 4. Usar modelo")
    print("predictions = detector.test_model(nuevos_datos)")
    print("```")

    # Opción B: Si tienes datos ya limpios
    print("\n📁 Opción B: Desde datos ya procesados")
    print("```python")
    print("from AIlibrary import FraudDetectionLibrary")
    print("import pandas as pd")
    print("")
    print("# 1. Cargar datos limpios")
    print("df = pd.read_csv('data.csv')")
    print("")
    print("# 2. Entrenar y usar")
    print("detector = FraudDetectionLibrary()")
    print("results = detector.train_model(df)")
    print("predictions = detector.test_model(nuevos_datos)")
    print("```")


if __name__ == "__main__":
    # Ejecutar demo completo
    pipeline_results = complete_pipeline_demo()

    # Mostrar ejemplos de uso
    simple_usage_example()

    print("\n💡 Para más detalles, revisa App.py o la documentación en README.md")
