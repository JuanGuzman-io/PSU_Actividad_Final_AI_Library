import pandas as pd
import numpy as np
from AIlibrary import FraudDetectionLibrary
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def main():
    """
    Función principal que demuestra el uso completo de la librería.
    """
    print("🚀 DEMO: Librería de Detección de Fraude")
    print("="*50)

    print("\n📁 Paso 1: Cargando datos...")

    data_path = "credit_card_transactions.csv"

    try:
        df = pd.read_csv(data_path)
        print(
            f"✅ Datos cargados: {df.shape[0]} transacciones, {df.shape[1]} características")
    except FileNotFoundError:
        print("❌ Archivo credit_card_transactions.csv no encontrado.")
        print("💡 Por favor, transfiere el archivo desde Colab a este directorio.")

        print("🔄 Generando datos sintéticos para demostración...")
        df = create_synthetic_data()

    print(f"\n📊 Información del dataset:")
    print(f"   - Total de transacciones: {len(df)}")
    print(f"   - Características: {df.columns.tolist()}")

    if 'is_fraud' in df.columns:
        fraud_distribution = df['is_fraud'].value_counts()
        print(f"   - Distribución de fraude: {dict(fraud_distribution)}")
    else:
        print("⚠️  Columna 'is_fraud' no encontrada. Creando etiquetas sintéticas...")
        df['is_fraud'] = create_fraud_labels(df)

    print("\n📋 Paso 2: Preparando datos para entrenamiento y prueba...")

    train_data, test_data = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['is_fraud'])

    print(f"   - Datos de entrenamiento: {len(train_data)} transacciones")
    print(f"   - Datos para predicción: {len(test_data)} transacciones")

    print("\n🤖 Paso 3: Entrenando modelo de detección de fraude...")

    fraud_detector = FraudDetectionLibrary()

    train_params = {
        'algorithm': 'random_forest',
        'n_estimators': 100,
        'max_depth': 10,
        'test_size': 0.2,
        'use_smote': True,
        'cv_folds': 5,
        'random_state': 42
    }

    training_results = fraud_detector.train_model(
        train_data, train_params=train_params)

    print("\n💾 Paso 4: Guardando modelo entrenado...")

    model_name = "fraud_detector_v1"
    save_path = fraud_detector.save_model(model_name)

    print("\n📂 Paso 5: Cargando modelo (simulando uso posterior)...")

    new_detector = FraudDetectionLibrary()
    new_detector.load_model(model_name)

    print("\n🔍 Paso 6: Realizando predicciones sobre datos nuevos...")

    test_features = test_data.drop(columns=['is_fraud'])

    prediction_results = new_detector.test_model(test_features)

    print("\n📊 Paso 7: Evaluando calidad de las predicciones...")

    y_true = test_data['is_fraud'].values
    y_pred = prediction_results['predictions']

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    accuracy = accuracy_score(y_true, y_pred)
    print(f"🎯 Precisión en datos nuevos: {accuracy:.4f}")

    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_true, y_pred))

    print("\n🔢 Matriz de confusión:")
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)

    print("\n📈 Paso 8: Generando visualizaciones...")

    create_visualizations(training_results, prediction_results, y_true, y_pred)

    print("\n🚨 Paso 9: Análisis de transacciones de alto riesgo...")

    analyze_high_risk_transactions(test_data, prediction_results)

    print("\n✅ Demo completada exitosamente!")
    print("="*50)


def create_synthetic_data(n_samples=1000):
    """
    Crea un dataset sintético para demostración cuando no está disponible el real.
    """
    np.random.seed(42)

    data = {
        'amt': np.random.exponential(50, n_samples),
        'gender': np.random.choice([0, 1], n_samples),
        'lat': np.random.uniform(25, 50, n_samples),
        'long': np.random.uniform(-125, -65, n_samples),
        'city_pop': np.random.exponential(10000, n_samples),
        'job': np.random.choice(range(20), n_samples),
        'merch_lat': np.random.uniform(25, 50, n_samples),
        'merch_long': np.random.uniform(-125, -65, n_samples),
        'category': np.random.choice(range(15), n_samples),
        'trans_hour': np.random.choice(range(24), n_samples),
        'trans_weekday': np.random.choice(range(7), n_samples),
        'card_holder_age': np.random.normal(45, 15, n_samples),
        'distance': np.random.exponential(20, n_samples)
    }

    for i in range(10):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)

    df = pd.DataFrame(data)

    df['is_fraud'] = create_fraud_labels(df)

    return df


def create_fraud_labels(df):
    """
    Crea etiquetas de fraude basadas en reglas heurísticas.
    """
    fraud_conditions = (
        (df['amt'] > 1000) |
        (df['trans_hour'].isin([2, 3, 4])) |
        (df.get('distance', 0) > 100)
    )

    random_fraud = np.random.random(len(df)) < 0.02

    return ((fraud_conditions) | random_fraud).astype(int)


def create_visualizations(training_results, prediction_results, y_true, y_pred):
    """
    Crea visualizaciones de los resultados.
    """
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.heatmap(training_results['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Matriz de Confusión - Entrenamiento')
    axes[0, 0].set_ylabel('Valores Reales')
    axes[0, 0].set_xlabel('Predicciones')

    if training_results['feature_importance'] is not None:
        top_features = training_results['feature_importance'].head(10)
        axes[0, 1].barh(top_features['feature'], top_features['importance'])
        axes[0, 1].set_title('Top 10 Características más Importantes')
        axes[0, 1].set_xlabel('Importancia')

    fraud_probs = prediction_results['fraud_probability']
    axes[1, 0].hist(fraud_probs[y_true == 0], bins=50,
                    alpha=0.7, label='Legítimo', color='green')
    axes[1, 0].hist(fraud_probs[y_true == 1], bins=50,
                    alpha=0.7, label='Fraude', color='red')
    axes[1, 0].set_title('Distribución de Probabilidades de Fraude')
    axes[1, 0].set_xlabel('Probabilidad de Fraude')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].legend()

    metrics = ['Precisión', 'Recall', 'F1-Score', 'AUC-ROC']
    values = [
        training_results['precision'],
        training_results['recall'],
        training_results['f1_score'],
        training_results['auc_score']
    ]

    bars = axes[1, 1].bar(metrics, values, color=[
                          'skyblue', 'lightgreen', 'orange', 'pink'])
    axes[1, 1].set_title('Métricas de Rendimiento')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)

    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('fraud_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("📊 Visualizaciones guardadas como 'fraud_detection_results.png'")


def analyze_high_risk_transactions(test_data, prediction_results):
    """
    Analiza las transacciones identificadas como de alto riesgo.
    """

    high_risk_threshold = 0.8
    high_risk_indices = prediction_results['fraud_probability'] > high_risk_threshold

    if np.sum(high_risk_indices) > 0:
        high_risk_transactions = test_data[high_risk_indices]

        print(
            f"🚨 Encontradas {len(high_risk_transactions)} transacciones de alto riesgo (>80% probabilidad)")
        print("\n📋 Características promedio de transacciones de alto riesgo:")

        numeric_columns = high_risk_transactions.select_dtypes(
            include=[np.number]).columns
        summary = high_risk_transactions[numeric_columns].mean()

        for col, value in summary.items():
            if col != 'is_fraud':
                print(f"   - {col}: {value:.2f}")

        print(f"\n🔍 Ejemplo de transacciones de alto riesgo:")
        display_columns = ['amt', 'trans_hour', 'card_holder_age', 'distance'] if 'distance' in high_risk_transactions.columns else [
            'amt', 'trans_hour', 'card_holder_age']
        if all(col in high_risk_transactions.columns for col in display_columns):
            print(high_risk_transactions[display_columns].head())
    else:
        print("✅ No se encontraron transacciones de muy alto riesgo (>80% probabilidad)")


def demo_real_time_prediction():
    """
    Demuestra cómo usar la librería para predicciones en tiempo real.
    """
    print("\n🔄 DEMO: Predicción en Tiempo Real")
    print("-"*30)

    detector = FraudDetectionLibrary()

    try:
        detector.load_model("fraud_detector_v1")

        new_transaction = pd.DataFrame({
            'amt': [1500.0],
            'gender': [1],
            'lat': [40.7128],
            'long': [-74.0060],
            'city_pop': [8000000.0],
            'job': [5],
            'merch_lat': [40.7589],
            'merch_long': [-73.9851],
            'category': [2],
            'trans_hour': [3],
            'trans_weekday': [1],
            'card_holder_age': [35.0],
            'distance': [15.5]
        })

        for i in range(10):
            if f'feature_{i}' not in new_transaction.columns:
                new_transaction[f'feature_{i}'] = [0.0]

        missing_features = set(detector.feature_names) - \
            set(new_transaction.columns)
        for feature in missing_features:
            new_transaction[feature] = [0.0]

        result = detector.test_model(new_transaction)

        fraud_prob = result['fraud_probability'][0]
        is_fraud = result['predictions'][0]

        print(f"💳 Nueva transacción analizada:")
        print(f"   - Monto: ${new_transaction['amt'].iloc[0]:.2f}")
        print(f"   - Hora: {new_transaction['trans_hour'].iloc[0]}:00")
        print(f"   - Probabilidad de fraude: {fraud_prob:.1%}")
        print(
            f"   - Clasificación: {'🚨 FRAUDE' if is_fraud else '✅ LEGÍTIMA'}")

        if fraud_prob > 0.5:
            print("⚠️  Recomendación: Revisar manualmente esta transacción")

    except FileNotFoundError:
        print("❌ Modelo no encontrado. Ejecute primero el entrenamiento completo.")


if __name__ == "__main__":

    main()

    demo_real_time_prediction()

    print("\n🎉 ¡Todas las demostraciones completadas!")
    print("💡 La librería está lista para usar en aplicaciones de producción.")
