import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionLibrary:
    """
    Librería de alto nivel para la detección de fraude en transacciones financieras.

    Proporciona funcionalidades para:
    - Entrenamiento de modelos de clasificación
    - Guardar y cargar modelos entrenados
    - Realizar predicciones sobre nuevos datos
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_metadata = {}

    def train_model(self, data, target_column='is_fraud', train_params=None):
        """
        Entrena un modelo de clasificación para detección de fraude.

        Args:
            data (pd.DataFrame): Conjunto de datos preprocesado
            target_column (str): Nombre de la columna objetivo (default: 'is_fraud')
            train_params (dict): Parámetros de entrenamiento opcionales
                - algorithm: 'random_forest' o 'logistic_regression' (default: 'random_forest')
                - test_size: Proporción para conjunto de prueba (default: 0.2)
                - random_state: Semilla aleatoria (default: 42)
                - use_smote: Aplicar SMOTE para balanceo (default: True)
                - cv_folds: Número de folds para validación cruzada (default: 5)
                - n_estimators: Número de árboles para Random Forest (default: 100)
                - max_depth: Profundidad máxima para Random Forest (default: 10)

        Returns:
            dict: Matriz de confusión y métricas de evaluación
        """

        default_params = {
            'algorithm': 'random_forest',
            'test_size': 0.2,
            'random_state': 42,
            'use_smote': True,
            'cv_folds': 5,
            'n_estimators': 100,
            'max_depth': 10
        }

        if train_params is None:
            train_params = {}

        params = {**default_params, **train_params}

        if target_column not in data.columns:
            raise ValueError(
                f"La columna objetivo '{target_column}' no existe en los datos.")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        self.feature_names = X.columns.tolist()

        print(
            f"📊 Iniciando entrenamiento con {len(X)} muestras y {len(X.columns)} características")
        print(f"📈 Distribución de clases: {dict(y.value_counts())}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=params['test_size'],
            random_state=params['random_state'],
            stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if params['use_smote']:
            print("🔄 Aplicando SMOTE para balanceo de clases...")
            smote = SMOTE(random_state=params['random_state'])
            X_train_scaled, y_train = smote.fit_resample(
                X_train_scaled, y_train)
            print(
                f"📈 Nueva distribución después de SMOTE: {dict(pd.Series(y_train).value_counts())}")

        if params['algorithm'] == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=params['random_state'],
                class_weight='balanced'
            )
        elif params['algorithm'] == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=params['random_state'],
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(
                "Algoritmo no soportado. Use 'random_forest' o 'logistic_regression'")

        print(f"🚀 Entrenando modelo {params['algorithm']}...")
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        conf_matrix = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        print("🔍 Realizando validación cruzada...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=StratifiedKFold(
                n_splits=params['cv_folds'], shuffle=True, random_state=params['random_state']),
            scoring='f1'
        )

        self.model_metadata = {
            'algorithm': params['algorithm'],
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(self.feature_names),
            'use_smote': params['use_smote'],
            'parameters': params
        }

        results = {
            'confusion_matrix': conf_matrix,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'cv_mean_f1': cv_scores.mean(),
            'cv_std_f1': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': None
        }

        if params['algorithm'] == 'random_forest':
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = feature_importance

        print("\n" + "="*50)
        print("📊 RESULTADOS DEL ENTRENAMIENTO")
        print("="*50)
        print(f"🎯 Precisión: {precision:.4f}")
        print(f"🔍 Recall: {recall:.4f}")
        print(f"⚖️  F1-Score: {f1:.4f}")
        print(f"📈 AUC-ROC: {auc_score:.4f}")
        print(f"🔄 CV F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("\n📋 Matriz de Confusión:")
        print(conf_matrix)

        if results['feature_importance'] is not None:
            print("\n🎯 Top 5 Características más Importantes:")
            print(results['feature_importance'].head())

        return results

    def save_model(self, model_name, save_path='./models/'):
        """
        Guarda el modelo entrenado y sus componentes.

        Args:
            model_name (str): Nombre para el modelo
            save_path (str): Directorio donde guardar (default: './models/')

        Returns:
            str: Ruta completa donde se guardó el modelo
        """

        if self.model is None:
            raise ValueError(
                "No hay modelo entrenado para guardar. Ejecute train_model() primero.")

        os.makedirs(save_path, exist_ok=True)

        model_path = os.path.join(save_path, f"{model_name}_model.joblib")
        scaler_path = os.path.join(save_path, f"{model_name}_scaler.joblib")
        metadata_path = os.path.join(
            save_path, f"{model_name}_metadata.joblib")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        full_metadata = {
            **self.model_metadata,
            'feature_names': self.feature_names,
            'model_path': model_path,
            'scaler_path': scaler_path
        }
        joblib.dump(full_metadata, metadata_path)

        print(f"✅ Modelo guardado exitosamente en: {save_path}")
        print(f"📁 Archivos creados:")
        print(f"   - Modelo: {model_path}")
        print(f"   - Scaler: {scaler_path}")
        print(f"   - Metadata: {metadata_path}")

        return save_path

    def load_model(self, model_name, load_path='./models/'):
        """
        Carga un modelo previamente entrenado.

        Args:
            model_name (str): Nombre del modelo a cargar
            load_path (str): Directorio donde buscar el modelo (default: './models/')

        Returns:
            str: Ruta del modelo cargado
        """

        model_path = os.path.join(load_path, f"{model_name}_model.joblib")
        scaler_path = os.path.join(load_path, f"{model_name}_scaler.joblib")
        metadata_path = os.path.join(
            load_path, f"{model_name}_metadata.joblib")

        for path, name in [(model_path, "modelo"), (scaler_path, "scaler"), (metadata_path, "metadata")]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"No se encontró el archivo de {name}: {path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.model_metadata = joblib.load(metadata_path)
        self.feature_names = self.model_metadata.get('feature_names', [])

        print(f"✅ Modelo '{model_name}' cargado exitosamente")
        print(
            f"📅 Entrenado el: {self.model_metadata.get('training_date', 'Desconocido')}")
        print(
            f"🎯 Algoritmo: {self.model_metadata.get('algorithm', 'Desconocido')}")
        print(f"📊 Características: {len(self.feature_names)}")

        return load_path

    def test_model(self, data):
        """
        Realiza predicciones sobre nuevos datos usando el modelo entrenado.

        Args:
            data (pd.DataFrame): Datos para predicción (deben tener las mismas características)

        Returns:
            dict: Predicciones, probabilidades y estadísticas
        """

        if self.model is None:
            raise ValueError(
                "No hay modelo cargado. Use load_model() o train_model() primero.")

        missing_features = set(self.feature_names) - set(data.columns)
        if missing_features:
            raise ValueError(
                f"Faltan características en los datos: {missing_features}")

        X = data[self.feature_names]

        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        fraud_count = np.sum(predictions)
        total_count = len(predictions)
        fraud_percentage = (fraud_count / total_count) * 100

        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'fraud_probability': probabilities[:, 1],
            'fraud_count': fraud_count,
            'total_transactions': total_count,
            'fraud_percentage': fraud_percentage,
            'summary': {
                'total_transactions': total_count,
                'predicted_frauds': fraud_count,
                'predicted_legitimate': total_count - fraud_count,
                'fraud_rate': f"{fraud_percentage:.2f}%"
            }
        }

        print(f"🔍 Predicciones completadas para {total_count} transacciones")
        print(f"🚨 Fraudes detectados: {fraud_count} ({fraud_percentage:.2f}%)")
        print(
            f"✅ Transacciones legítimas: {total_count - fraud_count} ({100 - fraud_percentage:.2f}%)")

        return results


def train_model(data, train_params=None):
    """
    Función de conveniencia para entrenar un modelo.
    Mantiene la interfaz especificada en los requerimientos.
    """
    lib = FraudDetectionLibrary()
    return lib.train_model(data, train_params=train_params)


def save_model(model_name, lib_instance=None):
    """
    Función de conveniencia para guardar un modelo.
    """
    if lib_instance is None:
        raise ValueError(
            "Debe proporcionar una instancia de FraudDetectionLibrary entrenada")
    return lib_instance.save_model(model_name)


def load_model(model_name):
    """
    Función de conveniencia para cargar un modelo.
    """
    lib = FraudDetectionLibrary()
    lib.load_model(model_name)
    return lib


def test_model(data, lib_instance=None):
    """
    Función de conveniencia para hacer predicciones.
    """
    if lib_instance is None:
        raise ValueError(
            "Debe proporcionar una instancia de FraudDetectionLibrary con modelo cargado")
    return lib_instance.test_model(data)
