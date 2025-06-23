# Librería de Detección de Fraude - PSU Actividad Final

Sistema de detección de fraude en transacciones financieras desarrollado como parte del proyecto final de Fundamentos de IA para Ingenieros de Software.

## 📋 Descripción

Esta librería proporciona una interfaz de alto nivel para entrenar, guardar, cargar y usar modelos de inteligencia artificial especializados en la detección de fraude en transacciones financieras.

## 🎯 Características Principales

- ✅ **Entrenamiento automático** con Random Forest y Regresión Logística
- ✅ **Balanceo de clases** automático con SMOTE
- ✅ **Validación cruzada** estratificada
- ✅ **Persistencia de modelos** con joblib
- ✅ **Métricas especializadas** para detección de fraude
- ✅ **Visualizaciones** automáticas de resultados
- ✅ **Predicciones en tiempo real**

## 🚀 Instalación

1. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

2. **Generar datos de prueba (opcional):**
```bash
python data_generator.py
```

## 💻 Uso Básico

### Ejemplo Completo

```python
from AIlibrary import FraudDetectionLibrary
import pandas as pd

# 1. Cargar datos
df = pd.read_csv('data.csv')

# 2. Crear instancia de la librería
detector = FraudDetectionLibrary()

# 3. Entrenar modelo
params = {
    'algorithm': 'random_forest',
    'n_estimators': 100,
    'use_smote': True
}
results = detector.train_model(df, train_params=params)

# 4. Guardar modelo
detector.save_model('mi_modelo')

# 5. Cargar modelo (en otra sesión)
nuevo_detector = FraudDetectionLibrary()
nuevo_detector.load_model('mi_modelo')

# 6. Hacer predicciones
predicciones = nuevo_detector.test_model(nuevos_datos)
```

### Funciones Principales

#### `train_model(data, train_params=None)`
Entrena un modelo de clasificación para detección de fraude.

**Parámetros:**
- `data`: DataFrame con datos preprocesados
- `train_params`: Diccionario con parámetros de entrenamiento

**Retorna:** Diccionario con matriz de confusión y métricas

#### `save_model(model_name, save_path='./models/')`
Guarda el modelo entrenado y sus componentes.

**Parámetros:**
- `model_name`: Nombre para el modelo
- `save_path`: Directorio donde guardar

**Retorna:** Ruta donde se guardó el modelo

#### `load_model(model_name, load_path='./models/')`
Carga un modelo previamente entrenado.

#### `test_model(data)`
Realiza predicciones sobre nuevos datos.

## 📊 Ejecutar Demo

```bash
python App.py
```

El demo ejecutará automáticamente:
1. Carga de datos
2. Entrenamiento del modelo
3. Evaluación con métricas
4. Guardado y carga del modelo
5. Predicciones en tiempo real
6. Visualizaciones

## 🔧 Parámetros de Entrenamiento

```python
train_params = {
    'algorithm': 'random_forest',        # 'random_forest' o 'logistic_regression'
    'n_estimators': 100,                 # Número de árboles (Random Forest)
    'max_depth': 10,                     # Profundidad máxima
    'test_size': 0.2,                    # Proporción para test
    'use_smote': True,                   # Balanceo con SMOTE
    'cv_folds': 5,                       # Folds para validación cruzada
    'random_state': 42                   # Semilla aleatoria
}
```

## 📈 Métricas Incluidas

- **Matriz de Confusión**
- **Precisión, Recall, F1-Score**
- **AUC-ROC**
- **Validación Cruzada**
- **Importancia de Características**

## 📁 Estructura del Proyecto

```
PSU_Actividad_Final/
├── AIlibrary.py          # Librería principal
├── App.py                # Código de ejemplo
├── data_generator.py     # Generador de datos sintéticos
├── requirements.txt      # Dependencias
├── README.md            # Documentación
├── data.csv             # Datos de entrenamiento
└── models/              # Modelos guardados
    ├── modelo_model.joblib
    ├── modelo_scaler.joblib
    └── modelo_metadata.joblib
```

## 🎯 Transferir Datos Reales

Para usar los datos procesados desde Colab:

1. En Colab, después de ejecutar `data_clean()`:
```python
# Guardar datos limpios
cleaned_df.to_csv('credit_card_clean.csv', index=False)

# Descargar archivo
from google.colab import files
files.download('credit_card_clean.csv')
```

2. Colocar el archivo como `data.csv` en este directorio

## 📊 Resultados Esperados

La librería está optimizada para datasets de fraude y debería lograr:
- **F1-Score**: > 0.85
- **Recall**: > 0.80 (importante para detectar fraudes)
- **Precisión**: > 0.85
- **AUC-ROC**: > 0.90

## 🔍 Solución de Problemas

### Error: "Columna 'is_fraud' no encontrada"
Asegúrate de que tu dataset tenga una columna llamada `is_fraud` con valores 0/1.

### Error: "Faltan características"
Verifica que los datos de predicción tengan las mismas columnas que los datos de entrenamiento.

### Error: "Modelo no encontrado"
Asegúrate de entrenar y guardar el modelo antes de intentar cargarlo.

## 👥 Autores


---
**Universidad Internacional de La Rioja (UNIR)**  
Programa Superior Universitario en Inteligencia Artificial para Desarrollo de Software y DevOps
