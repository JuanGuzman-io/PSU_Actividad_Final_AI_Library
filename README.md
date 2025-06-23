# Librería de Detección de Fraude - PSU Actividad Final

Sistema de detección de fraude en transacciones financieras desarrollado como parte del proyecto final de Fundamentos de IA para Ingenieros de Software.

## 👥 Autores
- **Delgado Guzmán, Juan David**
- **Fonseca Bello, Diego Fernando**
- **Jácome Jami, Daniela Estefanía**
- **Parra Súa, Yohn Eduin**
- **Poveda Melo, Ingrid Carolina**

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

#### `data_clean(data_path, outlier_limit=2700, columns_to_remove=None)`
Limpia y preprocesa datos crudos de transacciones.

**Parámetros:**
- `data_path`: Ruta al archivo CSV crudo
- `outlier_limit`: Límite para filtrar outliers por monto
- `columns_to_remove`: Columnas a eliminar

**Retorna:** DataFrame limpio listo para entrenamiento

**Funcionalidades:**
- Filtrado de outliers por monto
- Procesamiento de fechas y características temporales
- Cálculo de edad del titular
- Cálculo de distancia geográfica entre cliente y comercio
- Codificación de variables categóricas
- Manejo de valores nulos

### 🌊 Pipeline Completo con data_clean()

```python
from AIlibrary import data_clean, FraudDetectionLibrary

# 1. Limpiar datos crudos
cleaned_df = data_clean('credit_card_transactions.csv')

# 2. Entrenar modelo
detector = FraudDetectionLibrary()
results = detector.train_model(cleaned_df)

# 3. Guardar modelo
detector.save_model('mi_modelo')

# 4. Usar modelo
predicciones = detector.test_model(nuevos_datos)
```

## 📊 Ejecutar Demo

### 🚀 **AppCompletePipeline.py** - Pipeline Completo (RECOMENDADO)
**Uso:** Demostración del pipeline completo desde datos crudos hasta predicciones

```bash
python AppCompletePipeline.py
```

**Características:**
- ✅ **Pipeline completo**: datos crudos → limpieza → entrenamiento → predicciones
- ✅ **Incluye `data_clean()`**: procesa datos sin preprocesar
- ✅ **Realistic workflow**: simula un caso de uso real
- ✅ **Rápida ejecución**: sin visualizaciones pesadas
- ✅ **Ideal para entrega**: muestra todas las funcionalidades requeridas

### 🎨 **App.py** - Demo con Visualizaciones
**Uso:** Demostración completa con gráficos y análisis visual

```bash
python App.py
```

**Características:**
- ✅ **Visualizaciones**: gráficos de métricas y análisis
- ✅ **Análisis detallado**: casos de alto riesgo con estadísticas
- ✅ **Demo interactiva**: simulación de predicciones en tiempo real
- ✅ **Datos sintéticos**: genera datos si no encuentra el archivo original
- ✅ **Ideal para presentaciones**: más visual y completo

### ⚡ **Alternativa: data_cleaning.py**
**Uso:** Si prefieres usar la función de limpieza por separado

```bash
python data_cleaning.py
```

**Características:**
- ✅ **Función standalone**: limpieza independiente
- ✅ **Compatible**: con el trabajo de tu compañero
- ✅ **Modular**: usar solo la parte que necesites

### 📋 **¿Cuál ejecutar?**

| Propósito | Archivo Recomendado |
|-----------|--------------------|
| **Entrega del proyecto** | `AppCompletePipeline.py` ✅ |
| **Presentación visual** | `App.py` 🎨 |
| **Solo limpieza de datos** | `data_cleaning.py` 🧽 |

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
├── 🎨 App.py                    # Demo con visualizaciones y gráficos
├── 🚀 AppCompletePipeline.py     # Pipeline completo (RECOMENDADO)
├── 🧠 AIlibrary.py              # Librería principal de detección de fraude
├── 🧽 data_cleaning.py          # Función de limpieza separada
├── 📄 README.md                 # Documentación (este archivo)
├── 📎 requirements.txt          # Dependencias del proyecto
├── 📋 credit_card_transactions.csv # Dataset crudo
└── 📁 models/                   # Modelos entrenados guardados
    ├── fraud_detector_v1_model.joblib
    ├── fraud_detector_v1_scaler.joblib
    └── fraud_detector_v1_metadata.joblib
```

### 🔑 **Archivos Principales:**

- **`AppCompletePipeline.py`** ✅ **Para entrega**: Pipeline completo con `data_clean()`
- **`App.py`** 🎨 **Para presentar**: Demo visual con gráficos 
- **`AIlibrary.py`** 🧠 **Core**: Librería principal con todas las funciones
- **`data_cleaning.py`** 🧽 **Utilidad**: Función de limpieza separada

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

---
**Universidad Internacional de La Rioja (UNIR)**  
Programa Superior Universitario en Inteligencia Artificial para Desarrollo de Software y DevOps
