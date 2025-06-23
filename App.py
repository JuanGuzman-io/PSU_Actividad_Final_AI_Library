# App.py
# Aplicación de detección de fraude en tiempo real

import pandas as pd
import numpy as np
from AIlibrary import FraudDetectionLibrary
import sys

class FraudDetectionApp:
    """
    Aplicación de detección de fraude que simula un sistema en producción.
    """
    
    def __init__(self, model_name="fraud_detector_v1"):
        """
        Inicializa la aplicación cargando el modelo entrenado.
        """
        print("🚀 INICIANDO APLICACIÓN DE DETECCIÓN DE FRAUDE")
        print("="*50)
        
        try:
            self.detector = FraudDetectionLibrary()
            self.detector.load_model(model_name)
            print("✅ Modelo cargado exitosamente")
            print("🛡️ Sistema listo para detectar fraudes\n")
        except FileNotFoundError:
            print("❌ Modelo no encontrado.")
            print("💡 Ejecuta primero 'python AppCompletePipeline.py' para entrenar el modelo")
            sys.exit(1)
    
    def check_transaction(self, transaction_data):
        """
        Analiza una transacción y determina si es fraude.
        
        Args:
            transaction_data (dict): Datos de la transacción
            
        Returns:
            dict: Resultado del análisis
        """
        
        # Convertir a DataFrame
        transaction_df = pd.DataFrame([transaction_data])
        
        # Agregar características faltantes con valores por defecto
        for feature_name in self.detector.feature_names:
            if feature_name not in transaction_df.columns:
                if feature_name.startswith('feature_'):
                    # Características sintéticas con valor 0
                    transaction_df[feature_name] = 0.0
                else:
                    # Otras características con valores por defecto
                    default_values = {
                        'cc_num': 100,
                        'merchant': 50,
                        'trans_year': 2024,
                        'trans_month': 6,
                        'trans_day': 15,
                        'trans_season': 2,
                        'trans_minute': 30,
                        'trans_second': 15
                    }
                    transaction_df[feature_name] = default_values.get(feature_name, 0)
        
        # Hacer predicción
        result = self.detector.test_model(transaction_df)
        
        fraud_probability = result['fraud_probability'][0]
        is_fraud = result['predictions'][0]
        
        # Determinar nivel de riesgo
        if fraud_probability > 0.8:
            risk_level = "🔴 ALTO RIESGO"
            recommendation = "BLOQUEAR TRANSACCIÓN"
        elif fraud_probability > 0.5:
            risk_level = "🟡 RIESGO MEDIO"
            recommendation = "REVISAR MANUALMENTE"
        elif fraud_probability > 0.2:
            risk_level = "🟠 RIESGO BAJO"
            recommendation = "MONITOREAR"
        else:
            risk_level = "🟢 RIESGO MÍNIMO"
            recommendation = "APROBAR"
        
        return {
            'is_fraud': bool(is_fraud),
            'fraud_probability': fraud_probability,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'transaction_id': transaction_data.get('id', 'N/A')
        }
    
    def print_analysis(self, transaction_data, result):
        """
        Imprime el análisis de la transacción de forma clara.
        """
        print("\n" + "="*60)
        print(f"📊 ANÁLISIS DE TRANSACCIÓN #{result['transaction_id']}")
        print("="*60)
        
        # Datos de la transacción
        print(f"💰 Monto: ${transaction_data['amt']:.2f}")
        print(f"🕒 Hora: {transaction_data.get('trans_hour', 'N/A')}:00")
        print(f"👤 Edad del titular: {transaction_data.get('card_holder_age', 'N/A')} años")
        print(f"📍 Distancia: {transaction_data.get('distance', 'N/A')} km")
        print(f"🏪 Categoría: {transaction_data.get('category', 'N/A')}")
        
        # Resultado del análisis
        print(f"\n🎯 RESULTADO DEL ANÁLISIS:")
        print(f"   📈 Probabilidad de fraude: {result['fraud_probability']:.1%}")
        print(f"   🚨 Clasificación: {'FRAUDE' if result['is_fraud'] else 'LEGÍTIMA'}")
        print(f"   ⚠️  Nivel de riesgo: {result['risk_level']}")
        print(f"   💡 Recomendación: {result['recommendation']}")
        
        print("="*60)

def create_sample_transactions():
    """
    Crea transacciones de ejemplo para demostrar la aplicación.
    """
    transactions = [
        {
            'id': 'TXN001',
            'amt': 45.99,
            'trans_hour': 14,
            'card_holder_age': 35,
            'distance': 5.2,
            'category': 1,
            'gender': 1,
            'lat': 40.7128,
            'long': -74.0060,
            'city_pop': 8000000,
            'job': 15,
            'merch_lat': 40.7589,
            'merch_long': -73.9851,
            'cc_num': 123,
            'merchant': 45,
            'trans_year': 2024,
            'trans_month': 6,
            'trans_day': 15,
            'trans_season': 2,
            'trans_weekday': 3,
            'trans_minute': 30,
            'trans_second': 15,
            # Características adicionales del modelo
            'feature_0': 0.1,
            'feature_1': -0.2,
            'feature_2': 0.3,
            'feature_3': 0.0,
            'feature_4': 0.5,
            'feature_5': -0.1,
            'feature_6': 0.2,
            'feature_7': 0.4,
            'feature_8': -0.3,
            'feature_9': 0.1
        },
        {
            'id': 'TXN002',
            'amt': 2500.00,  # Monto alto - sospechoso
            'trans_hour': 3,  # Hora inusual - sospechoso
            'card_holder_age': 22,
            'distance': 150.5,  # Distancia larga - sospechoso
            'category': 5,
            'gender': 0,
            'lat': 34.0522,
            'long': -118.2437,
            'city_pop': 4000000,
            'job': 8,
            'merch_lat': 36.1627,
            'merch_long': -115.1969,
            'cc_num': 456,
            'merchant': 78,
            'trans_year': 2024,
            'trans_month': 6,
            'trans_day': 15,
            'trans_season': 2,
            'trans_weekday': 5,
            'trans_minute': 45,
            'trans_second': 33,
            # Características que pueden indicar fraude
            'feature_0': 1.2,
            'feature_1': 2.1,
            'feature_2': -1.5,
            'feature_3': 0.8,
            'feature_4': 1.7,
            'feature_5': -0.9,
            'feature_6': 1.3,
            'feature_7': 2.0,
            'feature_8': -1.1,
            'feature_9': 1.4
        },
        {
            'id': 'TXN003',
            'amt': 89.50,
            'trans_hour': 10,
            'card_holder_age': 45,
            'distance': 2.1,
            'category': 2,
            'gender': 1,
            'lat': 41.8781,
            'long': -87.6298,
            'city_pop': 2700000,
            'job': 12,
            'merch_lat': 41.8825,
            'merch_long': -87.6230,
            'cc_num': 789,
            'merchant': 23,
            'trans_year': 2024,
            'trans_month': 6,
            'trans_day': 15,
            'trans_season': 2,
            'trans_weekday': 3,
            'trans_minute': 20,
            'trans_second': 8,
            # Características normales
            'feature_0': 0.0,
            'feature_1': 0.1,
            'feature_2': -0.1,
            'feature_3': 0.2,
            'feature_4': 0.0,
            'feature_5': 0.1,
            'feature_6': -0.2,
            'feature_7': 0.0,
            'feature_8': 0.1,
            'feature_9': 0.0
        }
    ]
    
    return transactions

def interactive_mode():
    """
    Modo interactivo para ingresar transacciones manualmente.
    """
    print("\n🔧 MODO INTERACTIVO")
    print("Ingresa los datos de la transacción:")
    
    try:
        transaction = {
            'id': input("ID de transacción: ") or "MANUAL",
            'amt': float(input("Monto ($): ")),
            'trans_hour': int(input("Hora (0-23): ")),
            'card_holder_age': int(input("Edad del titular: ")),
            'distance': float(input("Distancia al comercio (km): ")),
            'category': int(input("Categoría del comercio (0-15): ")),
            'gender': int(input("Género (0=M, 1=F): ")),
            'lat': float(input("Latitud del cliente: ") or "40.7128"),
            'long': float(input("Longitud del cliente: ") or "-74.0060"),
            'city_pop': int(input("Población de la ciudad: ") or "1000000"),
            'job': int(input("Código de trabajo (0-20): ") or "10"),
            'merch_lat': float(input("Latitud del comercio: ") or "40.7589"),
            'merch_long': float(input("Longitud del comercio: ") or "-73.9851"),
        }
        
        # Agregar campos adicionales con valores por defecto
        transaction.update({
            'cc_num': 100,
            'merchant': 50,
            'trans_year': 2024,
            'trans_month': 6,
            'trans_day': 15,
            'trans_season': 2,
            'trans_weekday': 3,
            'trans_minute': 30,
            'trans_second': 15
        })
        
        return transaction
        
    except ValueError:
        print("❌ Error: Ingresa valores numéricos válidos")
        return None

def main():
    """
    Función principal de la aplicación.
    """
    
    # Inicializar aplicación
    app = FraudDetectionApp()
    
    # Menú principal
    while True:
        print("\n🎯 SISTEMA DE DETECCIÓN DE FRAUDE")
        print("="*40)
        print("1. 📊 Analizar transacciones de ejemplo")
        print("2. ✍️  Ingresar transacción manualmente")
        print("3. 🚪 Salir")
        
        choice = input("\nSelecciona una opción (1-3): ").strip()
        
        if choice == '1':
            print("\n📋 ANALIZANDO TRANSACCIONES DE EJEMPLO...")
            
            sample_transactions = create_sample_transactions()
            
            for transaction in sample_transactions:
                result = app.check_transaction(transaction)
                app.print_analysis(transaction, result)
                
                input("\n⏸️  Presiona Enter para continuar...")
        
        elif choice == '2':
            transaction = interactive_mode()
            if transaction:
                result = app.check_transaction(transaction)
                app.print_analysis(transaction, result)
        
        elif choice == '3':
            print("\n👋 ¡Gracias por usar el sistema de detección de fraude!")
            break
        
        else:
            print("❌ Opción inválida. Por favor selecciona 1, 2 o 3.")

if __name__ == "__main__":
    main()
