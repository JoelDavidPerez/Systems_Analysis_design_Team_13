import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_data(df, sequence_length=10, scaler_info=None):
    """
    Prepara datos para el modelo LSTM
    
    Args:
        df: DataFrame con los datos
        sequence_length: Longitud de la secuencia
        scaler_info: Información de escalado previo (opcional)
    
    Returns:
        X: Arrays de entrada (num_samples, sequence_length, features)
        y: Array de salida (num_samples,)
        scaler_info: Información de escalado
    """
    
    try:
        # Seleccionar columnas numéricas relevantes
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return None, None, None
        
        # Usar máximo 5 features
        numeric_cols = numeric_cols[:min(5, len(numeric_cols))]
        data = df[numeric_cols].values
        
        # Normalizar datos
        if scaler_info is None:
            scaler = MinMaxScaler()
            data_normalized = scaler.fit_transform(data)
            scaler_info = {
                'min': scaler.data_min_.tolist(),
                'max': scaler.data_max_.tolist(),
                'scale': scaler.scale_.tolist(),
                'columns': numeric_cols
            }
        else:
            # Usar scaler info previo
            min_vals = np.array(scaler_info['min'])
            max_vals = np.array(scaler_info['max'])
            data_normalized = (data - min_vals) / (max_vals - min_vals)
        
        # Crear secuencias
        X, y = [], []
        
        for i in range(len(data_normalized) - sequence_length):
            X.append(data_normalized[i:i+sequence_length])
            # Predecir el valor siguiente del primer feature
            y.append(data_normalized[i+sequence_length, 0])
        
        return np.array(X), np.array(y), scaler_info
    
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None, None

def normalize_data(data, scaler_info):
    """Normaliza datos usando información de scaler"""
    min_vals = np.array(scaler_info['min'])
    max_vals = np.array(scaler_info['max'])
    return (data - min_vals) / (max_vals - min_vals)

def denormalize_data(data, scaler_info):
    """Desnormaliza datos usando información de scaler"""
    min_vals = np.array(scaler_info['min'])[0]
    max_vals = np.array(scaler_info['max'])[0]
    return data * (max_vals - min_vals) + min_vals

def validate_csv(filepath):
    """Valida que el CSV sea válido"""
    try:
        df = pd.read_csv(filepath)
        return len(df) > 0 and len(df.columns) > 0
    except:
        return False