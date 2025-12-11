import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import time

class VentilatorModel:
    def __init__(self, model_type='fast'):
        """
        model_type: 'fast' (RandomForest) o 'accurate' (GradientBoosting)
        """
        if model_type == 'fast':
            # Modelo más rápido para demo
            self.model = RandomForestRegressor(
                n_estimators=100,  # Más rápido que 200
                max_depth=15,
                random_state=42,
                n_jobs=-1,  # Usar todos los CPUs
                verbose=1    # Mostrar progreso
            )
        else:
            # Modelo más preciso pero más lento
            self.model = GradientBoostingRegressor(
                n_estimators=100,  # Reducido de 200 a 100
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=1  # Mostrar progreso
            )
        
        self.scaler = StandardScaler()
        self.model_type = model_type
        
    def prepare_features(self, df):
        """
        Crear features temporales y de ventana
        """
        print(f"Preparando features de {len(df)} registros...")
        start_time = time.time()
        
        features = []
        targets = []
        
        breath_ids = df['breath_id'].unique()
        total_breaths = len(breath_ids)
        
        for idx, breath_id in enumerate(breath_ids):
            # Mostrar progreso cada 10 ciclos
            if idx % 10 == 0:
                print(f"Procesando ciclo {idx+1}/{total_breaths} ({(idx/total_breaths)*100:.1f}%)")
            
            breath_data = df[df['breath_id'] == breath_id].sort_values('time_step')
            
            for i in range(len(breath_data)):
                row = breath_data.iloc[i]
                
                # Features básicas
                feature_vector = [
                    row['R'],
                    row['C'],
                    row['time_step'],
                    row['u_in'],
                    row['u_out']
                ]
                
                # Features temporales (ventana de 3 pasos anteriores)
                if i > 0:
                    prev_row = breath_data.iloc[i-1]
                    feature_vector.extend([
                        prev_row['u_in'],
                        prev_row['u_out']
                    ])
                else:
                    feature_vector.extend([0, 0])
                
                if i > 1:
                    prev2_row = breath_data.iloc[i-2]
                    feature_vector.extend([
                        prev2_row['u_in'],
                        prev2_row['u_out']
                    ])
                else:
                    feature_vector.extend([0, 0])
                
                features.append(feature_vector)
                
                if 'pressure' in row:
                    targets.append(row['pressure'])
        
        elapsed = time.time() - start_time
        print(f"✓ Features preparadas en {elapsed:.2f} segundos")
        
        return np.array(features), np.array(targets) if targets else None
    
    def train(self, df, validation_split=0.2):
        """Entrenar el modelo con validación"""
        print(f"\n{'='*60}")
        print(f"INICIANDO ENTRENAMIENTO - Modelo: {self.model_type.upper()}")
        print(f"{'='*60}\n")
        
        # Preparar datos
        X, y = self.prepare_features(df)
        
        print(f"\nDatos de entrenamiento:")
        print(f"  - Total de muestras: {len(X):,}")
        print(f"  - Features: {X.shape[1]}")
        print(f"  - Ciclos respiratorios: {df['breath_id'].nunique()}")
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        print(f"\nDivisión de datos:")
        print(f"  - Entrenamiento: {len(X_train):,} muestras")
        print(f"  - Validación: {len(X_val):,} muestras")
        
        # Normalizar features
        print("\nNormalizando features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Entrenar
        print(f"\n{'='*60}")
        print("ENTRENANDO MODELO...")
        print(f"{'='*60}\n")
        start_time = time.time()
        
        self.model.fit(X_train_scaled, y_train)
        
        training_time = time.time() - start_time
        
        # Calcular métricas
        print("\nCalculando métricas...")
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        train_mae = np.mean(np.abs(train_pred - y_train))
        val_mae = np.mean(np.abs(val_pred - y_val))
        
        train_rmse = np.sqrt(np.mean((train_pred - y_train)**2))
        val_rmse = np.sqrt(np.mean((val_pred - y_val)**2))
        
        # Resultados
        print(f"\n{'='*60}")
        print("RESULTADOS DEL ENTRENAMIENTO")
        print(f"{'='*60}\n")
        print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
        print(f"\nMétricas en ENTRENAMIENTO:")
        print(f"  - MAE:  {train_mae:.4f} cmH₂O")
        print(f"  - RMSE: {train_rmse:.4f} cmH₂O")
        print(f"\nMétricas en VALIDACIÓN:")
        print(f"  - MAE:  {val_mae:.4f} cmH₂O")
        print(f"  - RMSE: {val_rmse:.4f} cmH₂O")
        
        # Verificar overfitting
        if val_mae > train_mae * 1.5:
            print(f"\n⚠️  ADVERTENCIA: Posible overfitting detectado")
            print(f"   (MAE validación es {(val_mae/train_mae):.2f}x el MAE de entrenamiento)")
        else:
            print(f"\n✓ Modelo generaliza bien (validación/train ratio: {(val_mae/train_mae):.2f})")
        
        print(f"\n{'='*60}\n")
        
        return val_mae  # Retornar MAE de validación
    
    def predict(self, df):
        """Hacer predicciones"""
        print(f"\nRealizando predicciones en {len(df)} registros...")
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        print("Generando predicciones...")
        predictions = self.model.predict(X_scaled)
        
        print(f"✓ {len(predictions)} predicciones completadas")
        print(f"  Rango: [{predictions.min():.2f}, {predictions.max():.2f}] cmH₂O")
        print(f"  Media: {predictions.mean():.2f} cmH₂O")
        
        return predictions
    
    def save(self, filepath='model.pkl'):
        """Guardar modelo"""
        print(f"\nGuardando modelo en {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type
            }, f)
        print("✓ Modelo guardado exitosamente")
    
    def load(self, filepath='model.pkl'):
        """Cargar modelo"""
        print(f"\nCargando modelo desde {filepath}...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.model_type = data.get('model_type', 'unknown')
        print(f"✓ Modelo cargado (tipo: {self.model_type})")