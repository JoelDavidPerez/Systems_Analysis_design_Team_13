import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pickle

class VentilatorModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """
        Crear features temporales y de ventana
        """
        features = []
        targets = []
        
        for breath_id in df['breath_id'].unique():
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
        
        return np.array(features), np.array(targets) if targets else None
    
    def train(self, df):
        """Entrenar el modelo"""
        X, y = self.prepare_features(df)
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar
        self.model.fit(X_scaled, y)
        
        # Calcular MAE en entrenamiento
        predictions = self.model.predict(X_scaled)
        mae = np.mean(np.abs(predictions - y))
        
        return mae
    
    def predict(self, df):
        """Hacer predicciones"""
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def save(self, filepath='model.pkl'):
        """Guardar modelo"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
    
    def load(self, filepath='model.pkl'):
        """Cargar modelo"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']