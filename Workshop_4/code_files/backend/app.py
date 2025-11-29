from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from model import VentilatorModel
import io

app = Flask(__name__)
CORS(app)

# Variable global para el modelo
model = VentilatorModel()
model_trained = False

@app.route('/api/train', methods=['POST'])
def train():
    """Entrenar modelo con CSV cargado"""
    global model, model_trained
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Leer CSV
        df = pd.read_csv(file)
        
        # Usar muestra pequeña para demo
        df = df.head(5000)
        
        print(f"Entrenando con {len(df)} registros...")
        
        # Entrenar
        mae = model.train(df)
        
        # Guardar
        model.save('model.pkl')
        
        model_trained = True
        
        return jsonify({
            'message': 'Training completed',
            'mae': float(mae),
            'samples': len(df)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Hacer predicciones en test data"""
    global model, model_trained
    
    if not model_trained:
        return jsonify({'error': 'Model not trained'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Leer CSV
        df_test = pd.read_csv(file)
        
        # Muestra pequeña
        df_test = df_test.head(5000)
        
        print(f"Prediciendo {len(df_test)} registros...")
        
        # Predecir
        predictions = model.predict(df_test)
        
        # Preparar respuesta
        results = []
        for i, (idx, row) in enumerate(df_test.iterrows()):
            results.append({
                'id': int(row['id']),
                'breath_id': int(row['breath_id']),
                'pressure': float(predictions[i])
            })
        
        # Calcular MAE simulado (comparar con fórmula física)
        simulated_pressures = df_test.apply(
            lambda row: row['R'] * row['u_in'] * 0.1 + (1/row['C']) * row['time_step'] * 0.5,
            axis=1
        )
        mae = abs(predictions - simulated_pressures).mean()
        
        return jsonify({
            'predictions': results[:100],  # Solo primeros 100 para no saturar
            'total_predictions': len(results),
            'estimated_mae': float(mae)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_model', methods=['GET'])
def load_model():
    """Cargar modelo pre-entrenado"""
    global model, model_trained
    
    try:
        model.load('model.pkl')
        model_trained = True
        return jsonify({'message': 'Model loaded successfully'})
    except Exception as e:
        return jsonify({'error': 'No model found: ' + str(e)}), 404

@app.route('/api/status', methods=['GET'])
def status():
    """Check si el servidor está funcionando"""
    return jsonify({
        'status': 'running',
        'model_trained': model_trained
    })

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📍 API disponible en: http://localhost:5000")
    app.run(debug=True, port=5000)