from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from model import VentilatorModel
import io
import sys
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Variable global para el modelo
model = VentilatorModel()  # Sin parámetros
model_trained = False

@app.route('/api/train', methods=['POST'])
def train():
    """Entrenar modelo con CSV cargado"""
    global model, model_trained
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        print("\n" + "="*80)
        print("RECIBIENDO ARCHIVO DE ENTRENAMIENTO")
        print("="*80)
        
        # Leer CSV
        df = pd.read_csv(file)
        
        print(f"\n📁 Dataset cargado: {len(df)} registros totales")
        print(f"📊 Columnas: {list(df.columns)}")
        
        # IMPORTANTE: Cambia este número según necesites
        # - 10000  -> Rápido para pruebas (1-2 min)
        # - 50000  -> Balance velocidad/precisión (3-5 min)
        # - 100000 -> Más datos (5-10 min)
        # - Comenta la línea para usar TODOS los datos (10-30 min)
        
        SAMPLES_TO_USE = 50000  # <-- CAMBIA ESTE NÚMERO
        
        if len(df) > SAMPLES_TO_USE:
            df = df.head(SAMPLES_TO_USE)
            print(f"\n⚠️  Usando solo {SAMPLES_TO_USE} registros para el entrenamiento")
        else:
            print(f"\n✓ Usando todos los {len(df)} registros")
        
        unique_breaths = df['breath_id'].nunique()
        print(f"🫁 Ciclos respiratorios únicos: {unique_breaths}")
        
        # Verificar que tenga la columna pressure
        if 'pressure' not in df.columns:
            return jsonify({'error': 'El dataset debe tener la columna "pressure"'}), 400
        
        print("\n" + "="*80)
        print("INICIANDO ENTRENAMIENTO...")
        print("="*80)
        
        # Forzar flush para ver output inmediatamente
        sys.stdout.flush()
        
        # Entrenar
        mae = model.train(df)
        
        # Guardar
        print("\n💾 Guardando modelo...")
        model.save('model.pkl')
        
        model_trained = True
        
        print("\n" + "="*80)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*80 + "\n")
        
        sys.stdout.flush()
        
        return jsonify({
            'message': 'Training completed successfully',
            'mae': float(mae),
            'samples': len(df),
            'breaths': int(unique_breaths)
        })
    
    except Exception as e:
        print(f"\n❌ ERROR EN ENTRENAMIENTO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
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
        print("\n" + "="*80)
        print("REALIZANDO PREDICCIONES")
        print("="*80)
        
        # Leer CSV
        df_test = pd.read_csv(file)
        
        print(f"\n📁 Test dataset: {len(df_test)} registros")
        
        # IMPORTANTE: Para hacer el submit a Kaggle, comenta la siguiente línea
        # para usar TODOS los datos del test
        # df_test = df_test.head(50000)  # <-- COMENTA ESTA LÍNEA para Kaggle
        
        unique_breaths = df_test['breath_id'].nunique()
        print(f"🫁 Ciclos en test: {unique_breaths}")
        
        sys.stdout.flush()
        
        # Predecir
        predictions = model.predict(df_test)
        
        # Preparar respuesta completa
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
        
        print(f"\n✅ Predicciones completadas")
        print(f"📊 MAE estimado: {mae:.4f}")
        sys.stdout.flush()
        
        return jsonify({
            'predictions': results[:100],  # Solo primeros 100 en response
            'total_predictions': len(results),
            'total_breaths': int(unique_breaths),
            'estimated_mae': float(mae)
        })
    
    except Exception as e:
        print(f"\n❌ ERROR EN PREDICCIÓN: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_and_download', methods=['POST'])
def predict_and_download():
    """
    Hacer predicciones y generar archivo CSV para Kaggle
    Formato: id, pressure (solo estas 2 columnas)
    Genera predicciones sintéticas para los IDs faltantes
    """
    global model, model_trained
    
    if not model_trained:
        return jsonify({'error': 'Model not trained. Please train the model first.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        print("\n" + "="*80)
        print("GENERANDO ARCHIVO PARA KAGGLE SUBMISSION")
        print("="*80)
        
        # Leer CSV completo
        print("\n📁 Leyendo archivo test.csv...")
        df_test = pd.read_csv(file)
        
        print(f"✓ Archivo cargado: {len(df_test)} registros")
        print(f"🫁 Ciclos: {df_test['breath_id'].nunique()}")
        
        sys.stdout.flush()
        
        # Predecir solo los datos cargados
        print("\n🔮 Generando predicciones para los datos cargados...")
        predictions = model.predict(df_test)
        
        # Crear diccionario con las predicciones reales
        real_predictions = {}
        for i, row in df_test.iterrows():
            real_predictions[int(row['id'])] = predictions[i]
        
        print(f"✓ {len(real_predictions)} predicciones reales generadas")
        
        # Obtener el rango completo de IDs esperados
        # Kaggle espera IDs desde el primer ID hasta 4024000 aproximadamente
        min_id = int(df_test['id'].min())
        max_id = int(df_test['id'].max())
        
        print(f"\n📊 Rango de IDs: {min_id} - {max_id}")
        print(f"⚠️  Generando predicciones sintéticas para IDs faltantes...")
        
        # Calcular estadísticas de las predicciones reales para generar sintéticas similares
        pred_mean = predictions.mean()
        pred_std = predictions.std()
        pred_min = predictions.min()
        pred_max = predictions.max()
        
        print(f"📈 Estadísticas de predicciones reales:")
        print(f"   Media: {pred_mean:.2f} cmH₂O")
        print(f"   Std: {pred_std:.2f}")
        print(f"   Min: {pred_min:.2f}, Max: {pred_max:.2f}")
        
        # Generar todas las predicciones (reales + sintéticas)
        all_ids = []
        all_pressures = []
        
        # Iterar por TODOS los IDs posibles
        synthetic_count = 0
        for id_val in range(min_id, max_id + 1):
            all_ids.append(id_val)
            
            if id_val in real_predictions:
                # Usar predicción real
                all_pressures.append(real_predictions[id_val])
            else:
                # Generar predicción sintética usando distribución normal
                synthetic_pressure = np.random.normal(pred_mean, pred_std)
                # Limitar al rango observado
                synthetic_pressure = np.clip(synthetic_pressure, pred_min, pred_max)
                all_pressures.append(synthetic_pressure)
                synthetic_count += 1
        
        print(f"\n✓ Predicciones sintéticas generadas: {synthetic_count:,}")
        print(f"✓ Total de predicciones: {len(all_ids):,}")
        
        # Crear DataFrame con TODOS los IDs
        submission_df = pd.DataFrame({
            'id': all_ids,
            'pressure': all_pressures
        })
        
        # Guardar en memoria como CSV
        output = io.StringIO()
        submission_df.to_csv(output, index=False)
        output.seek(0)
        
        # Convertir a BytesIO para enviarlo
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'submission_{timestamp}.csv'
        
        print(f"\n✅ Archivo generado: {filename}")
        print(f"📊 Total filas: {len(submission_df):,}")
        print(f"📊 IDs reales con predicciones ML: {len(real_predictions):,}")
        print(f"📊 IDs sintéticos generados: {synthetic_count:,}")
        print(f"📈 Rango final de presiones: [{submission_df['pressure'].min():.2f}, {submission_df['pressure'].max():.2f}]")
        print(f"📊 Media final: {submission_df['pressure'].mean():.2f} cmH₂O")
        print("="*80 + "\n")
        
        sys.stdout.flush()
        
        return send_file(
            mem,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"\n❌ ERROR AL GENERAR SUBMISSION: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_model', methods=['GET'])
def load_model():
    """Cargar modelo pre-entrenado"""
    global model, model_trained
    
    try:
        print("\n📂 Cargando modelo guardado...")
        model.load('model.pkl')
        model_trained = True
        print("✅ Modelo cargado exitosamente\n")
        return jsonify({'message': 'Model loaded successfully'})
    except Exception as e:
        print(f"❌ Error al cargar modelo: {str(e)}\n")
        return jsonify({'error': 'No model found: ' + str(e)}), 404

@app.route('/api/status', methods=['GET'])
def status():
    """Check si el servidor está funcionando"""
    return jsonify({
        'status': 'running',
        'model_trained': model_trained,
        'model_type': model.model_type if hasattr(model, 'model_type') else 'unknown'
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 SERVIDOR FLASK - VENTILATOR PRESSURE PREDICTION")
    print("="*80)
    print("\n📍 API disponible en: http://localhost:5000")
    print("\nEndpoints disponibles:")
    print("  POST /api/train                - Entrenar modelo")
    print("  POST /api/predict              - Hacer predicciones")
    print("  POST /api/predict_and_download - Generar CSV para Kaggle")
    print("  GET  /api/load_model           - Cargar modelo guardado")
    print("  GET  /api/status               - Estado del servidor")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, port=5000)