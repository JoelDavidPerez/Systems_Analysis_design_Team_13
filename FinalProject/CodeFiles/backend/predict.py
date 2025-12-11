import pandas as pd
from model import VentilatorModel

def predict_test(model_path, test_csv_path, output_csv='submission.csv'):
    print("Cargando modelo...")
    model = VentilatorModel()
    model.load(model_path)
    
    print("Cargando datos de test...")
    df_test = pd.read_csv(test_csv_path)
    
    # Usar solo una muestra para demo
    df_test = df_test.head(10000)
    
    print(f"Datos de test: {len(df_test)} registros")
    
    print("\nHaciendo predicciones...")
    predictions = model.predict(df_test)
    
    print("\nCreando archivo de submission...")
    submission = pd.DataFrame({
        'id': df_test['id'],
        'pressure': predictions
    })
    
    submission.to_csv(output_csv, index=False)
    print(f"✓ Predicciones guardadas en {output_csv}")
    
    return submission

if __name__ == "__main__":
    predict_test('model.pkl', 'test.csv')