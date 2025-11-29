import pandas as pd
from model import VentilatorModel

def train_model(train_path, output_path='model.pkl'):
    print("Cargando datos...")
    df = pd.read_csv(train_path)
    
    # Usar solo una muestra para demo (primeros 10000 registros)
    df = df.head(10000)
    
    print(f"Datos cargados: {len(df)} registros")
    print(f"Breaths únicos: {df['breath_id'].nunique()}")
    
    print("\nEntrenando modelo...")
    model = VentilatorModel()
    mae = model.train(df)
    
    print(f"\n✓ Entrenamiento completado!")
    print(f"MAE en entrenamiento: {mae:.4f} cmH₂O")
    
    print(f"\nGuardando modelo en {output_path}...")
    model.save(output_path)
    print("✓ Modelo guardado!")
    
    return model, mae

if __name__ == "__main__":
    model, mae = train_model('train.csv')