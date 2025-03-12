import pandas as pd
from joblib import load

def predict_price(data):
    try:
        model = load('./Models/Best_model.pkl')
        preprocessor = load('./Models/preprocessor.pkl')
        
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data])
        

        processed_data = preprocessor.transform(data)
        
        prediction = model.predict(processed_data)
        
        return round(prediction[0],2)
    except Exception as e:
        raise Exception(f"Error en la predicción: {str(e)}")