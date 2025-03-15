from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
import pandas as pd


app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title='Car Price Prediction API',
    description='API para predecir el precio de un automóvil'
)


ns = api.namespace('predict', description='Car Price Prediction')

parser = api.parser()
parser.add_argument('Year', type=int, required=True, help='Año del vehículo', location='args')
parser.add_argument('Mileage', type=float, required=True, help='Kilometraje del vehículo', location='args')
parser.add_argument('State', type=str, required=True, help='Estado del vehículo', location='args')
parser.add_argument('Make', type=str, required=True, help='Marca del vehículo', location='args')
parser.add_argument('Model', type=str, required=True, help='Modelo del vehículo', location='args')


prediction_model = api.model('PredictionResult', {
    'predicted_price': fields.Float(description='Precio predicho del vehículo')
})

error_model = api.model('ErrorResult', {
    'message': fields.String(description='Mensaje de error')
})


def load_model():
    try:
        return joblib.load('./Models/Best_model.pkl')
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

model = load_model()


def predict_price(input_data):
    try:

        df = pd.DataFrame([input_data])

        prediction = model.predict(df)[0]
        
        return prediction
    except Exception as e:
        raise Exception(f"Error en la predicción: {e}")

@ns.route('/')
class CarPricePrediction(Resource):
    @api.doc(parser=parser)
    @api.response(200, 'Success', prediction_model)
    @api.response(400, 'Bad Request', error_model)
    def get(self):
        try:
            args = parser.parse_args()
            input_data = {
                'Year': int(args['Year']),
                'Mileage': float(args['Mileage']),
                'State': str(args['State']),
                'Make': str(args['Make']),
                'Model': str(args['Model'])
            }
        
            prediction = predict_price(input_data)
        
            return {"predicted_price": round(float(prediction),2)}, 200
        except Exception as e:
            return {"message": str(e)}, 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')