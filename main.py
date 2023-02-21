import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
with open('target_action_prediction_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    utm_source: str = None
    utm_medium: str = None
    utm_campaign: str = None
    utm_adcontent: str = None
    utm_keyword: str = None
    device_category: str = None
    device_os: str = None
    device_brand: str = None
    device_model: str = None
    device_browser: str = None
    device_screen_width: int = None
    device_screen_height: int = None
    geo_country: str = None
    geo_city: str = None


class Prediction(BaseModel):
    utm_medium: str
    utm_campaign: str
    device_category: str
    device_brand: str
    geo_country: str
    geo_city: str
    Result: int

@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    return {
        'utm_medium': form.utm_medium,
        'utm_campaign': form.utm_campaign,
        'device_category': form.device_category,
        'device_brand': form.device_brand,
        'geo_country': form.geo_country,
        'geo_city': form.geo_city,
        'Result': y[0]
    }
