from flask import Flask, jsonify, request
from flask_cors import CORS


from PIL import Image
import numpy as np
import json
import base64
from io import BytesIO, StringIO
import time
import requests
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


from util.stockmodel import get_stockmodel
from util.np_encoder import NpEncoder

app = Flask(__name__)
CORS(app)

stockmodel = get_stockmodel('data/model/stock-pattern-v1.onnx')
print("stockmodel", stockmodel)


@app.route("/")
def index():
    return jsonify({
        'message': 'welcome to stockmaster-be-ml'
    }), 200


@app.route("/stock-pattern", methods=['POST'])
def predict_stock_pattern():
    image = None

    if 'image-type' in request.args.to_dict() and request.args['image-type'] == 'b64':
        print('using base64')
        body = request.json
        if 'image' not in request.json:
            response = jsonify(json.loads(json.dumps(
                {'message': 'No image'}, cls=NpEncoder)))
            response.status_code = 400
            response.headers['Content-Type'] = 'application/json'
            return response

        base64String = body['image'].split(',')[-1]
        print(base64String)

        # im_bytes is a binary image
        im_bytes = base64.b64decode(base64String)
        im_file = BytesIO(im_bytes)  # convert image to file-like object
        image = Image.open(im_file)   # img is now PIL Image object

    else:
        print('using file image')
        if 'image' not in request.files:
            response = jsonify(json.loads(json.dumps(
                {'message': 'No image'}, cls=NpEncoder)))
            response.status_code = 400
            response.headers['Content-Type'] = 'application/json'
            return response

        file = request.files['image']
        image = Image.open(file)

    image = image.resize((48, 48), Image.LANCZOS)

    image = image.convert("RGB")

    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    inference_time_start = time.time()
    output = stockmodel.run(None, {'image_input': image})
    inference_time_stop = time.time()
    inference_duration = inference_time_stop - inference_time_start
    classnames = ['double-top',
                  'bearish-pennant',
                  'bullish-rectangle',
                  'falling-wedge',
                  'inverse-head-and-shoulders',
                  'bullish-pennant',
                  'inverse-cup-and-handle',
                  'double-bottom',
                  'bullish-flag',
                  'bearish-rectangle',
                  'cup-and-handle',
                  'rising-wedge',
                  'head-and-shoulders',
                  'bearish-flag']

    output_index = np.argmax(output)
    output_classname = classnames[output_index]

    response = jsonify(json.loads(json.dumps({
        'classIndex': output_index,
        'className': output_classname,
        'inferenceTimeSeconds': inference_duration
    }, cls=NpEncoder)))
    response.status_code = 200
    return response


@app.route("/stock-updates", methods=["POST"])
def get_stock_updates():
    body = request.json
    if 'symbol' not in body:
        response = jsonify(json.loads(json.dumps(
            {'message': 'No symbol'}, cls=NpEncoder)))
        response.status_code = 400
        response.headers['Content-Type'] = 'application/json'
        return response

    symbol = body['symbol']
    print('symbol', symbol)

    # get public api
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey=6PMW4SIUWX5TSDHX&datatype=csv'
    r = requests.get(url)
    csv_data = r.text

    df = pd.read_csv(StringIO(csv_data))
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    df = df[-52*4:]
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # seasonal decompose
    sd_result = seasonal_decompose(
        x=df['close'], model='additive', period=52)

    # df['observed'] = sd_result.observed
    df['trend'] = sd_result.trend
    df['seasonal'] = sd_result.seasonal
    df['residual'] = sd_result.resid

    # sarimax
    sarimax_model = SARIMAX(endog=df['close'].to_numpy(), order=(
        1, 0, 0), seasonal_order=(2, 1, 0, 52))
    sarimax_result = sarimax_model.fit()

    forecast = sarimax_result.forecast(steps=52)

    next_dates = pd.date_range(
        start=df['timestamp'].iloc[-1] + pd.Timedelta(days=7), periods=52, freq='7D')
    forecast_df = pd.DataFrame({
        'timestamp': next_dates,
        'forecast': forecast
    })

    df['timestamp'] = df['timestamp'].dt.strftime(
        '%Y-%m-%dT%H:%M:%S.%fZ')
    forecast_df['timestamp'] = forecast_df['timestamp'].dt.strftime(
        '%Y-%m-%dT%H:%M:%S.%fZ')

    df = df.drop(columns=['open', 'high', 'low'])
    df_json = df.to_json(orient='records')

    forecast_df_json = forecast_df.to_json(orient='records')

    update_result = {
        'real': json.loads(df_json),
        'forecast': json.loads(forecast_df_json)
    }

    response = jsonify(json.loads(json.dumps(update_result, cls=NpEncoder)))
    response.status_code = 200
    return response


if __name__ == "__main__":
    app.run(port=5001)
