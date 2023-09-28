from flask import Flask, jsonify, request
from flask_cors import CORS


from PIL import Image
import numpy as np
import json
import base64
from io import BytesIO
import time

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


if __name__ == "__main__":
    app.run(port=5001)
