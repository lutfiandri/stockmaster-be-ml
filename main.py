from flask import Flask, jsonify, request
from flask_cors import CORS


from PIL import Image
import numpy as np
import json

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

    output = stockmodel.run(None, {'image_input': image})
    classnames = ['double-top',
                  'bearish-pennant',
                  'bullish-rectangle',
                  'falling-wedge',
                  'invers-head-and-shoulder',
                  'bullish-pennant',
                  'inverse-cup-and-handle',
                  'double-bottom',
                  'bullish-flag',
                  'bearish-rectangle',
                  'cup-and-handle',
                  'rising-wedge',
                  'head-and-shoulder',
                  'bearish-flag']

    output_index = np.argmax(output)
    output_classname = classnames[np.argmax(output)]

    response = jsonify(json.loads(json.dumps({
        'class_index': output_index,
        'class_name': output_classname
    }, cls=NpEncoder)))
    response.status_code = 200
    return response


if __name__ == "__main__":
    app.run()
