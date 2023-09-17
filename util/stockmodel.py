import onnxruntime as ort
import numpy as np


def get_stockmodel(model_path: str):
    session = ort.InferenceSession(model_path)
    return session

# Run the inference
# output = session.run(None, {'image_input': img})

# Process the output data
# print(output)
