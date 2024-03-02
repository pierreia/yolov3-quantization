from keras.models import load_model
from yolo3 import *
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import load_img
from keras.utils import img_to_array
import random

n_layers = 10
threshold = 0.05

model_path = "coco_yolo3.h5"
model = load_model(model_path)
image = plt.imread('image.jpg')
width, height = image.shape[1], image.shape[0]


x = load_img('image.jpg', target_size=(YOLO3.width, YOLO3.height))
x = img_to_array(x) / 255
x = np.expand_dims(x, axis=0)




def quantize_model(model, threshold, n_layers):
    def quantize_weights(weights):
        quantized_weights = np.zeros_like(weights)
        quantized_weights[np.abs(weights) > threshold] = np.sign(weights[np.abs(weights) > threshold]) * threshold
        return quantized_weights

    conv_idx = [i for i, l in enumerate(model.layers) if "" in str(l)]
    qtz_idx = random.choices(conv_idx, k=n_layers)

    for i in qtz_idx:
        layer = model.layers[i]
        weights = layer.get_weights()
        if len(weights) > 0:  # Check if the layer has weights
            quantized_weights = [quantize_weights(w) for w in weights]
            layer.set_weights(quantized_weights)
    print('Quantized layers:', qtz_idx)
    return model

y = model.predict(x)
boxes = decode_predictions(y, width, height)

print('Detections before quantization:')
for box in boxes:
    print(f'({box.xmin}, {box.ymin}), ({box.xmax}, {box.ymax}), {box.label}, {box.score}')
draw_boxes('image.jpg', boxes, "before_qt.jpg")

model = quantize_model(model,threshold, n_layers)

y = model.predict(x)
boxes = decode_predictions(y, width, height)

print('Detections after quantization:')
for box in boxes:
    print(f'({box.xmin}, {box.ymin}), ({box.xmax}, {box.ymax}), {box.label}, {box.score}')
draw_boxes('image.jpg', boxes, "after_qt.jpg")





