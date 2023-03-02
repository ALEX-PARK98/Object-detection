import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def flatten(img_set: np.array) -> np.array:

flatten_x_train = x_train.reshape(60000,784)
print(flatten_x_train[0])