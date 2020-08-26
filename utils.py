import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image


def img_to_input(img):
    img = tf.keras.applications.vgg19.preprocess_input(img)
    img = np.expand_dims(img, 0)
    img = tf.Variable(img)
    return img


def input_to_img(img):
    img = np.squeeze(img.numpy())
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def reshape_img(img, crop_as=None, resize_as=None):
    if crop_as is not None:
        img = img[crop_as[0]:crop_as[1], crop_as[2]:crop_as[3]]
    if resize_as is not None:
        img = Image.fromarray(img)
        img = img.resize((resize_as[0], resize_as[1]))
        img, _ = np.asarray(img), img.close()
    return img