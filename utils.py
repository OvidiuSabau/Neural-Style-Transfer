import cv2
import numpy as np
import tensorflow as tf


def readAndNormalize(path, target_height=800, target_width=500):
    im = cv2.imread(path)
    im = cv2.resize(im, dsize=(target_height, target_width))
    means = np.sum(np.sum(im, axis=0), axis=0) / (target_height * target_width)
    variances = np.sum(np.sum(im ** 2, axis=0), axis=0) / (target_height * target_width)
    im = (im - means) / variances
    return means, variances, im


def addGausianNoise(image, noise_sigma=800):
    height, width = image.shape[0], image.shape[1]
    temp_image = cv2.resize(image, dsize=(width, height))

    noise = np.random.randn(height, width) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float32)
    noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
    noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
    noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

    # cv2.imwrite("noisy.jpg", noisy_image)
    means = (np.sum(np.sum(noisy_image, axis=0), axis=0) / (height * width)).astype(np.float32)
    variances = (np.sum(np.sum(noisy_image ** 2, axis=0), axis=0).astype(np.float32) / (height * width)).astype(
        np.float32)
    noisy_image = (noisy_image - means) / variances
    noisy_image = noisy_image[np.newaxis, ...]
    return means, variances, noisy_image


def contentCost(content_activation, target_activation):
    _, nH, nW, nC = content_activation.get_shape().as_list()
    regularizing_term = 4 * nH * nH * nC * nC * nW * nW

    return tf.reduce_sum(tf.square(tf.subtract(content_activation, target_activation))) / regularizing_term


def singleLayerStyleCost(style_activation, target_activation, lamd):
    def correlation_matrix(activation):
        return tf.matmul(activation, tf.transpose(activation))

    _, nH, nW, nC = style_activation.get_shape().as_list()
    regularizing_term = 4 * nH * nH * nC * nC * nW * nW

    style_activation = tf.reshape(tf.transpose(style_activation[0, :, :, :]), shape=(nC, nH * nW))
    target_activation = tf.reshape(tf.transpose(target_activation[0, :, :, :]), shape=(nC, nH * nW))

    style_matrix = correlation_matrix(style_activation)
    target_matrix = correlation_matrix(target_activation)

    return lamd * tf.reduce_sum(tf.square(tf.subtract(style_matrix, target_matrix))) / regularizing_term


def styleCost(style_activations, target_style_activations, lambdas):
    J_style = 0
    for index, key in enumerate(sorted(style_activations.keys())):
        J_style += singleLayerStyleCost(style_activations.get(key), target_style_activations.get(key), lambdas[index])
    return J_style


def totalCost(content_cost, style_cost, alpha, beta):
    return alpha * content_cost + beta * style_cost
