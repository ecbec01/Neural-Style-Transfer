from copy import deepcopy

import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm


class StyleTransferer:
    def __init__(self, base_model, content_image, style_image,
                 content_layer, style_layers, style_weights):

        # Base model
        base_model.trainable = False
        self.base_model = base_model

        # Content
        self.content_image = content_image
        self.content_layer = content_layer

        # Style
        weights = [weight / sum(style_weights) for weight in style_weights]
        layers = [base_model.get_layer(layer) for layer in style_layers]
        outputs = [layer.output for layer in layers]
        shapes = [out.get_shape().as_list()[1:] for out in outputs]
        self.style_image = style_image
        self.style_layers = style_layers
        self.style_weights = weights
        self.style_shapes = shapes

        # Neural Style Transfer model
        self.model = self.generate_extractor()

        # Original content and style data
        self.orig_content = self.model(content_image)[0]
        self.orig_styles = self.model(style_image)[1:]

        # Initialize generated image
        self.generated_image = deepcopy(content_image)

    def reset_generated_image(self):
        self.generated_image = deepcopy(self.content_image)

    def generate_extractor(self):
        inputs = self.base_model.inputs
        outputs = [*self._get_content_tensors(), *self._get_style_tensors()]
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def _get_content_tensors(self):
        layers = [self.content_layer]
        return [self.base_model.get_layer(layer).output for layer in layers]

    def _get_style_tensors(self):
        layers = self.style_layers
        tensors = [self.base_model.get_layer(layer).output for layer in layers]
        depths = [tensor.get_shape()[-1] for tensor in tensors]
        tensors = [tf.reshape(tensor, [-1, depth])
                   for tensor, depth in zip(tensors, depths)]
        return [tf.matmul(tensor, tensor, transpose_a=True)
                for tensor in tensors]

    def transform(self, n_iter, alpha, beta, opt):
        init_lr = opt._hyper['learning_rate']
        pbar = tqdm(range(n_iter), total=n_iter)
        for epoch in pbar:
            lr = init_lr - init_lr * epoch / (2 * n_iter)
            opt._hyper['learning_rate'] = lr
            cost = self._training_step(alpha, beta, opt)
            pbar.set_description(f'Transforming (cost = {cost:.4e}) ')

    def _training_step(self, alpha, beta, opt):
        with tf.GradientTape() as tape:
            content, *styles = self.model(self.generated_image)
            content_cost = self._content_cost(content)
            style_cost = self._style_cost(styles)
            total_cost = alpha * content_cost + beta * style_cost
        grad = tape.gradient(total_cost, self.generated_image)
        opt.apply_gradients([(grad, self.generated_image)])
        return total_cost

    def _content_cost(self, content):
        return tf.reduce_sum((self.orig_content - content) ** 2) / 2

    def _style_cost(self, styles):
        losses = []
        for idx in range(len(styles)):
            normalizer = np.prod(self.style_shapes[idx], dtype='float32')
            normalizer = normalizer ** 2 * 4
            loss = self.orig_styles[idx] - styles[idx]
            loss = tf.reduce_sum(loss ** 2) / normalizer
            losses.append(loss * self.style_weights[idx])
        return tf.reduce_sum(losses)