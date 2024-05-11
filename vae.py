from keras import Model
from keras.layers import (Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape,
                                     Conv2DTranspose, Activation, Lambda, Layer)
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from tensorflow.keras import backend as K
import tensorflow as tf

import os
import pickle
import numpy as np

# tf.compat.v1.disable_eager_execution()


class KLLossLayer(Layer):
    def __init__(self, **kwargs):
        super(KLLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_loss = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=1)
        return kl_loss


class VAE:
    def __init__(self, input_shape, conv_filters, conv_kernels_size, conv_strides, latent_space_dim):
        self.input_shape = input_shape  # [28,28,1]
        self.conv_filters = conv_filters  # [2,4,8]
        self.conv_kernels_size = conv_kernels_size  # [3,5,3]
        self.conv_strides = conv_strides  # [1,2,2]
        self.latent_space_dim = latent_space_dim  # 2
        self.reconstruction_loss_weight = 1000000

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def compile(self, learning_rate=0.0001):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=MeanSquaredError())
                           # metrics=[self._calculate_combined_loss,
                           #          self._calculate_reconstruction_loss,
                           #          self._calculate_kl_loss])

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)

    def save(self, folder="."):
        self._create_folder_if_it_doesnt_exist(folder)
        self._save_parameters(folder)
        self._save_weights(folder)

    @classmethod
    def load(cls, folder):
        params_path = os.path.join(folder, "parameters.pkl")
        with open(params_path, "rb") as f:
            params = pickle.load(f)

        vae = VAE(*params)
        weights_path = os.path.join(folder, ".weights.h5")
        vae.load_weights(weights_path)
        return vae

    def load_weights(self, path):
        self.model.load_weights(path)

    def _calculate_combined_loss(self, y_target, y_predicted):
        # reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        # return reconstruction_loss + self.losses
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)

        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def kl_reconstruction_loss(self, y_target, y_predicted):
        kl_loss = 1 + self.log_var - K.square(self.mu) - K.exp(self.log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(kl_loss)

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        print("\ny_target", type(y_target), y_target.shape, y_target)
        print("\ny_predicted", type(y_predicted), y_target.shape, y_predicted)
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        print("\nreconstruction_loss", type(reconstruction_loss), reconstruction_loss.shape, reconstruction_loss)
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        print("\nself.mu", type(self.mu), self.mu.shape, self.mu)
        print("\nself.log_var", type(self.log_var), self.log_var.shape, self.log_var)
        kl_loss = KLLossLayer()([self.mu, self.log_var])
        # self.model.add_loss(kl_loss)
        print("\nkl_loss", type(kl_loss), kl_loss.shape, kl_loss)
        return kl_loss

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, folder):
        params = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels_size,
            self.conv_strides,
            self.latent_space_dim
        ]

        safe_path = os.path.join(folder, "parameters.pkl")
        with open(safe_path, "wb") as f:
            pickle.dump(params, f)

    def _save_weights(self, folder):
        safe_path = os.path.join(folder, ".weights.h5")
        self.model.save_weights(safe_path)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_vae()

    def _build_vae(self):
        model_input = self._model_input
        mu, log_var, z = self.encoder(model_input)
        model_output = self.decoder(z)
        self.model = Model(model_input, model_output, name="vae")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=(self.latent_space_dim,), name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(target_shape=self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_number = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels_size[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_number}",
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_number}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels_size[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}",
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="decoder_output")(x)
        return output_layer

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        x = encoder_input
        for i in range(self._num_conv_layers):
            x = self._add_conv_layer(i, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels_size[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}",
        )

        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        mu = Dense(self.latent_space_dim, name="mu")(x)
        log_var = Dense(self.latent_space_dim, name="log_var")(x)
        self.mu = mu
        self.log_var = log_var

        def sample_point_from_normal_distribution(args):
            mu, log_var = args
            eps = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
            return mu + K.exp(log_var / 2) * eps

        x = Lambda(sample_point_from_normal_distribution, name="encoder_output", output_shape=(self.latent_space_dim,))([mu, log_var])
        return [mu, log_var, x]

    def reconstruct(self, images):
        mu, log_var, latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations


if __name__ == "__main__":
    vae = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels_size=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=10
    )
    vae.summary()