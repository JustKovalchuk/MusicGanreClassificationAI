from autoencoder import Autoencoder
from vae import VAE
from keras.datasets import mnist
import os
import numpy as np


LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 100


def load_mnist():
    # mndata = MNIST('./dir_with_mnist_data_files')
    # images, labels = mndata.load_training()
    # x_train, x_test, y_train, y_test = train_test_split(images, labels)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, file_paths


def train(x_train, learning_rate, batch_size, epochs):
    # vae = VAE(
    #     input_shape=(28, 28, 1),
    #     conv_filters=(32, 64, 64, 64),
    #     conv_kernels_size=(3, 3, 3, 3),
    #     conv_strides=(1, 2, 2, 1),
    #     latent_space_dim=16
    # )
    vae = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels_size=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, batch_size, epochs)
    return vae


if __name__ == "__main__":
    x_train, _ = load_fsdd("data/spectrogram/")
    print(len(x_train))
    print(x_train[0].shape)
    # tf.compat.v1.disable_eager_execution()
    vae = train(x_train[:], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    vae.save("model")
    # ae2 = VAE.load("model")
    # ae2.summary()
