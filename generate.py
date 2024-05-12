import os
import pickle
from preprocess import plot_spect

import numpy as np
import soundfile as sf

from soundgenerator import SoundGenerator
from vae import VAE
from train import load_fsdd


HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "data/samples/original/"
SAVE_DIR_GENERATED = "data/samples/generated/"
MIN_MAX_VALUES_PATH = "data/min_max_values/min_max_values.pkl"
SPECTROGRAMS_PATH = "data/spectrogram/"


def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]

    return sampled_spectrogrmas, sampled_min_max_values, file_paths


def save_signals(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


if __name__ == "__main__":
    # initialise sound generator
    vae = VAE.load("data/model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    # sample spectrograms + min max values
    sampled_specs, sampled_min_max_values, sampled_paths = select_spectrograms(specs,
                                                                file_paths,
                                                                min_max_values,
                                                                5)

    sampled_paths = [i.split("/")[-1] for i in sampled_paths]

    # generate audio for sampled spectrograms
    signals, _ = sound_generator.generate(sampled_specs, sampled_paths,
                                          sampled_min_max_values)

    # convert spectrogram samples to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_specs, sampled_paths, sampled_min_max_values, name="real")

    # save audio signals
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)