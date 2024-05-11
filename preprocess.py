import os
import pickle

import librosa
import numpy as np


class Loader:
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, path):
        signal = librosa.load(
            path, sr=self.sample_rate,
            duration=self.duration,
            mono=self.mono)[0]

        return signal


class Padder:
    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class MinMaxNormalizer:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def normalize(self, array):
        array_min, array_max = array.min(), array.max()
        normalized_array = (array - array_min) / (array_max - array_min)
        normalized_array = normalized_array * (self.max - self.min) + self.min
        return normalized_array

    def denormalize(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        denormalized_array = array * (original_max - original_min) + original_min
        return denormalized_array


class Saver:
    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, path):
        save_path = self._generate_save_path(path)
        np.save(save_path, feature)
        return save_path

    def _generate_save_path(self, path):
        file_name = os.path.split(path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
        self.save(min_max_values, save_path)

    @staticmethod
    def save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)


class PreprocessingPipeline:
    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.min_max_values = dict()

        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(self.loader.sample_rate * self.loader.duration)

    def process(self, audio_files_dir):
        for root, __, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file: {file_path}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self._loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())

    def _is_padding_necessary(self, signal):
        return len(signal) < self._num_expected_samples

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }


if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAM_SAVE_DIR = "data/spectrogram/"
    MIN_MAX_VALUES_SAVE_DIR = "data/min_max_values/"
    FILES_DIR = "data/audio_files/"

    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normalizer = MinMaxNormalizer(0, 1)
    saver = Saver(SPECTROGRAM_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessor = PreprocessingPipeline()
    preprocessor.loader = loader
    preprocessor.padder = padder
    preprocessor.extractor = log_spectrogram_extractor
    preprocessor.normalizer = min_max_normalizer
    preprocessor.saver = saver

    preprocessor.process(FILES_DIR)
