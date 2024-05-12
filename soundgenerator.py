from preprocess import MinMaxNormalizer, plot_spect, plot_spects
import librosa


class SoundGenerator:
    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self.normalizer = MinMaxNormalizer(0, 1)

    def generate(self, spectrograms, labels, min_max_values):
        generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, labels, min_max_values, name="generated")
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms, labels, min_max_values, name="name"):
        signals = []

        log_specs = []

        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            log_spectrogram = spectrogram[:, :, 0]
            log_specs.append(log_spectrogram)
            denorm_log_spectrogram = self.normalizer.denormalize(log_spectrogram, min_max_value["min"], min_max_value["max"])
            spec = librosa.db_to_amplitude(denorm_log_spectrogram)
            signal = librosa.istft(spec, hop_length=self.hop_length)
            signals.append(signal)

        plot_spects(log_specs, labels, name=name)

        return signals
