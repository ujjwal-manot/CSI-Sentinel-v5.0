import numpy as np
from scipy import signal
from typing import Optional, Tuple


class SpectrogramGenerator:
    def __init__(
        self,
        sample_rate: float = 200.0,
        n_fft: int = 256,
        hop_length: int = 16,
        win_length: Optional[int] = None,
        window: str = "hann",
        n_mels: int = 128,
        fmin: float = 0.1,
        fmax: Optional[float] = None,
        power: float = 2.0,
        normalize: bool = True
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2.0
        self.power = power
        self.normalize = normalize

        self._window_func = signal.get_window(self.window, self.win_length)
        self._mel_filterbank = self._create_mel_filterbank()

    def _create_mel_filterbank(self) -> np.ndarray:
        fmin_mel = self._hz_to_mel(self.fmin)
        fmax_mel = self._hz_to_mel(self.fmax)
        mel_points = np.linspace(fmin_mel, fmax_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))

        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)

            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return filterbank

    @staticmethod
    def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def stft(self, signal_data: np.ndarray) -> np.ndarray:
        pad_length = self.n_fft // 2
        padded = np.pad(signal_data, (pad_length, pad_length), mode='reflect')

        num_frames = 1 + (len(padded) - self.n_fft) // self.hop_length
        stft_matrix = np.zeros((self.n_fft // 2 + 1, num_frames), dtype=np.complex64)

        for i in range(num_frames):
            start = i * self.hop_length
            frame = padded[start:start + self.n_fft]

            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))

            windowed = frame[:self.win_length] * self._window_func
            if self.win_length < self.n_fft:
                windowed = np.pad(windowed, (0, self.n_fft - self.win_length))

            spectrum = np.fft.rfft(windowed)
            stft_matrix[:, i] = spectrum

        return stft_matrix

    def spectrogram(self, signal_data: np.ndarray) -> np.ndarray:
        stft_matrix = self.stft(signal_data)
        spec = np.abs(stft_matrix) ** self.power

        if self.normalize:
            spec = spec / (np.max(spec) + 1e-10)

        return spec

    def mel_spectrogram(self, signal_data: np.ndarray) -> np.ndarray:
        spec = self.spectrogram(signal_data)
        mel_spec = self._mel_filterbank @ spec

        if self.normalize:
            mel_spec = mel_spec / (np.max(mel_spec) + 1e-10)

        return mel_spec

    def log_mel_spectrogram(self, signal_data: np.ndarray, ref: float = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
        mel_spec = self.mel_spectrogram(signal_data)
        log_spec = 10.0 * np.log10(np.maximum(mel_spec, amin) / ref)

        log_spec = np.maximum(log_spec, np.max(log_spec) - top_db)

        if self.normalize:
            log_spec = (log_spec - np.min(log_spec)) / (np.max(log_spec) - np.min(log_spec) + 1e-10)

        return log_spec

    def doppler_spectrogram(self, phase_data: np.ndarray) -> np.ndarray:
        if phase_data.ndim == 1:
            return self.log_mel_spectrogram(phase_data)

        num_subcarriers = phase_data.shape[1]
        spectrograms = []

        for i in range(num_subcarriers):
            spec = self.log_mel_spectrogram(phase_data[:, i])
            spectrograms.append(spec)

        combined = np.mean(spectrograms, axis=0)
        return combined

    def generate_for_window(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if amplitude.ndim == 2:
            amp_combined = np.mean(amplitude, axis=1)
            phase_combined = np.mean(phase, axis=1)
        else:
            amp_combined = amplitude
            phase_combined = phase

        amp_spec = self.log_mel_spectrogram(amp_combined)
        doppler_spec = self.log_mel_spectrogram(phase_combined)

        return amp_spec, doppler_spec

    def generate_rgb_spectrogram(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        doppler: np.ndarray
    ) -> np.ndarray:
        amp_spec = self.log_mel_spectrogram(amplitude if amplitude.ndim == 1 else np.mean(amplitude, axis=1))
        phase_spec = self.log_mel_spectrogram(phase if phase.ndim == 1 else np.mean(phase, axis=1))
        doppler_spec = self.log_mel_spectrogram(doppler if doppler.ndim == 1 else np.mean(doppler, axis=1))

        target_shape = amp_spec.shape
        phase_spec = self._resize_to_shape(phase_spec, target_shape)
        doppler_spec = self._resize_to_shape(doppler_spec, target_shape)

        rgb = np.stack([amp_spec, phase_spec, doppler_spec], axis=0)
        return rgb

    @staticmethod
    def _resize_to_shape(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        if arr.shape == target_shape:
            return arr

        from scipy.ndimage import zoom
        zoom_factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
        return zoom(arr, zoom_factors, order=1)
