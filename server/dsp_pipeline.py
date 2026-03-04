import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DSPConfig:
    sample_rate: float = 200.0
    num_subcarriers: int = 64
    hampel_window: int = 5
    hampel_threshold: float = 3.0
    butterworth_order: int = 4
    lowpass_cutoff: float = 80.0
    highpass_cutoff: float = 0.3
    enable_phase_sanitization: bool = True
    enable_amplitude_calibration: bool = True


class DSPPipeline:
    def __init__(self, config: Optional[DSPConfig] = None):
        self.config = config or DSPConfig()
        self._init_filters()
        self._phase_ref: Optional[np.ndarray] = None
        self._amp_ref: Optional[np.ndarray] = None
        self._prev_csi: Optional[np.ndarray] = None

    def _init_filters(self) -> None:
        nyq = self.config.sample_rate / 2.0

        low_norm = self.config.lowpass_cutoff / nyq
        low_norm = min(low_norm, 0.99)
        self._lowpass_b, self._lowpass_a = signal.butter(
            self.config.butterworth_order, low_norm, btype='low'
        )

        high_norm = self.config.highpass_cutoff / nyq
        high_norm = max(high_norm, 0.001)
        self._highpass_b, self._highpass_a = signal.butter(
            self.config.butterworth_order, high_norm, btype='high'
        )

    def process_frame(self, csi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        amplitude = np.abs(csi)
        phase = np.angle(csi)

        if self.config.enable_amplitude_calibration:
            if self._amp_ref is None:
                self._amp_ref = amplitude.copy()
            amplitude = amplitude / (self._amp_ref + 1e-10)

        if self.config.enable_phase_sanitization:
            phase = self._sanitize_phase(phase, csi)

        self._prev_csi = csi.copy()

        return amplitude, phase

    def _sanitize_phase(self, phase: np.ndarray, csi: np.ndarray) -> np.ndarray:
        if self._prev_csi is not None:
            conjugate_product = csi * np.conj(self._prev_csi)
            phase_diff = np.angle(conjugate_product)
            return phase_diff

        unwrapped = np.unwrap(phase)
        n = len(unwrapped)
        k = np.arange(n)

        slope = (unwrapped[-1] - unwrapped[0]) / (n - 1) if n > 1 else 0
        intercept = unwrapped[0]
        linear_fit = slope * k + intercept

        return unwrapped - linear_fit

    def process_batch(self, csi_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size, num_subcarriers = csi_batch.shape

        amplitudes = np.zeros((batch_size, num_subcarriers), dtype=np.float32)
        phases = np.zeros((batch_size, num_subcarriers), dtype=np.float32)

        for i in range(batch_size):
            amp, ph = self.process_frame(csi_batch[i])
            amplitudes[i] = amp
            phases[i] = ph

        return amplitudes, phases

    def hampel_filter(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        window = self.config.hampel_window
        threshold = self.config.hampel_threshold

        filtered = data.copy()
        half_window = window // 2

        if axis == 0:
            for i in range(half_window, len(data) - half_window):
                window_data = data[i - half_window:i + half_window + 1]
                median_val = np.median(window_data, axis=0)
                mad = np.median(np.abs(window_data - median_val), axis=0)
                mad_scaled = 1.4826 * mad

                outlier_mask = np.abs(data[i] - median_val) > threshold * mad_scaled
                filtered[i] = np.where(outlier_mask, median_val, data[i])
        else:
            for j in range(data.shape[1]):
                col = data[:, j]
                for i in range(half_window, len(col) - half_window):
                    window_data = col[i - half_window:i + half_window + 1]
                    median_val = np.median(window_data)
                    mad = np.median(np.abs(window_data - median_val))
                    mad_scaled = 1.4826 * mad

                    if np.abs(col[i] - median_val) > threshold * mad_scaled:
                        filtered[i, j] = median_val

        return filtered

    def apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        if len(data) < 3 * self.config.butterworth_order:
            return data

        filtered = signal.filtfilt(self._highpass_b, self._highpass_a, data, axis=0)
        filtered = signal.filtfilt(self._lowpass_b, self._lowpass_a, filtered, axis=0)

        return filtered

    def extract_doppler(self, phase_data: np.ndarray) -> np.ndarray:
        if len(phase_data) < 2:
            return np.zeros_like(phase_data)

        doppler = np.diff(phase_data, axis=0)
        doppler = np.vstack([doppler[0:1], doppler])

        doppler = np.unwrap(doppler, axis=0)

        return doppler

    def pca_denoise(self, data: np.ndarray, n_components: int = 5) -> np.ndarray:
        mean = np.mean(data, axis=0, keepdims=True)
        centered = data - mean

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        n_components = min(n_components, len(S))
        U_reduced = U[:, :n_components]
        S_reduced = S[:n_components]
        Vt_reduced = Vt[:n_components, :]

        reconstructed = U_reduced @ np.diag(S_reduced) @ Vt_reduced
        return reconstructed + mean

    def reset(self) -> None:
        self._phase_ref = None
        self._amp_ref = None
        self._prev_csi = None


class VariationalModeDecomposition:
    def __init__(self, n_modes: int = 5, alpha: float = 2000, tau: float = 0, tol: float = 1e-7, max_iter: int = 500):
        self.n_modes = n_modes
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter

    def decompose(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = len(signal_data)
        T = N

        f_mirror = np.concatenate([signal_data[::-1], signal_data, signal_data[::-1]])
        f_hat = np.fft.fft(f_mirror)
        f_hat = f_hat[:len(f_hat) // 2 + 1]

        freqs = np.arange(len(f_hat)) / len(f_mirror)

        u_hat = np.zeros((self.n_modes, len(f_hat)), dtype=complex)
        omega = np.zeros((self.n_modes, self.max_iter))

        for k in range(self.n_modes):
            omega[k, 0] = (0.5 / self.n_modes) * k

        lambda_hat = np.zeros(len(f_hat), dtype=complex)

        for iteration in range(self.max_iter - 1):
            u_hat_sum = np.sum(u_hat, axis=0)

            for k in range(self.n_modes):
                sum_minus_k = u_hat_sum - u_hat[k]
                numerator = f_hat - sum_minus_k + lambda_hat / 2
                denominator = 1 + 2 * self.alpha * (freqs - omega[k, iteration]) ** 2
                u_hat[k] = numerator / denominator

                omega_num = np.sum(freqs * np.abs(u_hat[k]) ** 2)
                omega_den = np.sum(np.abs(u_hat[k]) ** 2) + 1e-10
                omega[k, iteration + 1] = omega_num / omega_den

            lambda_hat = lambda_hat + self.tau * (f_hat - np.sum(u_hat, axis=0))

            convergence = np.sum(np.abs(omega[:, iteration + 1] - omega[:, iteration]) ** 2)
            if convergence < self.tol:
                break

        modes = np.zeros((self.n_modes, N))
        for k in range(self.n_modes):
            u_hat_full = np.concatenate([u_hat[k], np.conj(u_hat[k][-2:0:-1])])
            u_full = np.fft.ifft(u_hat_full).real
            modes[k] = u_full[N:2 * N]

        center_freqs = omega[:, iteration + 1]

        return modes, center_freqs
