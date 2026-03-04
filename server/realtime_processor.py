import logging
import numpy as np
import threading
import queue
import time
from typing import Any, Callable, Optional, List, Tuple
from collections import deque
from dataclasses import dataclass

from .udp_receiver import UDPReceiver
from .csi_parser import CSIPacket
from .dsp_pipeline import DSPPipeline, DSPConfig
from .spectrogram import SpectrogramGenerator

logger = logging.getLogger(__name__)


@dataclass
class ProcessedFrame:
    timestamp: float
    amplitude: np.ndarray
    phase: np.ndarray
    doppler: np.ndarray
    spectrogram: Optional[np.ndarray]
    rssi: int
    device_id: int


@dataclass
class InferenceResult:
    timestamp: float
    activity: str
    confidence: float
    all_scores: dict


class RealtimeProcessor:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5500,
        sample_rate: float = 200.0,
        window_size: int = 200,
        hop_size: int = 50,
        num_subcarriers: int = 64
    ):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.num_subcarriers = num_subcarriers

        self._receiver = UDPReceiver(host=host, port=port)

        dsp_config = DSPConfig(
            sample_rate=sample_rate,
            num_subcarriers=num_subcarriers
        )
        self._dsp = DSPPipeline(config=dsp_config)

        self._spectrogram_gen = SpectrogramGenerator(
            sample_rate=sample_rate,
            n_fft=256,
            hop_length=16,
            n_mels=128
        )

        self._csi_buffer = deque(maxlen=window_size * 2)
        self._frame_queue = queue.Queue(maxsize=100)
        self._result_queue = queue.Queue(maxsize=100)

        self._running = False
        self._process_thread: Optional[threading.Thread] = None
        self._inference_callback: Optional[Callable] = None
        self._frame_callbacks: List[Callable] = []

        self._last_window_time = 0.0
        self._frames_processed = 0
        self._inference_model: Optional[Any] = None

    def set_inference_model(self, model: Any) -> None:
        self._inference_model = model

    def register_frame_callback(self, callback: Callable[[ProcessedFrame], None]) -> None:
        self._frame_callbacks.append(callback)

    def register_inference_callback(self, callback: Callable[[InferenceResult], None]) -> None:
        self._inference_callback = callback

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._receiver.start()
        self._receiver.register_callback(self._on_packet)

        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def stop(self) -> None:
        self._running = False
        self._receiver.stop()

        if self._process_thread:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

    def _on_packet(self, packet: CSIPacket) -> None:
        self._csi_buffer.append(packet)

        if len(self._csi_buffer) >= self.window_size:
            current_time = time.time()
            if current_time - self._last_window_time >= self.hop_size / self.sample_rate:
                self._last_window_time = current_time
                window_packets = list(self._csi_buffer)[-self.window_size:]

                try:
                    self._frame_queue.put_nowait(window_packets)
                except queue.Full:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(window_packets)

    def _process_loop(self) -> None:
        while self._running:
            try:
                window_packets = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            processed = self._process_window(window_packets)
            if processed is None:
                continue

            self._frames_processed += 1

            for callback in self._frame_callbacks:
                try:
                    callback(processed)
                except Exception as e:
                    logger.error(f"Frame callback failed: {e}")

            if self._inference_model is not None:
                result = self._run_inference(processed)
                if result and self._inference_callback:
                    try:
                        self._inference_callback(result)
                    except Exception as e:
                        logger.error(f"Inference callback failed: {e}")

    def _process_window(self, packets: List[CSIPacket]) -> Optional[ProcessedFrame]:
        if not packets:
            return None

        csi_matrix = np.array([p.csi_complex for p in packets], dtype=np.complex64)
        amplitudes, phases = self._dsp.process_batch(csi_matrix)

        amplitudes = self._dsp.hampel_filter(amplitudes)
        phases = self._dsp.hampel_filter(phases)

        amplitudes = self._dsp.apply_bandpass(amplitudes)
        phases = self._dsp.apply_bandpass(phases)

        doppler = self._dsp.extract_doppler(phases)

        amp_mean = np.mean(amplitudes, axis=1)
        phase_mean = np.mean(phases, axis=1)
        doppler_mean = np.mean(doppler, axis=1)

        spectrogram = self._spectrogram_gen.generate_rgb_spectrogram(
            amp_mean, phase_mean, doppler_mean
        )

        return ProcessedFrame(
            timestamp=packets[-1].timestamp_s,
            amplitude=amplitudes,
            phase=phases,
            doppler=doppler,
            spectrogram=spectrogram,
            rssi=packets[-1].rssi,
            device_id=packets[-1].device_id
        )

    def _run_inference(self, frame: ProcessedFrame) -> Optional[InferenceResult]:
        if self._inference_model is None or frame.spectrogram is None:
            return None

        try:
            import torch
            spec_tensor = torch.from_numpy(frame.spectrogram).unsqueeze(0).float()

            with torch.no_grad():
                if hasattr(self._inference_model, 'predict'):
                    activity, confidence, all_scores = self._inference_model.predict(spec_tensor)
                else:
                    logits = self._inference_model(spec_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    confidence, pred_idx = torch.max(probs, dim=-1)
                    activity = str(pred_idx.item())
                    all_scores = {str(i): p.item() for i, p in enumerate(probs[0])}

            return InferenceResult(
                timestamp=frame.timestamp,
                activity=activity,
                confidence=confidence if isinstance(confidence, float) else confidence.item(),
                all_scores=all_scores
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None

    def get_latest_frame(self) -> Optional[ProcessedFrame]:
        frames = []
        while True:
            try:
                frames.append(self._frame_queue.get_nowait())
            except queue.Empty:
                break

        if frames:
            return self._process_window(frames[-1])
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def packets_received(self) -> int:
        return self._receiver.packets_received

    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "packets_received": self._receiver.packets_received,
            "bytes_received": self._receiver.bytes_received,
            "dropped_packets": self._receiver.dropped_packets,
            "frames_processed": self._frames_processed,
            "buffer_size": len(self._csi_buffer),
        }
