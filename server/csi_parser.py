import struct
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CSIPacket:
    magic: int
    version: int
    device_id: int
    timestamp_us: int
    sequence: int
    rssi: int
    noise_floor: int
    channel: int
    secondary_channel: int
    num_subcarriers: int
    csi_complex: np.ndarray

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.csi_complex)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.csi_complex)

    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_us / 1000.0

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_us / 1000000.0


class CSIParser:
    HEADER_FORMAT = "<HBBIHBBBBH"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    MAGIC_VALUE = 0xC510
    MAX_TRACKED_DEVICES = 256

    def __init__(self, num_subcarriers: int = 64):
        self.num_subcarriers = num_subcarriers
        self._last_sequence: Dict[int, int] = {}
        self._dropped_packets = 0

    def parse(self, data: bytes) -> Optional[CSIPacket]:
        if len(data) < self.HEADER_SIZE:
            return None

        try:
            header = struct.unpack(self.HEADER_FORMAT, data[:self.HEADER_SIZE])
        except struct.error:
            return None

        magic = header[0]
        if magic != self.MAGIC_VALUE:
            return None

        version = header[1]
        device_id = header[2]
        timestamp_us = header[3]
        sequence = header[4]
        rssi = header[5]
        noise_floor = header[6]
        channel = header[7]
        secondary_channel = header[8]
        num_subcarriers = header[9]

        if device_id in self._last_sequence:
            expected = (self._last_sequence[device_id] + 1) & 0xFFFF
            if sequence != expected and sequence != 0xFFFF:
                self._dropped_packets += (sequence - expected) & 0xFFFF

        if len(self._last_sequence) >= self.MAX_TRACKED_DEVICES:
            oldest_key = next(iter(self._last_sequence))
            del self._last_sequence[oldest_key]
        self._last_sequence[device_id] = sequence

        if sequence == 0xFFFF:
            return None

        csi_data_start = self.HEADER_SIZE
        csi_data_length = num_subcarriers * 2

        if len(data) < csi_data_start + csi_data_length:
            return None

        csi_bytes = data[csi_data_start:csi_data_start + csi_data_length]
        csi_raw = np.frombuffer(csi_bytes, dtype=np.int8)

        real_parts = csi_raw[0::2].astype(np.float32)
        imag_parts = csi_raw[1::2].astype(np.float32)
        csi_complex = real_parts + 1j * imag_parts
        csi_complex = csi_complex.astype(np.complex64)

        return CSIPacket(
            magic=magic,
            version=version,
            device_id=device_id,
            timestamp_us=timestamp_us,
            sequence=sequence,
            rssi=rssi,
            noise_floor=noise_floor,
            channel=channel,
            secondary_channel=secondary_channel,
            num_subcarriers=num_subcarriers,
            csi_complex=csi_complex
        )

    def parse_batch(self, data_list: List[bytes]) -> List[CSIPacket]:
        packets = []
        for data in data_list:
            packet = self.parse(data)
            if packet is not None:
                packets.append(packet)
        return packets

    @property
    def dropped_packets(self) -> int:
        return self._dropped_packets

    def reset_stats(self) -> None:
        self._last_sequence.clear()
        self._dropped_packets = 0
