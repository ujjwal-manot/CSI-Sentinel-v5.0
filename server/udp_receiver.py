import socket
import threading
import queue
from typing import Callable, Optional
from .csi_parser import CSIParser, CSIPacket


class UDPReceiver:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5500,
        buffer_size: int = 2048,
        queue_size: int = 1000
    ):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size

        self._socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._packet_queue = queue.Queue(maxsize=queue_size)
        self._parser = CSIParser()
        self._callbacks: list = []

        self._packets_received = 0
        self._bytes_received = 0

    def start(self) -> None:
        if self._running:
            return

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(1.0)
        self._socket.bind((self.host, self.port))

        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._socket:
            self._socket.close()
            self._socket = None

    def _receive_loop(self) -> None:
        while self._running:
            try:
                data, addr = self._socket.recvfrom(self.buffer_size)
                self._bytes_received += len(data)

                packet = self._parser.parse(data)
                if packet:
                    self._packets_received += 1

                    try:
                        self._packet_queue.put_nowait(packet)
                    except queue.Full:
                        self._packet_queue.get_nowait()
                        self._packet_queue.put_nowait(packet)

                    for callback in self._callbacks:
                        try:
                            callback(packet)
                        except Exception:
                            pass

            except socket.timeout:
                continue
            except OSError:
                break

    def register_callback(self, callback: Callable[[CSIPacket], None]) -> None:
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[CSIPacket], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_packet(self, timeout: float = 1.0) -> Optional[CSIPacket]:
        try:
            return self._packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_packets(self, max_count: int = 100) -> list:
        packets = []
        while len(packets) < max_count:
            try:
                packet = self._packet_queue.get_nowait()
                packets.append(packet)
            except queue.Empty:
                break
        return packets

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def packets_received(self) -> int:
        return self._packets_received

    @property
    def bytes_received(self) -> int:
        return self._bytes_received

    @property
    def dropped_packets(self) -> int:
        return self._parser.dropped_packets

    def reset_stats(self) -> None:
        self._packets_received = 0
        self._bytes_received = 0
        self._parser.reset_stats()
