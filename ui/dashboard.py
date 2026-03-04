import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))

from server.realtime_processor import RealtimeProcessor, ProcessedFrame, InferenceResult
from models.inference import InferenceEngine
from models.wi_clip import WiCLIP
from models.classifier import ActivityClassifier
from ui.visualizers import SpectrogramVisualizer, HeatmapVisualizer, SignalVisualizer


class CSIDashboard:
    def __init__(self):
        self.processor: Optional[RealtimeProcessor] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.frame_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)

        self.spec_viz = SpectrogramVisualizer()
        self.heatmap_viz = HeatmapVisualizer()
        self.signal_viz = SignalVisualizer()

        self.activities = [
            "walk", "run", "sit", "stand", "fall",
            "lie_down", "wave", "jump", "crouch", "empty"
        ]

        self.history: Dict[str, List] = {
            "timestamps": [],
            "activities": [],
            "confidences": [],
            "rssi": []
        }

    def setup_page(self):
        st.set_page_config(
            page_title="CSI-Sentinel v5.0",
            page_icon="📡",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .status-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .status-connected {
            background-color: #C8E6C9;
            border: 2px solid #4CAF50;
        }
        .status-disconnected {
            background-color: #FFCDD2;
            border: 2px solid #F44336;
        }
        .activity-display {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            padding: 2rem;
            border-radius: 1rem;
            margin: 1rem 0;
        }
        .confidence-high { background-color: #C8E6C9; color: #2E7D32; }
        .confidence-medium { background-color: #FFF9C4; color: #F57F17; }
        .confidence-low { background-color: #FFCDD2; color: #C62828; }
        </style>
        """, unsafe_allow_html=True)

    def render_sidebar(self) -> dict:
        st.sidebar.markdown("## Settings")

        config = {}

        st.sidebar.markdown("### Network")
        config["host"] = st.sidebar.text_input("UDP Host", value="0.0.0.0")
        config["port"] = st.sidebar.number_input("UDP Port", value=5500, min_value=1024, max_value=65535)

        st.sidebar.markdown("### Model")
        config["model_type"] = st.sidebar.selectbox("Model Type", ["Wi-CLIP", "Classifier"])
        config["model_path"] = st.sidebar.text_input("Model Path", value="checkpoints/best_model.pt")
        config["confidence_threshold"] = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)

        st.sidebar.markdown("### Processing")
        config["sample_rate"] = st.sidebar.number_input("Sample Rate (Hz)", value=200, min_value=50, max_value=500)
        config["window_size"] = st.sidebar.number_input("Window Size", value=200, min_value=50, max_value=500)
        config["smoothing"] = st.sidebar.slider("Smoothing Window", 1, 20, 5)

        st.sidebar.markdown("### Display")
        config["show_spectrogram"] = st.sidebar.checkbox("Show Spectrogram", value=True)
        config["show_signals"] = st.sidebar.checkbox("Show Raw Signals", value=True)
        config["show_heatmap"] = st.sidebar.checkbox("Show Spatial Heatmap", value=True)
        config["auto_refresh"] = st.sidebar.checkbox("Auto Refresh", value=True)
        config["refresh_rate"] = st.sidebar.slider("Refresh Rate (Hz)", 1, 30, 10)

        return config

    def render_status(self, is_connected: bool, stats: dict):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status_class = "status-connected" if is_connected else "status-disconnected"
            status_text = "Connected" if is_connected else "Disconnected"
            st.markdown(f"""
            <div class="status-box {status_class}">
                <strong>Status:</strong> {status_text}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.metric("Packets Received", stats.get("packets_received", 0))

        with col3:
            st.metric("Frames Processed", stats.get("frames_processed", 0))

        with col4:
            st.metric("Dropped Packets", stats.get("dropped_packets", 0))

    def render_activity_display(self, activity: str, confidence: float, scores: Dict[str, float]):
        if confidence >= 0.7:
            conf_class = "confidence-high"
        elif confidence >= 0.5:
            conf_class = "confidence-medium"
        else:
            conf_class = "confidence-low"

        st.markdown(f"""
        <div class="activity-display {conf_class}">
            {activity.upper()}<br>
            <span style="font-size: 1.5rem;">Confidence: {confidence:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Activity Probabilities")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for act, prob in sorted_scores:
                color = "green" if act == activity else "blue"
                st.progress(prob, text=f"{act}: {prob:.1%}")

        with col2:
            st.markdown("### Probability Chart")
            fig, ax = plt.subplots(figsize=(6, 4))
            activities = list(scores.keys())
            probs = list(scores.values())
            colors = ['#4CAF50' if act == activity else '#2196F3' for act in activities]
            ax.barh(activities, probs, color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    def render_visualizations(
        self,
        frame: Optional[ProcessedFrame],
        config: dict
    ):
        if frame is None:
            st.warning("No data available. Waiting for CSI frames...")
            return

        cols = st.columns(2)

        with cols[0]:
            if config["show_spectrogram"] and frame.spectrogram is not None:
                st.markdown("### Doppler Spectrogram")
                fig = self.spec_viz.plot(
                    frame.spectrogram,
                    title="Real-time Doppler Spectrogram",
                    xlabel="Time (frames)",
                    ylabel="Frequency (Hz)"
                )
                st.pyplot(fig)
                plt.close()

        with cols[1]:
            if config["show_signals"]:
                st.markdown("### CSI Amplitude")
                fig = self.signal_viz.plot_csi_amplitude(
                    frame.amplitude,
                    sample_rate=config["sample_rate"],
                    title="CSI Amplitude (Selected Subcarriers)"
                )
                st.pyplot(fig)
                plt.close()

        if config["show_heatmap"]:
            st.markdown("### Spatial Activity Heatmap")
            heatmap = self._generate_spatial_heatmap(frame)
            fig = self.heatmap_viz.plot_spatial_heatmap(
                heatmap,
                room_dimensions=(5.0, 5.0),
                title="Activity Detection Zone"
            )
            st.pyplot(fig)
            plt.close()

    def _generate_spatial_heatmap(self, frame: ProcessedFrame) -> np.ndarray:
        heatmap = np.zeros((50, 50))

        amp_energy = np.mean(np.abs(frame.amplitude))
        phase_var = np.var(frame.phase)

        center_x, center_y = 25, 25
        for i in range(50):
            for j in range(50):
                dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                heatmap[i, j] = amp_energy * np.exp(-dist / 15) * (1 + 0.1 * np.random.randn())

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap

    def render_history(self):
        if not self.history["timestamps"]:
            st.info("No activity history yet.")
            return

        st.markdown("### Activity History")

        df = pd.DataFrame({
            "Time": self.history["timestamps"][-100:],
            "Activity": self.history["activities"][-100:],
            "Confidence": self.history["confidences"][-100:],
            "RSSI": self.history["rssi"][-100:]
        })

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(df.tail(20), use_container_width=True)

        with col2:
            fig = self.signal_viz.plot_activity_timeline(
                self.history["activities"][-50:],
                list(range(len(self.history["activities"][-50:]))),
                self.history["confidences"][-50:],
                title="Recent Activity Timeline"
            )
            st.pyplot(fig)
            plt.close()

    def render_alerts(self, result: Optional[InferenceResult]):
        if result is None:
            return

        if result.activity == "fall" and result.confidence >= 0.7:
            st.error(f"""
            FALL DETECTED!

            Time: {datetime.now().strftime('%H:%M:%S')}
            Confidence: {result.confidence:.1%}

            Please check on the monitored person immediately.
            """)

            st.balloons()

    def run(self):
        self.setup_page()

        st.markdown('<h1 class="main-header">CSI-Sentinel v5.0</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center;">Non-Line-of-Sight Semantic Wireless Sensing</p>', unsafe_allow_html=True)

        config = self.render_sidebar()

        control_col1, control_col2, control_col3 = st.columns(3)

        with control_col1:
            start_button = st.button("Start Monitoring", type="primary", use_container_width=True)

        with control_col2:
            stop_button = st.button("Stop Monitoring", type="secondary", use_container_width=True)

        with control_col3:
            clear_button = st.button("Clear History", use_container_width=True)

        if clear_button:
            self.history = {"timestamps": [], "activities": [], "confidences": [], "rssi": []}
            st.rerun()

        is_connected = self.processor is not None and self.processor.is_running
        stats = self.processor.get_stats() if self.processor else {}

        self.render_status(is_connected, stats)

        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(["Live Monitoring", "Visualizations", "History", "Settings"])

        with tab1:
            demo_activity = np.random.choice(self.activities)
            demo_confidence = np.random.uniform(0.6, 0.95)
            demo_scores = {act: np.random.uniform(0, 1) for act in self.activities}
            demo_scores[demo_activity] = demo_confidence
            total = sum(demo_scores.values())
            demo_scores = {k: v / total for k, v in demo_scores.items()}

            self.render_activity_display(demo_activity, demo_confidence, demo_scores)

        with tab2:
            demo_frame = ProcessedFrame(
                timestamp=time.time(),
                amplitude=np.random.randn(200, 64).astype(np.float32),
                phase=np.random.randn(200, 64).astype(np.float32),
                doppler=np.random.randn(200, 64).astype(np.float32),
                spectrogram=np.random.randn(3, 128, 128).astype(np.float32),
                rssi=-50,
                device_id=1
            )
            self.render_visualizations(demo_frame, config)

        with tab3:
            if not self.history["timestamps"]:
                for i in range(20):
                    self.history["timestamps"].append(datetime.now().strftime('%H:%M:%S'))
                    self.history["activities"].append(np.random.choice(self.activities))
                    self.history["confidences"].append(np.random.uniform(0.5, 0.95))
                    self.history["rssi"].append(np.random.randint(-70, -40))

            self.render_history()

        with tab4:
            st.markdown("### System Configuration")

            st.json({
                "network": {"host": config["host"], "port": config["port"]},
                "model": {"type": config["model_type"], "path": config["model_path"]},
                "processing": {
                    "sample_rate": config["sample_rate"],
                    "window_size": config["window_size"],
                    "smoothing": config["smoothing"]
                },
                "inference": {"confidence_threshold": config["confidence_threshold"]}
            })

            st.markdown("### About")
            st.markdown("""
            **CSI-Sentinel v5.0** is a privacy-preserving, device-free activity recognition system
            that uses Wi-Fi Channel State Information (CSI) for through-wall monitoring.

            **Features:**
            - Real-time activity detection using Wi-CLIP semantic alignment
            - Zero-shot recognition of untrained activities
            - Synthetic data generation using CSI-Diffusion
            - Sub-₹3000 hardware cost with ESP32-UE

            **Author:** Ujjwal Manot
            **Domain:** Non-Line-of-Sight Semantic Wireless Sensing
            """)


def main():
    dashboard = CSIDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
