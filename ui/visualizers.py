import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from typing import Optional, Tuple, List, Dict
import io


class SpectrogramVisualizer:
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 6),
        cmap: str = "viridis",
        dpi: int = 100
    ):
        self.figsize = figsize
        self.cmap = cmap
        self.dpi = dpi

    def plot(
        self,
        spectrogram: np.ndarray,
        title: str = "Spectrogram",
        xlabel: str = "Time",
        ylabel: str = "Frequency",
        show_colorbar: bool = True
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if spectrogram.ndim == 3:
            if spectrogram.shape[0] == 3:
                spectrogram = np.transpose(spectrogram, (1, 2, 0))
                spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)
                im = ax.imshow(spectrogram, aspect='auto', origin='lower')
            else:
                spectrogram = spectrogram.mean(axis=0)
                im = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap=self.cmap)
        else:
            im = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap=self.cmap)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        if show_colorbar and spectrogram.ndim != 3:
            plt.colorbar(im, ax=ax, label='Magnitude')

        plt.tight_layout()
        return fig

    def plot_comparison(
        self,
        specs: List[np.ndarray],
        titles: List[str],
        main_title: str = "Spectrogram Comparison"
    ) -> Figure:
        n = len(specs)
        fig, axes = plt.subplots(1, n, figsize=(self.figsize[0] * n // 2, self.figsize[1]), dpi=self.dpi)

        if n == 1:
            axes = [axes]

        for ax, spec, title in zip(axes, specs, titles):
            if spec.ndim == 3 and spec.shape[0] == 3:
                spec = np.transpose(spec, (1, 2, 0))
                spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
                ax.imshow(spec, aspect='auto', origin='lower')
            else:
                if spec.ndim == 3:
                    spec = spec.mean(axis=0)
                ax.imshow(spec, aspect='auto', origin='lower', cmap=self.cmap)

            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')

        fig.suptitle(main_title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def to_image(self, fig: Figure) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()


class HeatmapVisualizer:
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "RdYlBu_r",
        dpi: int = 100
    ):
        self.figsize = figsize
        self.cmap = cmap
        self.dpi = dpi

    def plot_spatial_heatmap(
        self,
        heatmap: np.ndarray,
        room_dimensions: Tuple[float, float] = (5.0, 5.0),
        title: str = "Activity Heatmap",
        show_grid: bool = True
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        extent = [0, room_dimensions[0], 0, room_dimensions[1]]
        im = ax.imshow(heatmap, cmap=self.cmap, origin='lower', extent=extent, interpolation='bilinear')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)

        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.5)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activity Intensity', fontsize=10)

        tx_pos = (0.5, room_dimensions[1] / 2)
        rx_pos = (room_dimensions[0] - 0.5, room_dimensions[1] / 2)

        ax.plot(*tx_pos, 'g^', markersize=15, label='TX')
        ax.plot(*rx_pos, 'rv', markersize=15, label='RX')
        ax.legend(loc='upper right')

        plt.tight_layout()
        return fig

    def plot_temporal_heatmap(
        self,
        data: np.ndarray,
        time_labels: Optional[List[str]] = None,
        activity_labels: Optional[List[str]] = None,
        title: str = "Activity Timeline"
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        im = ax.imshow(data, cmap=self.cmap, aspect='auto')

        if time_labels:
            step = max(1, len(time_labels) // 10)
            ax.set_xticks(range(0, len(time_labels), step))
            ax.set_xticklabels(time_labels[::step], rotation=45, ha='right')

        if activity_labels:
            ax.set_yticks(range(len(activity_labels)))
            ax.set_yticklabels(activity_labels)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Activity', fontsize=12)

        plt.colorbar(im, ax=ax, label='Probability')
        plt.tight_layout()
        return fig

    def plot_confusion_matrix(
        self,
        matrix: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix"
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        matrix_norm = matrix.astype('float') / (matrix.sum(axis=1, keepdims=True) + 1e-8)

        im = ax.imshow(matrix_norm, cmap='Blues')

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{matrix[i, j]}\n({matrix_norm[i, j]:.2f})',
                              ha='center', va='center', fontsize=8)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)

        plt.colorbar(im, ax=ax, label='Normalized Value')
        plt.tight_layout()
        return fig


class SignalVisualizer:
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 100
    ):
        self.figsize = figsize
        self.dpi = dpi

    def plot_csi_amplitude(
        self,
        amplitudes: np.ndarray,
        sample_rate: float = 200.0,
        title: str = "CSI Amplitude"
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        num_samples = amplitudes.shape[0]
        time = np.arange(num_samples) / sample_rate

        if amplitudes.ndim == 2:
            num_subcarriers = amplitudes.shape[1]
            for i in range(0, num_subcarriers, max(1, num_subcarriers // 10)):
                ax.plot(time, amplitudes[:, i], alpha=0.7, label=f'SC {i}')
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.plot(time, amplitudes)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_csi_phase(
        self,
        phases: np.ndarray,
        sample_rate: float = 200.0,
        title: str = "CSI Phase"
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        num_samples = phases.shape[0]
        time = np.arange(num_samples) / sample_rate

        if phases.ndim == 2:
            num_subcarriers = phases.shape[1]
            for i in range(0, num_subcarriers, max(1, num_subcarriers // 10)):
                ax.plot(time, phases[:, i], alpha=0.7, label=f'SC {i}')
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.plot(time, phases)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Phase (rad)', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_doppler(
        self,
        doppler: np.ndarray,
        sample_rate: float = 200.0,
        title: str = "Doppler Signature"
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        num_samples = doppler.shape[0]
        time = np.arange(num_samples) / sample_rate

        if doppler.ndim == 2:
            doppler_mean = np.mean(doppler, axis=1)
        else:
            doppler_mean = doppler

        ax.plot(time, doppler_mean, 'b-', linewidth=1.5)
        ax.fill_between(time, 0, doppler_mean, alpha=0.3)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Doppler Shift', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig

    def plot_activity_timeline(
        self,
        activities: List[str],
        timestamps: List[float],
        confidences: List[float],
        title: str = "Activity Timeline"
    ) -> Figure:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi, sharex=True)

        unique_activities = list(set(activities))
        activity_to_idx = {act: i for i, act in enumerate(unique_activities)}
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_activities)))

        activity_indices = [activity_to_idx[act] for act in activities]
        activity_colors = [colors[idx] for idx in activity_indices]

        ax1.scatter(timestamps, activity_indices, c=activity_colors, s=50, alpha=0.7)
        ax1.set_yticks(range(len(unique_activities)))
        ax1.set_yticklabels(unique_activities)
        ax1.set_ylabel('Activity', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(timestamps, confidences, alpha=0.5)
        ax2.plot(timestamps, confidences, 'b-', linewidth=1.5)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Confidence', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_realtime_dashboard(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        spectrogram: np.ndarray,
        activity: str,
        confidence: float,
        scores: Dict[str, float]
    ) -> Figure:
        fig = plt.figure(figsize=(16, 10), dpi=self.dpi)

        ax1 = fig.add_subplot(2, 3, 1)
        if amplitude.ndim == 2:
            ax1.plot(np.mean(amplitude, axis=1))
        else:
            ax1.plot(amplitude)
        ax1.set_title('Amplitude', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(2, 3, 2)
        if phase.ndim == 2:
            ax2.plot(np.mean(phase, axis=1))
        else:
            ax2.plot(phase)
        ax2.set_title('Phase', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Phase (rad)')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(2, 3, 3)
        if spectrogram.ndim == 3:
            if spectrogram.shape[0] == 3:
                spec_display = np.transpose(spectrogram, (1, 2, 0))
                spec_display = (spec_display - spec_display.min()) / (spec_display.max() - spec_display.min() + 1e-8)
            else:
                spec_display = spectrogram.mean(axis=0)
        else:
            spec_display = spectrogram
        ax3.imshow(spec_display, aspect='auto', origin='lower', cmap='viridis')
        ax3.set_title('Spectrogram', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Frequency')

        ax4 = fig.add_subplot(2, 3, 4)
        activities = list(scores.keys())
        probs = list(scores.values())
        colors = ['green' if act == activity else 'blue' for act in activities]
        bars = ax4.barh(activities, probs, color=colors, alpha=0.7)
        ax4.set_xlim(0, 1)
        ax4.set_title('Activity Probabilities', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Probability')

        ax5 = fig.add_subplot(2, 3, (5, 6))
        ax5.text(0.5, 0.6, activity.upper(), fontsize=48, fontweight='bold',
                ha='center', va='center', transform=ax5.transAxes,
                color='green' if confidence > 0.7 else 'orange' if confidence > 0.5 else 'red')
        ax5.text(0.5, 0.3, f'Confidence: {confidence:.1%}', fontsize=24,
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        ax5.set_title('Current Activity', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig
