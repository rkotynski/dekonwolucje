"""Qt GUI and compatibility layer for the deconvolution application.

Concrete numerical algorithms live in :mod:`deconv.algorithms`; shared data
models and numerical helpers live in :mod:`deconv.core.runtime`.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import itertools
import time
import json
import os
import sys
import threading
import gc
from pathlib import Path

import numpy as np
from scipy.io import savemat
from PIL import Image

from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDoubleSpinBox as _QtQDoubleSpinBox, QSpinBox as _QtQSpinBox,
    QScrollArea, QSplitter, QGridLayout, QProgressBar, QAbstractSpinBox, QActionGroup
)
from deconv.gui.translated_widgets import (
    QTabWidget, QPushButton, QLabel, QFileDialog, QComboBox, QFormLayout,
    QGroupBox, QTextEdit, QMessageBox, QSlider, QCheckBox, QLineEdit, QAction,
    QInputDialog, QStatusBar,
)
from deconv.i18n import (
    get_language, language_display_name, register_retranslator, retranslate_all,
    set_language, translate,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from deconv.core.runtime import *
from deconv.optim.auto_process import AutoCancelledError, AutoProcessError, AutoNumericalProcessClient
from deconv.algorithms import (
    AlgorithmRegistry, DeconvolutionAlgorithm, DeconvolutionResult,
    BlindRichardsonLucyDeconvolution, TorchBlindAdamTVMAPDeconvolution,
    TorchAdamTVMAPDeconvolution,
)


def _calculation_image_from_state(state: Dict[str, Any]) -> Optional[GrayImage]:
    """Return the image that reconstruction algorithms will actually receive."""
    degraded = state.get("degraded")
    return degraded if isinstance(degraded, GrayImage) else None


def _synchronize_calculation_psf(
    state: Dict[str, Any],
    image_shape: Optional[Tuple[int, int]] = None,
    *,
    algorithm_model: str = LINEAR_SAME,
) -> Optional[PSF]:
    """Build and store the single PSF used by all numerical calculations.

    ``state['psf']`` remains the full thresholded source array for the full-PSF
    preview. ``state['calculation_psf']`` is the selected rectangle after
    zero-padding outside the source array, nonnegative projection and unit-sum
    normalization.
    """
    psf = state.get("psf")
    if not isinstance(psf, PSF):
        state.pop("calculation_psf", None)
        return None
    if image_shape is None:
        image = _calculation_image_from_state(state) or state.get("image")
        if isinstance(image, GrayImage):
            image_shape = tuple(int(v) for v in image.data.shape)
        else:
            image_shape = tuple(int(v) for v in psf.kernel.shape)
    calculation = calculation_psf_for_image(
        psf, tuple(int(v) for v in image_shape),
        algorithm_convolution_model=algorithm_model,
    )
    if calculation is not None:
        calculation.metadata = dict(calculation.metadata or {})
        calculation.metadata.update({
            "shown_as_calculation_psf": True,
            "threshold_fraction": float((psf.metadata or {}).get("lower_threshold_fraction", 0.0)),
        })
        state["calculation_psf"] = calculation
    else:
        state.pop("calculation_psf", None)
    return calculation


def _calculation_data_summary(state: Dict[str, Any]) -> str:
    image = _calculation_image_from_state(state)
    psf = state.get("calculation_psf")
    image_text = "none" if image is None else f"{image.data.shape[1]}×{image.data.shape[0]} px"
    if isinstance(psf, PSF):
        psf_text = f"{psf.kernel.shape[1]}×{psf.kernel.shape[0]} px, sum={float(psf.kernel.sum()):.8g}"
    else:
        psf_text = "none"
    return f"Calculation input: {image_text}; calculation PSF: {psf_text}."



class QDoubleSpinBox(_QtQDoubleSpinBox):
    """More editable double spin box with compact, non-padded display text.

    Qt normally validates text against the configured number of decimal places
    while ``keyboardTracking`` may immediately propagate every intermediate
    keystroke to connected slots.  In a parameter-heavy GUI that combination
    can reset the editor before the user finishes inserting a digit.  This
    subclass keeps 15 decimal places internally, emits changes only after the
    edit is committed, and removes insignificant trailing zeros from the
    displayed text.
    """

    _INTERNAL_DECIMALS = 15

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._requested_decimals = 6
        self._source_tooltip = ""
        super().__init__(parent)
        register_retranslator(self.retranslate)
        super().setDecimals(self._INTERNAL_DECIMALS)
        self.setKeyboardTracking(False)
        self.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        self.setMinimumWidth(145)
        self.lineEdit().setMaxLength(64)
        self.setAlignment(Qt.AlignRight)

    def setDecimals(self, precision: int) -> None:
        self._requested_decimals = max(0, int(precision))
        super().setDecimals(max(self._INTERNAL_DECIMALS, self._requested_decimals))

    def setToolTip(self, text: str) -> None:  # type: ignore[override]
        self._source_tooltip = str(text or "")
        super().setToolTip(translate(self._source_tooltip))

    def retranslate(self) -> None:
        if self._source_tooltip:
            super().setToolTip(translate(self._source_tooltip))

    def textFromValue(self, value: float) -> str:
        precision = max(self._INTERNAL_DECIMALS, self._requested_decimals)
        locale = self.locale()
        text = locale.toString(float(value), 'f', precision)
        decimal_point = locale.decimalPoint()
        if decimal_point in text:
            text = text.rstrip('0').rstrip(decimal_point)
        minus = locale.negativeSign()
        if text in ('', minus, minus + '0'):
            return '0'
        return text


class QSpinBox(_QtQSpinBox):
    """Integer spin box that does not interrupt an unfinished text edit."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._source_tooltip = ""
        super().__init__(parent)
        register_retranslator(self.retranslate)
        self.setKeyboardTracking(False)
        self.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        self.setMinimumWidth(100)
        self.lineEdit().setMaxLength(32)
        self.setAlignment(Qt.AlignRight)

    def setToolTip(self, text: str) -> None:  # type: ignore[override]
        self._source_tooltip = str(text or "")
        super().setToolTip(translate(self._source_tooltip))

    def retranslate(self) -> None:
        if self._source_tooltip:
            super().setToolTip(translate(self._source_tooltip))

APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

def _default_config_dir() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / "dekonwolucje"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "dekonwolucje"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "dekonwolucje"

CONFIG_DIR = _default_config_dir()
SETTINGS_FILE = CONFIG_DIR / "settings.json"
ACTIVE_PROFILE_FILE = CONFIG_DIR / "active_profile.json"
LEGACY_SETTINGS_FILE = Path.home() / ".deconvolution_gui_settings.json"
PACKAGE_LEGACY_SETTINGS_FILE = APP_DIR / "deconvolution_gui_settings.json"
SETTINGS_SCHEMA_VERSION = 103


def _active_settings_file() -> Path:
    """Return the last explicitly selected profile, falling back to the default."""
    try:
        data = json.loads(ACTIVE_PROFILE_FILE.read_text(encoding="utf-8"))
        raw_path = str(data.get("settings_file", "")).strip()
        if raw_path:
            return Path(raw_path).expanduser()
    except Exception:
        pass
    return SETTINGS_FILE


def _remember_active_settings_file(path: Path) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        ACTIVE_PROFILE_FILE.write_text(
            json.dumps({"settings_file": str(Path(path).expanduser().resolve())}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

# Numerical work is serialized across Auto and Test workers.  PyTorch/CUDA is
# generally thread-safe, but creating and destroying independent Qt worker
# threads around large CUDA workloads can expose allocator/driver races,
# especially in IDE kernels that reload modules.
_NUMERICAL_WORK_LOCK = threading.RLock()
_COMPUTE_STATE_LOCK = threading.Lock()
_COMPUTE_OWNER: Optional[str] = None

def _try_begin_numerical_work(owner: str) -> bool:
    global _COMPUTE_OWNER
    with _COMPUTE_STATE_LOCK:
        if _COMPUTE_OWNER is not None:
            return False
        _COMPUTE_OWNER = str(owner)
        return True

def _end_numerical_work(owner: str) -> None:
    global _COMPUTE_OWNER
    with _COMPUTE_STATE_LOCK:
        if _COMPUTE_OWNER == str(owner):
            _COMPUTE_OWNER = None

def _current_numerical_owner() -> Optional[str]:
    with _COMPUTE_STATE_LOCK:
        return _COMPUTE_OWNER

def _is_cuda_oom(exc: BaseException) -> bool:
    if TORCH_AVAILABLE:
        try:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message or "cufft" in message)

def _safe_torch_worker_cleanup() -> None:
    """Synchronize CUDA and release cached blocks before a worker exits.

    All algorithm results are converted to NumPy before this function is called.
    Keeping cleanup in the same worker thread that used CUDA avoids leaving
    pending kernels or cuFFT workspaces behind when the QThread is destroyed.
    """
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass



def _choose_mat_variable(parent: QWidget, path: str, preferred_keys: Tuple[str, ...], purpose: str) -> Optional[str]:
    """Ask the user which 2D numeric MAT variable should be loaded."""
    if Path(path).suffix.lower() != ".mat":
        return None
    candidates = mat_array_candidates(path)
    if not candidates:
        raise ValueError("The MAT file contains no numeric two-dimensional arrays.")
    ordered = [key for key in preferred_keys if key in candidates]
    ordered.extend(key for key in candidates if key not in ordered)
    if len(ordered) == 1:
        return ordered[0]
    labels = [f"{key}    {candidates[key].shape[1]} x {candidates[key].shape[0]}    {candidates[key].dtype}" for key in ordered]
    selected, ok = QInputDialog.getItem(
        parent,
        f"Select MAT variable for {purpose}",
        "Two-dimensional array:",
        labels,
        0,
        False,
    )
    if not ok:
        return ""
    return ordered[labels.index(selected)]

class ImageCanvas(FigureCanvas):
    def __init__(self, title: str = "") -> None:
        self.fig = Figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.title = str(title or "")
        self._current_title_source = self.title
        register_retranslator(self.retranslate)
        self._image_artist = None
        self._selection_rectangle = None
        self._selection_geometry: Optional[Tuple[int, int, int, int]] = None
        self._selection_image_shape: Optional[Tuple[int, int]] = None
        self._selection_edit_enabled = False
        self._selection_edit_callback = None
        self._selection_drag = None
        self.mpl_connect("button_press_event", self._selection_mouse_press)
        self.mpl_connect("motion_notify_event", self._selection_mouse_move)
        self.mpl_connect("button_release_event", self._selection_mouse_release)
        self.mpl_connect("scroll_event", self._selection_mouse_wheel)
        self.ax.axis("off")

    @staticmethod
    def normalize_for_display(data: np.ndarray) -> np.ndarray:
        """Normalize an array only for visualization, preserving the original data elsewhere."""
        arr = np.asarray(data, dtype=np.float32)
        arr = np.nan_to_num(arr)
        mn, mx = float(arr.min()), float(arr.max())
        if mx <= mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def set_selection_editing(self, enabled: bool, callback=None) -> None:
        """Enable mouse moving/resizing of the displayed rectangular selection.

        Clicking inside the rectangle moves it. Dragging near an edge or corner
        changes its width and/or height. The mouse wheel changes both dimensions
        around the current centre. The callback receives ``(x0, y0, width, height)``.
        """
        self._selection_edit_enabled = bool(enabled)
        self._selection_edit_callback = callback if enabled else None
        if not enabled:
            self._selection_drag = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def _update_selection_patch(self, geometry: Tuple[int, int, int, int]) -> None:
        x0, y0, width, height = [int(v) for v in geometry]
        self._selection_geometry = (x0, y0, width, height)
        if self._selection_rectangle is None:
            self._selection_rectangle = Rectangle(
                (x0 - 0.5, y0 - 0.5), width, height,
                fill=False, edgecolor="red", linewidth=1.5,
                linestyle="--", clip_on=True,
            )
            self.ax.add_patch(self._selection_rectangle)
        else:
            self._selection_rectangle.set_xy((x0 - 0.5, y0 - 0.5))
            self._selection_rectangle.set_width(width)
            self._selection_rectangle.set_height(height)
        self.draw_idle()

    def _selection_mouse_press(self, event) -> None:
        if (
            not self._selection_edit_enabled
            or self._selection_geometry is None
            or self._selection_rectangle is None
            or event.inaxes is not self.ax
            or event.button != 1
            or event.xdata is None
            or event.ydata is None
        ):
            return
        x0, y0, width, height = self._selection_geometry
        left, right = x0 - 0.5, x0 + width - 0.5
        top, bottom = y0 - 0.5, y0 + height - 0.5
        px, py = float(event.x), float(event.y)
        left_px = self.ax.transData.transform((left, event.ydata))[0]
        right_px = self.ax.transData.transform((right, event.ydata))[0]
        top_py = self.ax.transData.transform((event.xdata, top))[1]
        bottom_py = self.ax.transData.transform((event.xdata, bottom))[1]
        inside = left <= event.xdata <= right and top <= event.ydata <= bottom
        if not inside:
            return
        edge_tol = 9.0
        edges = {
            "left": abs(px - left_px) <= edge_tol,
            "right": abs(px - right_px) <= edge_tol,
            "top": abs(py - top_py) <= edge_tol,
            "bottom": abs(py - bottom_py) <= edge_tol,
        }
        mode = "resize" if any(edges.values()) else "move"
        self._selection_drag = {
            "mode": mode,
            "edges": edges,
            "press_x": float(event.xdata),
            "press_y": float(event.ydata),
            "geometry": (x0, y0, width, height),
        }
        self.setCursor(Qt.SizeAllCursor if mode == "move" else Qt.SizeFDiagCursor)

    def _selection_mouse_move(self, event) -> None:
        drag = self._selection_drag
        if drag is None or event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        x0, y0, width, height = drag["geometry"]
        image_shape = self._selection_image_shape
        if image_shape is None:
            return
        image_h, image_w = image_shape
        centre_x = x0 + width // 2
        centre_y = y0 + height // 2
        new_width, new_height = width, height
        if drag["mode"] == "move":
            dx = int(round(float(event.xdata) - drag["press_x"]))
            dy = int(round(float(event.ydata) - drag["press_y"]))
            centre_x = int(np.clip(centre_x + dx, 0, max(0, image_w - 1)))
            centre_y = int(np.clip(centre_y + dy, 0, max(0, image_h - 1)))
        else:
            edges = drag.get("edges", {})
            if edges.get("left") or edges.get("right"):
                radius_x = max(0, int(round(abs(float(event.xdata) - centre_x))))
                new_width = min(2 * radius_x + 1, max(1, image_w))
                if new_width > 1 and new_width % 2 == 0:
                    new_width -= 1
            if edges.get("top") or edges.get("bottom"):
                radius_y = max(0, int(round(abs(float(event.ydata) - centre_y))))
                new_height = min(2 * radius_y + 1, max(1, image_h))
                if new_height > 1 and new_height % 2 == 0:
                    new_height -= 1
        new_x0 = int(centre_x - new_width // 2)
        new_y0 = int(centre_y - new_height // 2)
        self._update_selection_patch((new_x0, new_y0, new_width, new_height))

    def _selection_mouse_release(self, event) -> None:
        if self._selection_drag is None:
            return
        self._selection_drag = None
        self.setCursor(Qt.CrossCursor if self._selection_edit_enabled else Qt.ArrowCursor)
        if self._selection_geometry is not None and callable(self._selection_edit_callback):
            self._selection_edit_callback(tuple(self._selection_geometry))

    def _selection_mouse_wheel(self, event) -> None:
        """Resize the selection around its centre with the mouse wheel.

        Wheel-up zooms into the selected PSF region (smaller rectangle), while
        wheel-down includes a larger region. Width and height are scaled by the
        same factor, approximately preserving a rectangular aspect ratio.
        """
        if (
            not self._selection_edit_enabled
            or self._selection_geometry is None
            or event.inaxes is not self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return
        x0, y0, width, height = self._selection_geometry
        image_shape = self._selection_image_shape
        if image_shape is None:
            return
        image_h, image_w = [int(v) for v in image_shape]
        step = float(getattr(event, "step", 0.0) or 0.0)
        if step == 0.0:
            return
        factor = (1.0 / 1.10) ** step
        new_width = int(np.clip(round(width * factor), 1, max(1, image_w)))
        new_height = int(np.clip(round(height * factor), 1, max(1, image_h)))
        if new_width == width and image_w > 1:
            new_width = int(np.clip(width + (-1 if step > 0 else 1), 1, image_w))
        if new_height == height and image_h > 1:
            new_height = int(np.clip(height + (-1 if step > 0 else 1), 1, image_h))
        centre_x = x0 + (width - 1) / 2.0
        centre_y = y0 + (height - 1) / 2.0
        new_x0 = int(round(centre_x - (new_width - 1) / 2.0))
        new_y0 = int(round(centre_y - (new_height - 1) / 2.0))
        self._update_selection_patch((new_x0, new_y0, new_width, new_height))
        if callable(self._selection_edit_callback):
            self._selection_edit_callback(tuple(self._selection_geometry))

    def show_image(
        self,
        data: Optional[np.ndarray],
        title: Optional[str] = None,
        normalize_display: bool = False,
        selection_rectangle: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """Show or update an image without rebuilding the Matplotlib axes.

        Keeping the existing ``AxesImage`` is much faster than ``ax.clear()`` +
        ``imshow()`` for every display-level change.
        """
        if data is None:
            if self._image_artist is not None:
                self._image_artist.remove()
                self._image_artist = None
            if self._selection_rectangle is not None:
                self._selection_rectangle.remove()
                self._selection_rectangle = None
            self._selection_geometry = None
            self._selection_image_shape = None
        else:
            shown = self.normalize_for_display(data) if normalize_display else np.asarray(data, dtype=np.float32)
            shown = np.nan_to_num(shown, copy=False)
            if self._image_artist is None:
                self._image_artist = self.ax.imshow(
                    shown, cmap="gray", vmin=0, vmax=1, interpolation="nearest", origin="upper"
                )
            else:
                self._image_artist.set_data(shown)
                self._image_artist.set_clim(0.0, 1.0)
            height, width = shown.shape[:2]
            # set_data() does not update the AxesImage extent.  Without this,
            # switching between a small PSF crop and the full loaded array
            # leaves the image in the old top-left-sized rectangle while the
            # red selection overlay already uses full-array coordinates.
            extent = (-0.5, width - 0.5, height - 0.5, -0.5)
            self._image_artist.set_extent(extent)
            self._selection_image_shape = (int(height), int(width))
            self.ax.set_xlim(-0.5, width - 0.5)
            self.ax.set_ylim(height - 0.5, -0.5)
            if selection_rectangle is None:
                if self._selection_rectangle is not None:
                    self._selection_rectangle.remove()
                    self._selection_rectangle = None
                self._selection_geometry = None
            else:
                x0, y0, rect_width, rect_height = [int(round(float(v))) for v in selection_rectangle]
                self._update_selection_patch((x0, y0, rect_width, rect_height))
        self._current_title_source = str(title or self.title)
        self.ax.set_title(translate(self._current_title_source))
        self.ax.axis("off")
        self.draw_idle()

    def retranslate(self) -> None:
        self.ax.set_title(translate(self._current_title_source or self.title))
        self.draw_idle()


class HistogramCanvas(FigureCanvas):
    """Compact 256-bin histogram aligned pixel-for-pixel with a [0, 1] slider.

    The axes deliberately occupy the full horizontal extent of the canvas.  The
    canvas is placed in the same layout column as its slider, so both widgets
    always have exactly the same width and a horizontal position in the
    histogram corresponds directly to the same slider position.
    """

    def __init__(self, title: str = "") -> None:
        self.fig = Figure(figsize=(5.0, 1.25))
        self.ax = self.fig.add_axes((0.0, 0.20, 1.0, 0.63))
        super().__init__(self.fig)
        self.title = str(title or "")
        self._current_title_source = self.title
        register_retranslator(self.retranslate)
        self.setMinimumHeight(95)
        self.setMaximumHeight(135)

    def show_histogram(
        self,
        data: Optional[np.ndarray],
        title: Optional[str] = None,
        threshold: Optional[float] = None,
        relative_to_peak: bool = False,
    ) -> None:
        self.ax.clear()
        self._current_title_source = str(title or self.title)
        self.ax.set_title(translate(self._current_title_source), fontsize=9, pad=2)
        if data is not None:
            arr = np.asarray(data, dtype=np.float64).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                arr = np.maximum(arr, 0.0)
                maximum = float(np.max(arr))
                if maximum > 0.0:
                    shown = arr / maximum if relative_to_peak else arr
                    # Threshold controls use [0, 1], therefore values outside
                    # that interval are clipped only for the histogram display.
                    shown = np.clip(shown, 0.0, 1.0)
                    self.ax.hist(shown, bins=256, range=(0.0, 1.0), log=True)
        if threshold is not None:
            self.ax.axvline(float(np.clip(threshold, 0.0, 1.0)), linewidth=1.2)
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
        self.ax.set_yticks([])
        self.ax.tick_params(axis="x", labelsize=7, pad=1)
        self.ax.margins(x=0.0)
        self.draw_idle()

    def retranslate(self) -> None:
        self.ax.set_title(translate(self._current_title_source or self.title), fontsize=9, pad=2)
        self.draw_idle()


class LoadGenerateTab(QWidget):
    calculationPsfSupportChanged = pyqtSignal(int)
    resetRequested = pyqtSignal()
    exitRequested = pyqtSignal()

    def __init__(self, app_state: Dict[str, Any]) -> None:
        super().__init__()
        self.state = app_state
        self.last_image_directory = str(Path.home())
        layout = QVBoxLayout(self)

        buttons = QHBoxLayout()
        btn_load_img = QPushButton("Load image")
        btn_load_psf = QPushButton("Load PSF")
        btn_synth_img = QPushButton("Generate test image")
        btn_psf = QPushButton("Generate selected PSF")
        btn_degrade = QPushButton("Generate degraded input")
        btn_reset = QPushButton("Reset")
        btn_exit = QPushButton("Exit")
        buttons.addWidget(btn_load_img)
        buttons.addWidget(btn_load_psf)
        buttons.addWidget(btn_synth_img)
        buttons.addWidget(btn_psf)
        buttons.addWidget(btn_degrade)
        buttons.addStretch(1)
        buttons.addWidget(btn_reset)
        buttons.addWidget(btn_exit)
        layout.addLayout(buttons)

        controls_content = QWidget()
        controls_content_layout = QVBoxLayout(controls_content)
        controls_content_layout.setContentsMargins(6, 6, 6, 6)
        controls = QFormLayout()
        # Allow long labels and controls to wrap instead of being clipped by the
        # preview panel when the window is narrower than the preferred size.
        controls.setRowWrapPolicy(QFormLayout.WrapLongRows)
        controls.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        controls.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        controls.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image_width_spin = QSpinBox()
        self.image_width_spin.setRange(32, 4096)
        self.image_width_spin.setSingleStep(32)
        self.image_width_spin.setValue(256)
        self.image_height_spin = QSpinBox()
        self.image_height_spin.setRange(32, 4096)
        self.image_height_spin.setSingleStep(32)
        self.image_height_spin.setValue(256)
        self.zero_padding_check = QCheckBox("Zero-pad image for full convolution")
        self.zero_padding_check.setChecked(False)
        self.zero_padding_check.setToolTip(
            "When enabled, the source image is reduced and surrounded by a zero frame large enough for the selected PSF support. "
            "It is disabled by default. The synthetic test image already contains broad internal margins. "
            "Convolution routines still apply internal compatibility padding when image and PSF array sizes differ."
        )
        self.psf_type_combo = QComboBox()
        self.psf_type_combo.addItems(["Gaussian", "Motion horizontal", "Motion oblique", "High-frequency", "Lens incoherent"])
        self.psf_size_spin = QSpinBox(); self.psf_size_spin.setRange(3, 2047); self.psf_size_spin.setSingleStep(2); self.psf_size_spin.setValue(21)
        self.psf_sigma_spin = QDoubleSpinBox(); self.psf_sigma_spin.setRange(0.1, 50.0); self.psf_sigma_spin.setDecimals(3); self.psf_sigma_spin.setValue(3.0)
        self.motion_angle_spin = QDoubleSpinBox(); self.motion_angle_spin.setRange(-180.0, 180.0); self.motion_angle_spin.setDecimals(2); self.motion_angle_spin.setValue(35.0)
        self.hf_freq_spin = QDoubleSpinBox(); self.hf_freq_spin.setRange(0.1, 30.0); self.hf_freq_spin.setDecimals(2); self.hf_freq_spin.setValue(5.0)
        self.lens_f_spin = QDoubleSpinBox(); self.lens_f_spin.setRange(0.001, 10.0); self.lens_f_spin.setDecimals(4); self.lens_f_spin.setValue(0.05)
        self.lens_before_spin = QDoubleSpinBox(); self.lens_before_spin.setRange(0.001, 100.0); self.lens_before_spin.setDecimals(4); self.lens_before_spin.setValue(0.10)
        self.lens_after_spin = QDoubleSpinBox(); self.lens_after_spin.setRange(0.001, 100.0); self.lens_after_spin.setDecimals(4); self.lens_after_spin.setValue(0.10)
        self.wavelength_spin = QDoubleSpinBox(); self.wavelength_spin.setRange(100.0, 2000.0); self.wavelength_spin.setDecimals(1); self.wavelength_spin.setValue(550.0)
        self.aperture_spin = QDoubleSpinBox(); self.aperture_spin.setRange(0.1, 100.0); self.aperture_spin.setDecimals(2); self.aperture_spin.setValue(5.0)
        self.rot_symmetry_check = QCheckBox("Radially average generated PSF")
        self.rot_symmetry_check.setChecked(True)
        self.noise_type_combo = QComboBox(); self.noise_type_combo.addItems(["Gaussian", "Correlated Gaussian / speckle", "Poisson"])
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.5)
        self.noise_spin.setDecimals(4)
        self.noise_spin.setSingleStep(0.005)
        self.noise_spin.setValue(0.01)
        controls.addRow("Calculation image width X", self.image_width_spin)
        controls.addRow("Calculation image height Y", self.image_height_spin)
        controls.addRow("Visible zero padding", self.zero_padding_check)
        controls.addRow("PSF type", self.psf_type_combo)
        controls.addRow("PSF size", self.psf_size_spin)
        controls.addRow("Gaussian / HF sigma", self.psf_sigma_spin)
        controls.addRow("Motion angle [deg]", self.motion_angle_spin)
        controls.addRow("HF frequency", self.hf_freq_spin)
        controls.addRow("Lens focal length f [m]", self.lens_f_spin)
        controls.addRow("Distance before lens [m]", self.lens_before_spin)
        controls.addRow("Distance after lens [m]", self.lens_after_spin)
        controls.addRow("Wavelength [nm]", self.wavelength_spin)
        controls.addRow("Aperture diameter [mm]", self.aperture_spin)
        controls.addRow("Rotational symmetry", self.rot_symmetry_check)
        controls.addRow("Noise type", self.noise_type_combo)
        controls.addRow("Noise strength", self.noise_spin)
        controls_content_layout.addLayout(controls)
        padding_note = QLabel(
            "The generated PSF size describes the generated source array. The exact thresholded, cropped and unit-sum-normalized PSF used in calculations is selected in Tab 2 and is shown in this tab. "
            "The reconstruction input shown here is the current processed degraded/measured image. Reset thresholds / PSF selection in Tab 2 is the only action that restores the disk-loaded or generated source arrays."
        )
        padding_note.setWordWrap(True)
        controls_content_layout.addWidget(padding_note)
        controls_content_layout.addStretch(1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        controls_scroll.setWidget(controls_content)
        controls_content.setMinimumWidth(390)
        controls_scroll.setMinimumWidth(440)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(controls_scroll)
        left_panel.setMinimumWidth(440)
        left_panel.setMaximumWidth(680)

        right_panel = QWidget()
        right_panel.setMinimumWidth(460)
        views = QGridLayout(right_panel)
        views.setContentsMargins(4, 4, 4, 4)
        views.setSpacing(6)
        self.image_canvas = ImageCanvas("Reference image")
        self.psf_canvas = ImageCanvas("Calculation PSF")
        self.degraded_canvas = ImageCanvas("Calculation input")
        self.calculation_image_histogram = HistogramCanvas("Calculation-input histogram (256 bins)")
        self.calculation_psf_histogram = HistogramCanvas("Calculation-PSF histogram (256 bins)")
        self.calculation_info_label = QLabel("Calculation input: none; calculation PSF: none.")
        self.calculation_info_label.setWordWrap(True)
        for canvas in (self.image_canvas, self.psf_canvas, self.degraded_canvas):
            # Keep a useful preview size without forcing the right panel to overlap
            # or clip the controls on medium-resolution displays.
            canvas.setMinimumSize(220, 180)
        views.addWidget(self.image_canvas, 0, 0)
        views.addWidget(self.psf_canvas, 0, 1)
        views.addWidget(self.degraded_canvas, 1, 0, 1, 2)
        views.addWidget(self.calculation_image_histogram, 2, 0)
        views.addWidget(self.calculation_psf_histogram, 2, 1)
        views.addWidget(self.calculation_info_label, 3, 0, 1, 2)
        views.setColumnStretch(0, 1)
        views.setColumnStretch(1, 1)
        views.setRowStretch(0, 1)
        views.setRowStretch(1, 1)

        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.setHandleWidth(8)
        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(right_panel)
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([520, 900])
        layout.addWidget(content_splitter, 1)

        btn_load_img.clicked.connect(self.load_image)
        btn_synth_img.clicked.connect(self.generate_image)
        btn_load_psf.clicked.connect(self.load_psf_image)
        btn_psf.clicked.connect(self.generate_selected_psf)
        btn_degrade.clicked.connect(self.generate_degraded_input)
        btn_reset.clicked.connect(self.resetRequested.emit)
        btn_exit.clicked.connect(self.exitRequested.emit)
        self.zero_padding_check.toggled.connect(self._on_zero_padding_changed)
        self.image_width_spin.valueChanged.connect(self._on_resolution_link_changed)
        self.image_height_spin.valueChanged.connect(self._on_resolution_link_changed)
        self.psf_type_combo.currentIndexChanged.connect(lambda _=None: self.update_rotational_symmetry_availability(self.psf_type_combo.currentText()))
        self.update_rotational_symmetry_availability(self.psf_type_combo.currentText())

    def _set_calculation_shape(self, width: int, height: int) -> None:
        """Update X/Y calculation size without triggering repeated rebuilds."""
        width = max(self.image_width_spin.minimum(), min(self.image_width_spin.maximum(), int(width)))
        height = max(self.image_height_spin.minimum(), min(self.image_height_spin.maximum(), int(height)))
        old_w = self.image_width_spin.blockSignals(True)
        old_h = self.image_height_spin.blockSignals(True)
        self.image_width_spin.setValue(width)
        self.image_height_spin.setValue(height)
        self.image_width_spin.blockSignals(old_w)
        self.image_height_spin.blockSignals(old_h)
        self.state["calculation_image_shape"] = (height, width)

    @staticmethod
    def _center_pad_image_object(image: GrayImage, target_shape: Tuple[int, int]) -> GrayImage:
        """Center-pad an image to ``target_shape`` without resampling its pixels."""
        target_h, target_w = int(target_shape[0]), int(target_shape[1])
        arr = np.asarray(image.data, dtype=np.float64)
        if arr.shape == (target_h, target_w):
            return image
        if arr.shape[0] > target_h or arr.shape[1] > target_w:
            raise ValueError("The reconciliation canvas must not crop an image.")
        out = np.zeros((target_h, target_w), dtype=np.float64)
        y0 = (target_h - arr.shape[0]) // 2
        x0 = (target_w - arr.shape[1]) // 2
        out[y0:y0 + arr.shape[0], x0:x0 + arr.shape[1]] = arr
        metadata = dict(image.metadata or {})
        roi = metadata.get("content_roi", (0, arr.shape[0], 0, arr.shape[1]))
        try:
            ry0, ry1, rx0, rx1 = [int(v) for v in roi]
        except Exception:
            ry0, ry1, rx0, rx1 = 0, arr.shape[0], 0, arr.shape[1]
        metadata.update({
            "_preserve_intensity": True,
            "calculation_size": (target_h, target_w),
            "content_roi": (ry0 + y0, ry1 + y0, rx0 + x0, rx1 + x0),
            "shape_reconciled_by_zero_padding": True,
            "shape_before_reconciliation": tuple(int(v) for v in arr.shape),
        })
        return GrayImage(out, name=image.name, metadata=metadata)

    @staticmethod
    def _center_pad_psf_object(psf: PSF, target_shape: Tuple[int, int]) -> Tuple[PSF, Tuple[int, int]]:
        """Zero-pad a PSF so its selected centre maps to the canvas centre."""
        target_h, target_w = int(target_shape[0]), int(target_shape[1])
        arr = np.asarray(psf.kernel, dtype=np.float64)
        if arr.shape == (target_h, target_w):
            metadata = dict(psf.metadata or {})
            requested = metadata.get("calculation_center")
            if isinstance(requested, (tuple, list)) and len(requested) == 2:
                center = (int(requested[0]), int(requested[1]))
            else:
                center = PSF.support_center(arr)
            return psf, center
        if arr.shape[0] > target_h or arr.shape[1] > target_w:
            raise ValueError("The reconciliation canvas must not crop a PSF.")
        metadata = dict(psf.metadata or {})
        requested = metadata.get("calculation_center")
        if isinstance(requested, (tuple, list)) and len(requested) == 2:
            source_center = (
                int(np.clip(int(round(float(requested[0]))), 0, arr.shape[0] - 1)),
                int(np.clip(int(round(float(requested[1]))), 0, arr.shape[1] - 1)),
            )
        else:
            source_center = PSF.support_center(arr)
        target_center = (target_h // 2, target_w // 2)

        def embed(source: np.ndarray) -> np.ndarray:
            source = np.asarray(source, dtype=np.float64)
            canvas = np.zeros((target_h, target_w), dtype=np.float64)
            destination_top = target_center[0] - source_center[0]
            destination_left = target_center[1] - source_center[1]
            sy0 = max(0, -destination_top)
            sx0 = max(0, -destination_left)
            sy1 = min(source.shape[0], target_h - destination_top)
            sx1 = min(source.shape[1], target_w - destination_left)
            if sy1 > sy0 and sx1 > sx0:
                dy0 = destination_top + sy0
                dx0 = destination_left + sx0
                canvas[dy0:dy0 + sy1 - sy0, dx0:dx0 + sx1 - sx0] = source[sy0:sy1, sx0:sx1]
            return canvas

        padded = embed(arr)
        raw = np.asarray(psf.raw_kernel if psf.raw_kernel is not None else arr, dtype=np.float64)
        raw_padded = embed(raw) if raw.shape == arr.shape else padded.copy()
        metadata.update({
            "source_psf_shape_before_reconciliation": tuple(int(v) for v in arr.shape),
            "shape_reconciled_by_zero_padding": True,
            "calculation_center": tuple(int(v) for v in target_center),
        })
        return PSF(padded, name=psf.name, raw_kernel=raw_padded, metadata=metadata), target_center

    def _reconcile_image_and_psf_shapes(self, reason: str = "") -> bool:
        """Make the current image arrays and full PSF array share one canvas.

        Only zero-padding is used; neither the image nor the PSF is resampled or
        cropped.  The selected PSF centre is mapped to the geometric centre of
        the common canvas, while the selected calculation width and height are
        preserved.
        """
        psf = self.state.get("psf")
        images = [obj for obj in (self.state.get("image"), self.state.get("degraded")) if isinstance(obj, GrayImage)]
        if not isinstance(psf, PSF) or not images:
            return False
        current_h = int(self.image_height_spin.value())
        current_w = int(self.image_width_spin.value())
        target_h = max([current_h, int(psf.kernel.shape[0])] + [int(obj.data.shape[0]) for obj in images])
        target_w = max([current_w, int(psf.kernel.shape[1])] + [int(obj.data.shape[1]) for obj in images])
        if target_h > self.image_height_spin.maximum() or target_w > self.image_width_spin.maximum():
            raise ValueError(
                f"A common image/PSF canvas of {target_w} × {target_h} px exceeds the supported GUI limit."
            )
        changed = any(tuple(obj.data.shape) != (target_h, target_w) for obj in images) or tuple(psf.kernel.shape) != (target_h, target_w)
        if not changed:
            self._set_calculation_shape(target_w, target_h)
            return False

        self._set_calculation_shape(target_w, target_h)
        if isinstance(self.state.get("image"), GrayImage):
            self.state["image"] = self._center_pad_image_object(self.state["image"], (target_h, target_w))
        if isinstance(self.state.get("degraded"), GrayImage):
            self.state["degraded"] = self._center_pad_image_object(self.state["degraded"], (target_h, target_w))

        padded_psf, target_center = self._center_pad_psf_object(psf, (target_h, target_w))
        self.state["psf"] = padded_psf
        self.state["psf_calculation_center_x"] = int(target_center[1])
        self.state["psf_calculation_center_y"] = int(target_center[0])
        automatic = self.state.get("psf_automatic_selection")
        if isinstance(automatic, dict):
            automatic = dict(automatic)
            automatic["center"] = tuple(int(v) for v in target_center)
            self.state["psf_automatic_selection"] = automatic
        self.state["psf_selection_generation"] = int(self.state.get("psf_selection_generation", 0)) + 1
        self._clear_tab2_threshold_bases()
        self.state["last_shape_reconciliation"] = {
            "reason": str(reason),
            "shape": (target_h, target_w),
            "method": "centered zero padding",
        }
        return True

    def _ask_to_adopt_loaded_shape(
        self,
        proposed_shape: Tuple[int, int],
        description: str,
        details: str = "",
    ) -> bool:
        """Ask whether the calculation grid should follow loaded data dimensions."""
        h, w = int(proposed_shape[0]), int(proposed_shape[1])
        current = (int(self.image_height_spin.value()), int(self.image_width_spin.value()))
        if (h, w) == current:
            return False
        message = (
            f"The {description} suggests a calculation size of {w} × {h} pixels, "
            f"while the current setting is {current[1]} × {current[0]}."
        )
        if details:
            message += f"\n\n{details}"
        message += "\n\nChange the calculation image size to match the loaded data?"
        answer = QMessageBox.question(
            self,
            "Loaded data size differs",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self._set_calculation_shape(w, h)
            return True
        return False

    def _image_dialog_start_directory(self) -> str:
        """Return a valid start directory for image/PSF file dialogs."""
        try:
            candidate = Path(str(self.last_image_directory)).expanduser()
            if candidate.is_file():
                candidate = candidate.parent
            if candidate.is_dir():
                return str(candidate)
        except Exception:
            pass
        return str(Path.home())

    def _remember_loaded_image_path(self, path: str) -> None:
        """Remember the parent directory of the most recently selected image-like file."""
        if not path:
            return
        try:
            self.last_image_directory = str(Path(path).expanduser().resolve().parent)
        except Exception:
            self.last_image_directory = str(Path(path).expanduser().parent)

    def settings(self) -> Dict[str, Any]:
        return {
            "image_width": self.image_width_spin.value(),
            "image_height": self.image_height_spin.value(),
            "last_image_directory": self._image_dialog_start_directory(),
            "zero_padding_enabled": self.zero_padding_check.isChecked(),
            "psf_type": self.psf_type_combo.currentText(),
            "psf_size": self.psf_size_spin.value(),
            "psf_sigma": self.psf_sigma_spin.value(),
            "motion_angle": self.motion_angle_spin.value(),
            "hf_frequency": self.hf_freq_spin.value(),
            "lens_f": self.lens_f_spin.value(),
            "lens_before": self.lens_before_spin.value(),
            "lens_after": self.lens_after_spin.value(),
            "wavelength_nm": self.wavelength_spin.value(),
            "aperture_mm": self.aperture_spin.value(),
            "rotational_symmetry": self.rot_symmetry_check.isChecked(),
            "noise_type": self.noise_type_combo.currentText(),
            "noise_strength": self.noise_spin.value(),
        }

    def apply_settings(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        if "last_image_directory" in data:
            value = str(data.get("last_image_directory") or "").strip()
            if value:
                self.last_image_directory = value
        if "image_width" in data:
            self.image_width_spin.setValue(_safe_int(data.get("image_width"), self.image_width_spin.value()))
        if "image_height" in data:
            self.image_height_spin.setValue(_safe_int(data.get("image_height"), self.image_height_spin.value()))
        # Backward compatibility with profiles created before independent X/Y sizes.
        if "image_size" in data and "image_width" not in data and "image_height" not in data:
            old_size = _safe_int(data.get("image_size"), min(self.image_width_spin.value(), self.image_height_spin.value()))
            self.image_width_spin.setValue(old_size)
            self.image_height_spin.setValue(old_size)
        if "zero_padding_enabled" in data:
            self.zero_padding_check.setChecked(bool(data.get("zero_padding_enabled")))
        self.state["zero_padding_enabled"] = self.zero_padding_check.isChecked()
        if "psf_type" in data:
            idx = self.psf_type_combo.findText(str(data.get("psf_type")))
            if idx >= 0:
                self.psf_type_combo.setCurrentIndex(idx)
        for key, widget in [
            ("psf_size", self.psf_size_spin),
            ("psf_sigma", self.psf_sigma_spin),
            ("motion_angle", self.motion_angle_spin),
            ("hf_frequency", self.hf_freq_spin),
            ("lens_f", self.lens_f_spin),
            ("lens_before", self.lens_before_spin),
            ("lens_after", self.lens_after_spin),
            ("wavelength_nm", self.wavelength_spin),
            ("aperture_mm", self.aperture_spin),
            ("noise_strength", self.noise_spin),
        ]:
            if key in data:
                if isinstance(widget, QSpinBox):
                    widget.setValue(_safe_int(data.get(key), widget.value()))
                else:
                    widget.setValue(_safe_float(data.get(key), widget.value()))
        if "rotational_symmetry" in data and self.rot_symmetry_check.isEnabled():
            self.rot_symmetry_check.setChecked(bool(data.get("rotational_symmetry")))
        if "noise_type" in data:
            idx = self.noise_type_combo.findText(str(data.get("noise_type")))
            if idx >= 0:
                self.noise_type_combo.setCurrentIndex(idx)
        self.update_rotational_symmetry_availability(self.psf_type_combo.currentText())

    def recommended_psf_support(self) -> int:
        """Internal blind-PSF initial width; known-PSF support is selected in Tab 2."""
        return resolution_linked_psf_support(
            (int(self.image_height_spin.value()), int(self.image_width_spin.value())),
            fraction=0.45,
        )

    def _on_resolution_link_changed(self, *args: Any) -> None:
        self.state["calculation_image_shape"] = (int(self.image_height_spin.value()), int(self.image_width_spin.value()))
        self.current_psf_support_width(self.state.get("psf"))
        self._clear_tab2_threshold_bases()
        self.reframe_reference_for_current_psf()
        self.update_psf_preview()

    def _clear_tab2_threshold_bases(self) -> None:
        """Discard cached threshold-preview sources after loading/generating new data."""
        self.state.pop("_tab2_threshold_base_degraded", None)
        self.state.pop("_tab2_threshold_base_psf", None)
        self.state.pop("_tab2_threshold_base_degradation_psf", None)
        self.state.pop("_tab2_threshold_base_psf_selection", None)

    def _initialize_tab2_psf_selection(self, psf: Optional[PSF]) -> None:
        """Choose a conservative almost-nonzero rectangular PSF window for Tab 2."""
        if not isinstance(psf, PSF):
            return
        estimate = PSF.automatic_support_selection(psf.kernel, peak_fraction=1e-2)
        cy, cx = estimate.get("center", PSF.support_center(psf.kernel))
        height = int(estimate.get("height", estimate.get("width", psf.kernel.shape[0])))
        width = int(estimate.get("width", psf.kernel.shape[1]))
        height_cap = int(self.image_height_spin.value())
        width_cap = int(self.image_width_spin.value())
        height = _odd_at_most(min(max(1, height), psf.kernel.shape[0], height_cap))
        width = _odd_at_most(min(max(1, width), psf.kernel.shape[1], width_cap))
        self.state["psf_support_height"] = int(height)
        self.state["psf_support_width"] = int(width)
        self.state["psf_calculation_center_mode"] = "center_of_mass"
        self.state["psf_calculation_center_x"] = int(cx)
        self.state["psf_calculation_center_y"] = int(cy)
        self.state["psf_automatic_selection"] = dict(estimate)
        self.state["psf_selection_generation"] = int(self.state.get("psf_selection_generation", 0)) + 1
        psf.metadata = dict(psf.metadata or {})
        psf.metadata.update({
            "calculation_center_mode": "center_of_mass",
            "calculation_center": (int(cy), int(cx)),
            "calculation_support_height": int(height),
            "calculation_support_width": int(width),
            "automatic_support_selection": dict(estimate),
        })
        self.calculationPsfSupportChanged.emit(int(max(height, width)))

    def current_psf_support_width(self, psf: Optional[PSF] = None) -> int:
        """Return the maximum selected PSF extent for legacy size policies.

        The actual known-PSF crop is rectangular and stored as separate height
        and width values.  Older padding and algorithm-policy code needs one
        conservative scalar, so this method returns the larger dimension.
        """
        image_h = int(self.image_height_spin.value())
        image_w = int(self.image_width_spin.value())
        hard_cap = max_psf_support_for_image((image_h, image_w))
        if psf is not None:
            full_h = min(int(psf.kernel.shape[0]), image_h)
            full_w = min(int(psf.kernel.shape[1]), image_w)
        else:
            full_h = min(int(self.psf_size_spin.value()), image_h)
            full_w = min(int(self.psf_size_spin.value()), image_w)
        selected_h = max(1, int(self.state.get("psf_support_height", full_h)))
        selected_w = max(1, int(self.state.get("psf_support_width", full_w)))
        selected_h = min(selected_h, full_h, image_h)
        selected_w = min(selected_w, full_w, image_w)
        self.state["psf_support_height"] = int(selected_h)
        self.state["psf_support_width"] = int(selected_w)
        self.state["psf_support_hard_cap"] = (image_h, image_w)
        self.state["limit_psf_support"] = selected_h < full_h or selected_w < full_w
        extent = int(max(selected_h, selected_w))
        self._last_emitted_calculation_psf_support = extent
        self.calculationPsfSupportChanged.emit(extent)
        return extent

    def current_zero_padding(self, psf: Optional[PSF] = None) -> int:
        """Return the visible image-frame padding selected by the user.

        When visible zero padding is disabled, this returns zero. Algorithms may
        still zero-pad the smaller array internally so image and PSF dimensions
        are compatible for FFT or linear convolution.
        """
        self.state["zero_padding_enabled"] = self.zero_padding_check.isChecked()
        if not self.zero_padding_check.isChecked():
            return 0
        width = int(self.image_width_spin.value())
        height = int(self.image_height_spin.value())
        support = self.current_psf_support_width(psf)
        return min(
            max(0, support // 2),
            max(0, (width - 1) // 2),
            max(0, (height - 1) // 2),
        )

    def _pad_image_to_psf_if_required(self, image: GrayImage, psf: Optional[PSF]) -> GrayImage:
        """Center-pad an image only when disabled visible padding requires compatibility.

        With the visible zero frame disabled, the configured image content first
        fills the requested calculation size. If a supplied PSF array is larger,
        the image is then centered in the smallest common canvas. If the image is
        larger, convolution routines pad the PSF internally. No image intensity
        rescaling is performed by this compatibility step.
        """
        if self.zero_padding_check.isChecked() or psf is None:
            image.metadata["compatibility_padding"] = False
            return image
        ih, iw = image.data.shape
        kh, kw = psf.kernel.shape
        target_h = max(ih, kh)
        target_w = max(iw, kw)
        if target_h == ih and target_w == iw:
            image.metadata["compatibility_padding"] = (kh, kw) != (ih, iw)
            image.metadata["compatibility_target_shape"] = (target_h, target_w)
            return image
        if target_h > 2048 or target_w > 2048:
            raise ValueError(
                f"Compatibility canvas {target_h}x{target_w} exceeds the supported 2048x2048 limit. "
                "Reduce the loaded PSF array, reduce the calculation image size, or enable visible zero padding."
            )
        out = np.zeros((target_h, target_w), dtype=np.float64)
        y0 = (target_h - ih) // 2
        x0 = (target_w - iw) // 2
        out[y0:y0 + ih, x0:x0 + iw] = image.data
        meta = dict(image.metadata)
        old_roi = meta.get("content_roi", (0, ih, 0, iw))
        try:
            ry0, ry1, rx0, rx1 = [int(v) for v in old_roi]
        except Exception:
            ry0, ry1, rx0, rx1 = 0, ih, 0, iw
        meta.update({
            "_preserve_intensity": True,
            "calculation_size": (target_h, target_w),
            "content_roi": (ry0 + y0, ry1 + y0, rx0 + x0, rx1 + x0),
            "compatibility_padding": True,
            "compatibility_target_shape": (target_h, target_w),
            "zero_padding_enabled": False,
        })
        return GrayImage(out, name=image.name, metadata=meta)

    def _padding_title(self, prefix: str, padding: int) -> str:
        if self.zero_padding_check.isChecked():
            return f"{prefix}, zero frame {padding}px"
        return f"{prefix}, visible zero padding disabled"

    def reframe_reference_for_current_psf(self) -> None:
        """Rebuild loaded data after a PSF-support or padding-mode change."""
        padding = self.current_zero_padding(self.state.get("psf"))
        width = int(self.image_width_spin.value())
        height = int(self.image_height_spin.value())
        reference_available = self.state.get("reference_available") is not False
        image: Optional[GrayImage] = self.state.get("image")
        psf: Optional[PSF] = self.state.get("psf")
        degraded_before: Optional[GrayImage] = self.state.get("degraded")
        recreate_loaded_measured = bool(
            reference_available
            and degraded_before is not None
            and degraded_before.metadata.get("measured_input", False)
        )

        if reference_available and image is not None:
            source = image.metadata.get("source_array")
            if source is None:
                source = image.data
            rebuilt = GrayImage.from_array_with_zero_frame(source, width=width, height=height, padding=padding, name=image.name)
            rebuilt.metadata["zero_padding_enabled"] = self.zero_padding_check.isChecked()
            rebuilt = self._pad_image_to_psf_if_required(rebuilt, psf)
            self.state["image"] = rebuilt
            self.state.pop("degraded", None)
            self.state.pop("degradation_psf", None)
            self.image_canvas.show_image(rebuilt.data, self._padding_title("Reference image", padding))
            if recreate_loaded_measured:
                measured_meta = dict(rebuilt.metadata)
                measured_meta.update({"measured_input": True, "_preserve_intensity": True})
                self.state["degraded"] = GrayImage(
                    rebuilt.data.copy(),
                    name=rebuilt.name + "_measured_input",
                    metadata=measured_meta,
                )
                self.degraded_canvas.show_image(self.state["degraded"].data, "Loaded image as measured/degraded input")
            else:
                self.degraded_canvas.show_image(None, "Degraded input")
            return

        # Paired measured data have no reference image. Rebuild their visible
        # frame from the preserved original source without changing PSF values.
        measured: Optional[GrayImage] = self.state.get("degraded")
        if measured is not None and measured.metadata.get("measured_input", False):
            source = measured.metadata.get("source_array")
            if source is not None:
                old_meta = dict(measured.metadata)
                rebuilt = GrayImage.from_array_with_zero_frame(
                    source, width=width, height=height, padding=padding, name=measured.name
                )
                new_layout = dict(rebuilt.metadata)
                rebuilt.metadata.update(old_meta)
                rebuilt.metadata.update({
                    "calculation_size": new_layout.get("calculation_size", rebuilt.data.shape),
                    "zero_padding": int(padding),
                    "zero_padding_enabled": self.zero_padding_check.isChecked(),
                    "inner_size": new_layout.get("inner_size"),
                    "content_roi": new_layout.get("content_roi"),
                    "source_array": source,
                    "measured_input": True,
                })
                rebuilt = self._pad_image_to_psf_if_required(rebuilt, self.state.get("psf"))
                self.state["degraded"] = rebuilt
                self.degraded_canvas.show_image(rebuilt.data, self._padding_title("Measured/degraded input", padding))

    def _on_zero_padding_changed(self, checked: bool) -> None:
        """Apply the visible-padding mode to already loaded/generated data."""
        self.state["zero_padding_enabled"] = bool(checked)
        self._clear_tab2_threshold_bases()
        self.reframe_reference_for_current_psf()
        self.update_psf_preview()

    def update_rotational_symmetry_availability(self, psf_type: str) -> None:
        """Disable radial averaging for motion PSF because it destroys motion directionality."""
        is_motion = psf_type.startswith("Motion")
        self.rot_symmetry_check.setEnabled(not is_motion)
        if is_motion:
            self.rot_symmetry_check.setChecked(False)
        elif not self.rot_symmetry_check.isChecked():
            self.rot_symmetry_check.setChecked(True)

    def load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load image", self._image_dialog_start_directory(),
            "Monochrome images and MAT (*.png *.tif *.tiff *.bmp *.jpg *.jpeg *.mat);;All files (*)"
        )
        if not path:
            return
        self._remember_loaded_image_path(path)
        try:
            self._clear_tab2_threshold_bases()
            mat_key = _choose_mat_variable(
                self, path, ("degraded", "measured", "image", "result", "reference"), "image"
            )
            if mat_key == "":
                return
            raw = load_monochrome_array(
                path, mat_key=mat_key, preferred_mat_keys=("degraded", "measured", "image", "result", "reference")
            )
            self._ask_to_adopt_loaded_shape(raw.shape, "loaded image")
            padding = self.current_zero_padding(self.state.get("psf"))
            self.state["image"] = GrayImage.from_array_with_zero_frame(
                raw,
                width=self.image_width_spin.value(),
                height=self.image_height_spin.value(),
                padding=padding,
                name=path,
            )
            self.state["image"].metadata["zero_padding_enabled"] = self.zero_padding_check.isChecked()
            self.state["image"] = self._pad_image_to_psf_if_required(self.state["image"], self.state.get("psf"))
            self.state["reference_available"] = True
            self.state["measured_pair_loaded"] = False
            self.image_canvas.setVisible(True)
            measured_meta = dict(self.state["image"].metadata)
            measured_meta.update({"measured_input": True, "_preserve_intensity": True})
            self.state["degraded"] = GrayImage(
                self.state["image"].data.copy(),
                name=self.state["image"].name + "_measured_input",
                metadata=measured_meta,
            )
            self.state.pop("degradation_psf", None)
            self.state.pop("result", None)
            self.state.pop("estimated_psf", None)
            self._reconcile_image_and_psf_shapes("image loaded")
            self.image_canvas.show_image(self.state["image"].data, self._padding_title("Reference image", padding))
            self.degraded_canvas.show_image(self.state["degraded"].data, "Loaded image as measured/degraded input")
            self.update_psf_preview()
        except Exception as exc:
            QMessageBox.warning(self, "Image load error", f"Could not load image:\n{exc}")

    def generate_image(self) -> None:
        self._clear_tab2_threshold_bases()
        padding = self.current_zero_padding(self.state.get("psf"))
        self.state["image"] = GrayImage.synthetic(
            width=self.image_width_spin.value(),
            height=self.image_height_spin.value(),
            padding=padding,
        )
        self.state["image"].metadata["zero_padding_enabled"] = self.zero_padding_check.isChecked()
        self.state["image"] = self._pad_image_to_psf_if_required(self.state["image"], self.state.get("psf"))
        self.state["reference_available"] = True
        self.state["measured_pair_loaded"] = False
        self.image_canvas.setVisible(True)
        self.state.pop("degraded", None)
        self.state.pop("degradation_psf", None)
        self._reconcile_image_and_psf_shapes("test image generated")
        self.image_canvas.show_image(self.state["image"].data, self._padding_title("Synthetic reference image", padding))
        self.degraded_canvas.show_image(None, "Degraded input")
        self.update_psf_preview()

    def load_psf_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load PSF", self._image_dialog_start_directory(),
            "Monochrome images and MAT (*.png *.tif *.tiff *.bmp *.jpg *.jpeg *.mat);;All files (*)"
        )
        if not path:
            return
        self._remember_loaded_image_path(path)
        try:
            self._clear_tab2_threshold_bases()
            mat_key = _choose_mat_variable(
                self, path, ("psf", "psf_kernel", "current_psf", "degradation_psf", "estimated_psf"), "PSF"
            )
            if mat_key == "":
                return
            raw = load_monochrome_array(
                path, mat_key=mat_key, preferred_mat_keys=("psf", "psf_kernel", "current_psf", "degradation_psf", "estimated_psf")
            )
            self.state["psf"] = PSF(raw, name=path, raw_kernel=raw)
            self._reconcile_image_and_psf_shapes("PSF loaded")
            self._initialize_tab2_psf_selection(self.state["psf"])
            self.current_psf_support_width(self.state["psf"])
            self.reframe_reference_for_current_psf()
            self._reconcile_image_and_psf_shapes("PSF loaded after image reframing")
        except Exception as exc:
            QMessageBox.warning(self, "PSF load error", f"Could not load PSF image:\n{exc}")
            return
        self.current_psf_support_width(self.state["psf"])
        self.reframe_reference_for_current_psf()
        self.update_psf_preview()

    def generate_selected_psf(self) -> None:
        self._clear_tab2_threshold_bases()
        psf_type = self.psf_type_combo.currentText()
        size = self.psf_size_spin.value()
        if psf_type == "Gaussian":
            self.state["psf"] = PSF.gaussian(size=size, sigma=self.psf_sigma_spin.value())
        elif psf_type == "Motion horizontal":
            self.state["psf"] = PSF.motion(size=size, angle_deg=0.0)
        elif psf_type == "Motion oblique":
            self.state["psf"] = PSF.motion(size=size, angle_deg=self.motion_angle_spin.value())
        elif psf_type == "High-frequency":
            self.state["psf"] = PSF.high_frequency(size=size, frequency=self.hf_freq_spin.value(), sigma=self.psf_sigma_spin.value())
        elif psf_type == "Lens incoherent":
            # The generated array size is controlled only by PSF size. The
            # calculation crop is selected later in Tab 2.
            calculation_min_side = min(self.image_width_spin.value(), self.image_height_spin.value())
            lens_size = min(calculation_min_side, max(int(size), 33))
            if lens_size % 2 == 0:
                lens_size -= 1
            self.state["psf"] = PSF.lens_incoherent(
                size=max(lens_size, 33),
                focal_length=self.lens_f_spin.value(),
                distance_before=self.lens_before_spin.value(),
                distance_after=self.lens_after_spin.value(),
                wavelength=self.wavelength_spin.value() * 1e-9,
                aperture_diameter=self.aperture_spin.value() * 1e-3,
                diffraction_grid_size=max(
                    2 * max(self.image_width_spin.value(), self.image_height_spin.value()),
                    2 * size,
                ),
            )
        if self.rot_symmetry_check.isChecked() and not psf_type.startswith("Motion"):
            old_psf = self.state["psf"]
            self.state["psf"] = PSF(
                PSF.rotational_average(old_psf.kernel),
                name=old_psf.name + "_rotational",
                raw_kernel=old_psf.raw_kernel,
            )
        self._reconcile_image_and_psf_shapes("PSF generated")
        self._initialize_tab2_psf_selection(self.state["psf"])
        self.current_psf_support_width(self.state["psf"])
        self.reframe_reference_for_current_psf()
        self._reconcile_image_and_psf_shapes("PSF generated after image reframing")
        self.update_psf_preview()

    def refresh_calculation_views(self) -> None:
        """Show exactly the image and PSF that numerical routines will use."""
        reference: Optional[GrayImage] = self.state.get("image")
        degraded = _calculation_image_from_state(self.state)
        image_shape = degraded.data.shape if degraded is not None else (reference.data.shape if isinstance(reference, GrayImage) else None)
        calculation_psf = _synchronize_calculation_psf(self.state, image_shape) if image_shape is not None else None
        self.image_canvas.setVisible(reference is not None and reference_metrics_available(self.state))
        self.image_canvas.show_image(
            reference.data if reference is not None and reference_metrics_available(self.state) else None,
            "Reference image (metrics only; not reconstruction input)",
        )
        self.degraded_canvas.show_image(
            degraded.data if degraded is not None else None,
            f"Calculation input {degraded.data.shape[1]}×{degraded.data.shape[0]} px (after thresholding)" if degraded is not None else "Calculation input",
        )
        self.psf_canvas.show_image(
            calculation_psf.kernel if calculation_psf is not None else None,
            (f"Calculation PSF {calculation_psf.kernel.shape[1]}×{calculation_psf.kernel.shape[0]} px; "
             f"sum={float(calculation_psf.kernel.sum()):.8g} (thresholded, cropped, normalized)")
            if calculation_psf is not None else "Calculation PSF",
            normalize_display=True,
        )
        self.calculation_image_histogram.show_histogram(
            degraded.data if degraded is not None else None,
            "Calculation-input histogram (256 bins; after thresholding)",
        )
        self.calculation_psf_histogram.show_histogram(
            calculation_psf.kernel if calculation_psf is not None else None,
            "Calculation-PSF histogram (256 bins; cropped and normalized)",
            relative_to_peak=True,
        )
        self.calculation_info_label.setText(_calculation_data_summary(self.state))

    def update_psf_preview(self) -> None:
        self.refresh_calculation_views()

    def generate_degraded_input(self) -> None:
        image: Optional[GrayImage] = self.state.get("image")
        psf: Optional[PSF] = self.state.get("psf")
        if image is None or psf is None:
            QMessageBox.warning(self, "Missing data", "Load/generate a reference image and PSF first.")
            return
        self._reconcile_image_and_psf_shapes("before degraded-input generation")
        image = self.state.get("image")
        psf = self.state.get("psf")
        run_psf = _synchronize_calculation_psf(self.state, image.data.shape)
        if run_psf is None:
            QMessageBox.warning(self, "Missing PSF", "No valid calculation PSF is available.")
            return
        self.state["degradation_psf"] = PSF(run_psf.kernel.copy(), name=run_psf.name + "_forward", metadata=dict(run_psf.metadata or {}))
        self.state["degraded"] = degrade_image(image, run_psf, self.noise_spin.value(), self.noise_type_combo.currentText())
        self.refresh_calculation_views()


class DegradedInputTab(QWidget):
    """Preview, threshold and regenerate the measured/degraded input and PSF."""

    calculationPsfSupportChanged = pyqtSignal(int)
    calculationDataChanged = pyqtSignal()
    wienerKOptimized = pyqtSignal(float)

    def __init__(self, app_state: Dict[str, Any]) -> None:
        super().__init__()
        self.state = app_state
        self._last_psf_object_id: Optional[int] = None
        self._last_psf_selection_generation = -1
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems(["Gaussian", "Correlated Gaussian / speckle", "Poisson"])
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.5)
        self.noise_spin.setDecimals(4)
        self.noise_spin.setSingleStep(0.005)
        self.noise_spin.setValue(0.01)
        btn_generate = QPushButton("Regenerate degraded input")
        controls.addWidget(QLabel("Noise type"))
        controls.addWidget(self.noise_type_combo)
        controls.addWidget(QLabel("Noise strength"))
        controls.addWidget(self.noise_spin)
        controls.addWidget(btn_generate)
        controls.addStretch(1)
        layout.addLayout(controls)

        threshold_group = QGroupBox("Interactive thresholds and calculation PSF selection")
        threshold_form = QFormLayout(threshold_group)
        self.image_histogram = HistogramCanvas("Measured-image histogram (256 bins)")
        self.psf_histogram = HistogramCanvas("PSF histogram (256 bins; intensity / peak)")
        self.image_floor_spin = QDoubleSpinBox()
        self.image_floor_spin.setRange(0.0, 1.0)
        self.image_floor_spin.setDecimals(5)
        self.image_floor_spin.setSingleStep(0.001)
        self.image_floor_spin.setValue(0.0)
        self.image_floor_spin.setToolTip("Values at or below this floor are set to zero; the remaining range is linearly rescaled to [0, original maximum].")
        self.image_floor_slider = QSlider()
        self.image_floor_slider.setOrientation(1)
        self.image_floor_slider.setRange(0, 10000)
        self.image_floor_slider.setSingleStep(1)
        self.image_floor_slider.setPageStep(100)
        self.image_floor_slider.setValue(0)
        self.auto_image_floor_button = QPushButton("Auto from border")
        self.auto_image_floor_button.setToolTip(
            "Set the image floor to the mean intensity of perimeter pixels in the original, non-padded image region."
        )
        # Histogram and slider share column 0 of the same grid.  Consequently
        # their widget widths remain identical when the window is resized.
        image_floor_grid = QGridLayout()
        image_floor_grid.setContentsMargins(0, 0, 0, 0)
        image_floor_grid.setHorizontalSpacing(6)
        image_floor_grid.setVerticalSpacing(1)
        image_floor_grid.addWidget(self.image_histogram, 0, 0)
        image_floor_grid.addWidget(self.image_floor_slider, 1, 0)
        image_floor_grid.addWidget(self.image_floor_spin, 1, 1)
        image_floor_grid.addWidget(self.auto_image_floor_button, 1, 2)
        image_floor_grid.setColumnStretch(0, 1)
        image_floor_widget = QWidget()
        image_floor_widget.setLayout(image_floor_grid)

        self.psf_floor_spin = QDoubleSpinBox()
        self.psf_floor_spin.setRange(0.0, 1.0)
        self.psf_floor_spin.setDecimals(5)
        self.psf_floor_spin.setSingleStep(0.0005)
        self.psf_floor_spin.setValue(0.0)
        self.psf_floor_spin.setToolTip("PSF values at or below this fraction of the peak are set to zero; the remaining range is rescaled to [0, original peak], then the PSF sum is normalized to 1.")
        self.psf_floor_slider = QSlider()
        self.psf_floor_slider.setOrientation(1)
        self.psf_floor_slider.setRange(0, 10000)
        self.psf_floor_slider.setSingleStep(1)
        self.psf_floor_slider.setPageStep(50)
        self.psf_floor_slider.setValue(0)
        self.auto_psf_floor_button = QPushButton("Auto from border")
        self.auto_psf_floor_button.setToolTip(
            "Set the PSF floor/peak ratio from the mean perimeter value of the original PSF image."
        )
        self.optimize_psf_floor_k_button = QPushButton("Optimize PSF floor + Wiener K")
        self.optimize_psf_floor_k_button.setToolTip(
            "Jointly optimize the PSF floor and Wiener regularization K. With an independent reference, "
            "the reconstruction MSE is minimized. For measured data without a reference, the floor is restricted "
            "by robust background statistics of the selected PSF frame, while Wiener GCV selects K only for each "
            "fixed candidate PSF. Nearly impulsive collapsed candidates are rejected."
        )
        psf_floor_grid = QGridLayout()
        psf_floor_grid.setContentsMargins(0, 0, 0, 0)
        psf_floor_grid.setHorizontalSpacing(6)
        psf_floor_grid.setVerticalSpacing(1)
        psf_floor_grid.addWidget(self.psf_histogram, 0, 0)
        psf_floor_grid.addWidget(self.psf_floor_slider, 1, 0)
        psf_floor_grid.addWidget(self.psf_floor_spin, 1, 1)
        psf_floor_grid.addWidget(self.auto_psf_floor_button, 1, 2)
        psf_floor_grid.addWidget(self.optimize_psf_floor_k_button, 1, 3)
        self.psf_floor_k_result_label = QLabel("Joint optimization result: -")
        self.psf_floor_k_result_label.setWordWrap(True)
        psf_floor_grid.addWidget(self.psf_floor_k_result_label, 2, 0, 1, 4)
        psf_floor_grid.setColumnStretch(0, 1)
        psf_floor_widget = QWidget()
        psf_floor_widget.setLayout(psf_floor_grid)

        self.psf_preview_mode_combo = QComboBox()
        self.psf_preview_mode_combo.addItems(["Selected calculation part", "Full PSF array"])
        self.psf_preview_mode_combo.setToolTip("Choose whether the PSF image below shows the complete loaded array or the part selected for calculations.")

        self.psf_calc_width_spin = QSpinBox()
        self.psf_calc_width_spin.setRange(1, 4096)
        self.psf_calc_width_spin.setSingleStep(1)
        self.psf_calc_width_spin.setValue(int(self.state.get("psf_support_width", 65)))
        self.psf_calc_width_spin.setToolTip("Horizontal size, in pixels, of the PSF part used by convolution and reconstruction. Even and odd sizes are supported.")
        self.psf_calc_height_spin = QSpinBox()
        self.psf_calc_height_spin.setRange(1, 4096)
        self.psf_calc_height_spin.setSingleStep(1)
        self.psf_calc_height_spin.setValue(int(self.state.get("psf_support_height", self.state.get("psf_support_width", 65))))
        self.psf_calc_height_spin.setToolTip("Vertical size, in pixels, of the PSF part used by convolution and reconstruction. Even and odd sizes are supported.")
        # Compatibility alias for older external scripts that accessed the
        # former square-size widget directly.
        self.psf_calc_size_spin = self.psf_calc_width_spin

        self.psf_center_mode_combo = QComboBox()
        self.psf_center_mode_combo.addItem("Center of mass", "center_of_mass")
        self.psf_center_mode_combo.addItem("Geometric center", "geometric")
        self.psf_center_mode_combo.addItem("Manual", "manual")
        initial_center_mode = str(self.state.get("psf_calculation_center_mode", "center_of_mass"))
        initial_center_index = max(0, self.psf_center_mode_combo.findData(initial_center_mode))
        self.psf_center_mode_combo.setCurrentIndex(initial_center_index)
        self.psf_center_mode_combo.setToolTip(
            "Choose the calculation-window centre. Drag the red frame in the full-PSF preview to switch to Manual."
        )

        self.psf_manual_x_spin = QSpinBox()
        self.psf_manual_x_spin.setRange(0, 4095)
        self.psf_manual_x_spin.setValue(int(self.state.get("psf_calculation_center_x", 0)))
        self.psf_manual_x_spin.setToolTip("Horizontal pixel coordinate of the manual PSF calculation-window centre.")
        self.psf_manual_y_spin = QSpinBox()
        self.psf_manual_y_spin.setRange(0, 4095)
        self.psf_manual_y_spin.setValue(int(self.state.get("psf_calculation_center_y", 0)))
        self.psf_manual_y_spin.setToolTip("Vertical pixel coordinate of the manual PSF calculation-window centre.")

        psf_selection_row = QHBoxLayout()
        psf_selection_row.addWidget(QLabel("Preview"))
        psf_selection_row.addWidget(self.psf_preview_mode_combo)
        psf_selection_row.addWidget(QLabel("Width [px]"))
        psf_selection_row.addWidget(self.psf_calc_width_spin)
        psf_selection_row.addWidget(QLabel("Height [px]"))
        psf_selection_row.addWidget(self.psf_calc_height_spin)
        psf_selection_row.addWidget(QLabel("Center"))
        psf_selection_row.addWidget(self.psf_center_mode_combo)
        psf_selection_row.addWidget(QLabel("x"))
        psf_selection_row.addWidget(self.psf_manual_x_spin)
        psf_selection_row.addWidget(QLabel("y"))
        psf_selection_row.addWidget(self.psf_manual_y_spin)
        self.reset_psf_frame_button = QPushButton("Reset frame to full PSF")
        self.reset_psf_frame_button.setToolTip(
            "Set the red calculation frame to the complete width and height of the full PSF array."
        )
        psf_selection_row.addWidget(self.reset_psf_frame_button)
        psf_selection_row.addStretch(1)
        psf_selection_widget = QWidget()
        psf_selection_widget.setLayout(psf_selection_row)

        btn_apply_threshold = QPushButton("Apply thresholds / PSF selection now")
        btn_reset_threshold = QPushButton("Reset thresholds / PSF selection")
        threshold_buttons = QHBoxLayout()
        threshold_buttons.addWidget(btn_apply_threshold)
        threshold_buttons.addWidget(btn_reset_threshold)
        threshold_buttons.addStretch(1)
        threshold_buttons_widget = QWidget()
        threshold_buttons_widget.setLayout(threshold_buttons)

        threshold_form.addRow("Measured image floor [0–1]", image_floor_widget)
        threshold_form.addRow("PSF floor / peak [0–1]", psf_floor_widget)
        threshold_form.addRow("PSF calculation part", psf_selection_widget)
        threshold_form.addRow("Actions", threshold_buttons_widget)
        layout.addWidget(threshold_group)

        # Controls below are pending edits. They update only the red frame and
        # threshold markers; calculation data, image previews and histogram
        # contents change only after the explicit Apply action.
        self._threshold_syncing = False
        self._threshold_preview_timer = QTimer(self)
        self._threshold_preview_timer.setSingleShot(True)
        self._threshold_preview_timer.setInterval(80)
        self._threshold_preview_timer.timeout.connect(self._refresh_pending_threshold_preview)

        self.threshold_status_label = QLabel(
            "Thresholding sets values at or below the selected floor to zero and linearly expands the remaining "
            "range to [0, original maximum]. Border Auto uses the mean perimeter level of the original non-padded "
            "image or raw PSF. Joint Auto searches the PSF floor together with Wiener K. PSF thresholding and every "
            "subsequent calculation crop are followed by unit-sum normalization. "
            "In the full-PSF preview, drag inside the red frame to move it, drag near an edge to resize it, "
            "or use the mouse wheel to shrink/enlarge it. Controls are pending until Apply is pressed; "
            "histograms and calculation data remain unchanged while editing. Reset is the only action that "
            "restores the disk-loaded or generated source image and PSF."
        )
        self.threshold_status_label.setWordWrap(True)
        layout.addWidget(self.threshold_status_label)

        views = QHBoxLayout()
        self.reference_canvas = ImageCanvas("Reference image")
        self.psf_canvas = ImageCanvas("PSF")
        self.degraded_canvas = ImageCanvas("Degraded input")
        views.addWidget(self.reference_canvas)
        views.addWidget(self.psf_canvas)
        views.addWidget(self.degraded_canvas)
        layout.addLayout(views)

        self.metrics_label = QLabel("Metrics in original (non-padded) region: PSNR: -    SSIM: -    TV: -")
        layout.addWidget(self.metrics_label)
        self.calculation_info_label = QLabel("Calculation input: none; calculation PSF: none.")
        self.calculation_info_label.setWordWrap(True)
        layout.addWidget(self.calculation_info_label)
        layout.addWidget(QLabel(
            "Selected calculation-part preview is the exact thresholded, cropped and unit-sum-normalized PSF used by every algorithm. "
            "Full PSF array is a source-array editing view; samples outside the red frame are not used."
        ))
        layout.addStretch(1)

        btn_generate.clicked.connect(self.generate_degraded_input)
        btn_apply_threshold.clicked.connect(self.apply_lower_thresholds)
        btn_reset_threshold.clicked.connect(self.reset_lower_thresholds)
        self.auto_image_floor_button.clicked.connect(self.auto_image_floor_from_border)
        self.auto_psf_floor_button.clicked.connect(self.auto_psf_floor_from_border)
        self.optimize_psf_floor_k_button.clicked.connect(self.optimize_psf_floor_and_wiener_k)
        self.image_floor_slider.valueChanged.connect(self._image_floor_slider_changed)
        self.image_floor_spin.valueChanged.connect(self._image_floor_spin_changed)
        self.psf_floor_slider.valueChanged.connect(self._psf_floor_slider_changed)
        self.psf_floor_spin.valueChanged.connect(self._psf_floor_spin_changed)
        self.psf_preview_mode_combo.currentIndexChanged.connect(self.refresh)
        self.psf_calc_width_spin.valueChanged.connect(self._psf_selection_control_changed)
        self.psf_calc_height_spin.valueChanged.connect(self._psf_selection_control_changed)
        self.psf_center_mode_combo.currentIndexChanged.connect(self._psf_center_mode_changed)
        self.psf_manual_x_spin.valueChanged.connect(self._psf_selection_control_changed)
        self.psf_manual_y_spin.valueChanged.connect(self._psf_selection_control_changed)
        self.reset_psf_frame_button.clicked.connect(self.reset_psf_frame_to_full)
        self._update_manual_center_controls()

    def _schedule_threshold_preview(self) -> None:
        if self._threshold_syncing:
            return
        self.threshold_status_label.setText(
            "Pending threshold/PSF-frame changes. Press Apply thresholds / PSF selection now to update "
            "the calculation image, calculation PSF, previews and histogram contents."
        )
        self._threshold_preview_timer.start()

    def _refresh_pending_threshold_preview(self) -> None:
        """Refresh only pending controls/overlay without changing calculation data."""
        self.refresh()

    def _image_floor_slider_changed(self, value: int) -> None:
        if self._threshold_syncing:
            return
        self._threshold_syncing = True
        try:
            self.image_floor_spin.setValue(float(value) / 10000.0)
        finally:
            self._threshold_syncing = False
        self._schedule_threshold_preview()

    def _image_floor_spin_changed(self, value: float) -> None:
        if self._threshold_syncing:
            return
        self._threshold_syncing = True
        try:
            self.image_floor_slider.setValue(int(round(float(value) * 10000.0)))
        finally:
            self._threshold_syncing = False
        self._schedule_threshold_preview()

    def _psf_floor_slider_changed(self, value: int) -> None:
        if self._threshold_syncing:
            return
        self._threshold_syncing = True
        try:
            self.psf_floor_spin.setValue(float(value) / 10000.0)
        finally:
            self._threshold_syncing = False
        self._schedule_threshold_preview()

    def _psf_floor_spin_changed(self, value: float) -> None:
        if self._threshold_syncing:
            return
        self._threshold_syncing = True
        try:
            self.psf_floor_slider.setValue(int(round(float(value) * 10000.0)))
        finally:
            self._threshold_syncing = False
        self._schedule_threshold_preview()

    def _psf_center_mode_key(self) -> str:
        value = self.psf_center_mode_combo.currentData()
        return str(value or "center_of_mass")

    def _update_manual_center_controls(self) -> None:
        manual = self._psf_center_mode_key() == "manual"
        self.psf_manual_x_spin.setEnabled(manual)
        self.psf_manual_y_spin.setEnabled(manual)

    def _psf_center_mode_changed(self, *args: Any) -> None:
        mode = self._psf_center_mode_key()
        psf: Optional[PSF] = self.state.get("psf")
        if mode == "manual" and psf is not None:
            # Initialise manual coordinates from the currently active automatic
            # centre unless a manual position has already been committed.
            has_manual = "psf_calculation_center_x" in self.state and "psf_calculation_center_y" in self.state
            if not has_manual:
                cy, cx = PSF.support_center(psf.kernel)
                self.psf_manual_x_spin.blockSignals(True)
                self.psf_manual_y_spin.blockSignals(True)
                self.psf_manual_x_spin.setValue(int(cx))
                self.psf_manual_y_spin.setValue(int(cy))
                self.psf_manual_x_spin.blockSignals(False)
                self.psf_manual_y_spin.blockSignals(False)
            # The full array is the meaningful view for mouse editing.
            full_idx = self.psf_preview_mode_combo.findText("Full PSF array")
            if full_idx >= 0:
                self.psf_preview_mode_combo.setCurrentIndex(full_idx)
        self._update_manual_center_controls()
        self._psf_selection_control_changed()

    def _psf_selection_control_changed(self, *args: Any) -> None:
        if self.isVisible() and self.state.get("psf") is not None:
            self._ensure_threshold_bases()
        # Width, height and centre remain pending GUI values until Apply.  In
        # particular, blind algorithms continue to see the last committed Tab-2
        # support and histograms continue to describe committed calculation data.
        self._schedule_threshold_preview()
        self.refresh()

    def reset_psf_frame_to_full(self) -> None:
        """Set the red calculation rectangle to the entire full PSF array."""
        psf: Optional[PSF] = self.state.get("psf")
        if psf is None:
            self.threshold_status_label.setText("Load or generate a PSF before resetting its calculation frame.")
            return
        self._ensure_threshold_bases()
        height, width = int(psf.kernel.shape[0]), int(psf.kernel.shape[1])
        cy, cx = height // 2, width // 2
        widgets = (
            self.psf_calc_width_spin, self.psf_calc_height_spin,
            self.psf_center_mode_combo, self.psf_manual_x_spin, self.psf_manual_y_spin,
            self.psf_preview_mode_combo,
        )
        blocked = [widget.blockSignals(True) for widget in widgets]
        try:
            self.psf_calc_width_spin.setValue(width)
            self.psf_calc_height_spin.setValue(height)
            geometric_index = self.psf_center_mode_combo.findData("geometric")
            if geometric_index >= 0:
                self.psf_center_mode_combo.setCurrentIndex(geometric_index)
            self.psf_manual_x_spin.setValue(cx)
            self.psf_manual_y_spin.setValue(cy)
            full_index = self.psf_preview_mode_combo.findText("Full PSF array")
            if full_index >= 0:
                self.psf_preview_mode_combo.setCurrentIndex(full_index)
        finally:
            for widget, old_state in zip(widgets, blocked):
                widget.blockSignals(old_state)
        self._update_manual_center_controls()
        self.threshold_status_label.setText(
            f"Pending PSF frame reset to the full array: {width}x{height} px. Press Apply to use it."
        )
        self.refresh()

    def _selected_psf_center(self, array: np.ndarray) -> Tuple[int, int]:
        arr = np.asarray(array, dtype=np.float64)
        mode = self._psf_center_mode_key()
        if mode == "geometric":
            return arr.shape[0] // 2, arr.shape[1] // 2
        if mode == "manual":
            cy = int(np.clip(self.psf_manual_y_spin.value(), 0, max(0, arr.shape[0] - 1)))
            cx = int(np.clip(self.psf_manual_x_spin.value(), 0, max(0, arr.shape[1] - 1)))
            return cy, cx
        automatic = self.state.get("psf_automatic_selection")
        if isinstance(automatic, dict):
            center = automatic.get("center")
            if isinstance(center, (tuple, list)) and len(center) == 2:
                cy = int(np.clip(int(round(float(center[0]))), 0, max(0, arr.shape[0] - 1)))
                cx = int(np.clip(int(round(float(center[1]))), 0, max(0, arr.shape[1] - 1)))
                return cy, cx
        return PSF.support_center(arr)

    def _selected_psf_shape(self, array: np.ndarray) -> Tuple[int, int]:
        arr = np.asarray(array)
        height = min(max(1, int(self.psf_calc_height_spin.value())), max(1, arr.shape[0]))
        width = min(max(1, int(self.psf_calc_width_spin.value())), max(1, arr.shape[1]))
        return max(1, height), max(1, width)

    def _selected_psf_array(self, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float64)
        height, width = self._selected_psf_shape(arr)
        selected = PSF.centered_window(arr, self._selected_psf_center(arr), height, width)
        selected = np.maximum(np.nan_to_num(selected), 0.0)
        total = float(np.sum(selected))
        if total <= 1e-18:
            out = np.zeros_like(selected, dtype=np.float64)
            out[selected.shape[0] // 2, selected.shape[1] // 2] = 1.0
            return out
        return selected / total

    def _selected_psf_rectangle(self, array: np.ndarray) -> Tuple[int, int, int, int]:
        """Return the rectangular calculation window in full-PSF coordinates."""
        arr = np.asarray(array)
        height, width = self._selected_psf_shape(arr)
        cy, cx = self._selected_psf_center(arr)
        return int(cx - width // 2), int(cy - height // 2), int(width), int(height)

    def _psf_rectangle_edited(self, geometry: Tuple[int, int, int, int]) -> None:
        """Commit a mouse-moved/resized rectangular preview frame."""
        psf: Optional[PSF] = self.state.get("psf")
        if psf is None:
            return
        self._ensure_threshold_bases()
        x0, y0, width, height = [int(v) for v in geometry]
        width = min(max(1, width), max(1, psf.kernel.shape[1]))
        height = min(max(1, height), max(1, psf.kernel.shape[0]))
        cx = int(np.clip(x0 + width // 2, 0, psf.kernel.shape[1] - 1))
        cy = int(np.clip(y0 + height // 2, 0, psf.kernel.shape[0] - 1))

        widgets = (
            self.psf_calc_width_spin, self.psf_calc_height_spin,
            self.psf_center_mode_combo, self.psf_manual_x_spin, self.psf_manual_y_spin,
        )
        old_states = [widget.blockSignals(True) for widget in widgets]
        try:
            self.psf_calc_width_spin.setValue(width)
            self.psf_calc_height_spin.setValue(height)
            manual_index = self.psf_center_mode_combo.findData("manual")
            if manual_index >= 0:
                self.psf_center_mode_combo.setCurrentIndex(manual_index)
            self.psf_manual_x_spin.setValue(cx)
            self.psf_manual_y_spin.setValue(cy)
        finally:
            for widget, old in zip(widgets, old_states):
                widget.blockSignals(old)
        self._update_manual_center_controls()
        self.threshold_status_label.setText(
            f"Pending manual PSF window: center=(x={cx}, y={cy}), size={width}x{height} px. "
            "Press Apply to use it and update the histograms."
        )
        self.refresh()

    @staticmethod
    def _perimeter_mean(data: np.ndarray) -> float:
        """Return the finite-value mean on the one-pixel perimeter of a 2D array."""
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return float("nan")
        arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
        h, w = arr.shape
        if h == 1 or w == 1:
            edge = arr.ravel()
        else:
            edge = np.concatenate((arr[0, :], arr[-1, :], arr[1:-1, 0], arr[1:-1, -1]))
        edge = edge[np.isfinite(edge)]
        return float(np.mean(edge)) if edge.size else float("nan")

    @staticmethod
    def _original_image_region_from_record(record: Dict[str, Any]) -> np.ndarray:
        """Extract the calculation-scale image before its surrounding zero frame."""
        arr = np.asarray(record.get("data"), dtype=np.float64)
        if arr.ndim != 2:
            return arr
        metadata = dict(record.get("metadata", {}) or {})
        roi = metadata.get("content_roi")
        if isinstance(roi, (tuple, list)) and len(roi) == 4:
            y0, y1, x0, x1 = [int(v) for v in roi]
            y0 = max(0, min(arr.shape[0], y0))
            y1 = max(y0 + 1, min(arr.shape[0], y1))
            x0 = max(0, min(arr.shape[1], x0))
            x1 = max(x0 + 1, min(arr.shape[1], x1))
            return arr[y0:y1, x0:x1]
        padding = int(max(0, metadata.get("zero_padding", 0) or 0))
        if padding > 0 and 2 * padding < min(arr.shape):
            return arr[padding:arr.shape[0] - padding, padding:arr.shape[1] - padding]
        return arr

    def _set_image_floor_controls(self, value: float) -> None:
        value = float(np.clip(value, self.image_floor_spin.minimum(), self.image_floor_spin.maximum()))
        self._threshold_preview_timer.stop()
        self._threshold_syncing = True
        try:
            self.image_floor_spin.setValue(value)
            self.image_floor_slider.setValue(int(round(value * 10000.0)))
        finally:
            self._threshold_syncing = False

    def _set_psf_floor_controls(self, value: float) -> None:
        value = float(np.clip(value, self.psf_floor_spin.minimum(), self.psf_floor_spin.maximum()))
        self._threshold_preview_timer.stop()
        self._threshold_syncing = True
        try:
            self.psf_floor_spin.setValue(value)
            self.psf_floor_slider.setValue(int(round(value * 10000.0)))
        finally:
            self._threshold_syncing = False

    def auto_image_floor_from_border(self) -> None:
        image_base, _ = self._ensure_threshold_bases()
        if image_base is None:
            self.threshold_status_label.setText("No measured/degraded image is available for automatic image-floor estimation.")
            return
        original = self._original_image_region_from_record(image_base)
        border_mean = self._perimeter_mean(original)
        if not np.isfinite(border_mean):
            self.threshold_status_label.setText("Could not estimate the image floor from perimeter pixels.")
            return
        floor = float(np.clip(border_mean, 0.0, 1.0))
        self._set_image_floor_controls(floor)
        self.threshold_status_label.setText(
            f"Pending auto image floor={floor:.5f}, estimated as the mean of perimeter pixels in the "
            f"original non-padded region ({original.shape[0]}x{original.shape[1]}). Press Apply to use it."
        )
        self.refresh()

    def auto_psf_floor_from_border(self) -> None:
        _, psf_base = self._ensure_threshold_bases()
        if psf_base is None:
            self.threshold_status_label.setText("No PSF is available for automatic PSF-floor estimation.")
            return
        raw = np.asarray(psf_base.get("raw_kernel", psf_base.get("kernel")), dtype=np.float64)
        raw = np.maximum(np.nan_to_num(raw), 0.0)
        border_mean = self._perimeter_mean(raw)
        peak = float(np.max(raw)) if raw.size else 0.0
        if not np.isfinite(border_mean) or peak <= 1e-18:
            self.threshold_status_label.setText("Could not estimate the PSF floor from perimeter pixels.")
            return
        fraction = float(np.clip(border_mean / peak, 0.0, 1.0))
        self._set_psf_floor_controls(fraction)
        self.threshold_status_label.setText(
            f"Pending auto PSF floor/peak={fraction:.5f}: perimeter mean={border_mean:.6g}, "
            f"peak={peak:.6g}, raw PSF size={raw.shape[0]}x{raw.shape[1]}. Press Apply to use it."
        )
        self.refresh()

    def optimize_psf_floor_and_wiener_k(self) -> None:
        """Jointly tune the PSF floor and Wiener K on a compact preview."""
        image_base, psf_base = self._ensure_threshold_bases()
        degraded: Optional[GrayImage] = self.state.get("degraded")
        if degraded is None or psf_base is None:
            self.threshold_status_label.setText("Load or generate a degraded image and PSF before joint optimization.")
            return
        if not _try_begin_numerical_work("tab2_psf_floor_wiener"):
            owner = _current_numerical_owner() or "another calculation"
            self.threshold_status_label.setText(f"Cannot optimize now: {owner} is still running.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.optimize_psf_floor_k_button.setEnabled(False)
        try:
            source = np.asarray(psf_base.get("kernel"), dtype=np.float64)
            center = self._selected_psf_center(source)
            support_width = int(self.psf_calc_width_spin.value())
            support_height = int(self.psf_calc_height_spin.value())
            reference = None
            if reference_metrics_available(self.state) and isinstance(self.state.get("image"), GrayImage):
                reference = np.asarray(self.state["image"].data, dtype=np.float64)
            current_k = float(self.state.get("wiener_profile_k", self.state.get("optimized_wiener_k", 1e-2)))
            measured_for_optimization = (
                self._original_image_region_from_record(image_base)
                if isinstance(image_base, dict)
                else np.asarray(degraded.data, dtype=np.float64)
            )
            if reference is not None:
                reference = crop_to_original_region(reference, degraded)
            result = optimize_psf_floor_and_wiener_k(
                measured=np.asarray(measured_for_optimization, dtype=np.float64),
                source_psf=source,
                center=center,
                support_width=support_width,
                support_height=support_height,
                reference=reference,
                current_k=current_k,
                current_floor=float(self.psf_floor_spin.value()),
                max_preview_side=256,
            )
            floor = float(result["floor_fraction"])
            kval = float(result["K"])
            self._set_psf_floor_controls(floor)
            self.state["optimized_wiener_k"] = kval
            self.state["wiener_profile_k"] = kval
            self.state["psf_floor_wiener_optimization"] = dict(result)
            self.wienerKOptimized.emit(kval)
            improvement = float(result.get("cost_improvement", float("nan")))
            candidate_info = dict(result.get("candidate_psf", {}) or {})
            components = dict(result.get("criterion_components", {}) or {})
            background = dict(result.get("psf_background", {}) or {})
            bounds = tuple(result.get("floor_search_bounds", (floor, floor)))
            has_reference = str(result.get("criterion", "")) == "MSE"
            if has_reference:
                details = (
                    f"MSE {float(result.get('initial_cost', float('nan'))):.6g} → "
                    f"{float(result.get('cost', float('nan'))):.6g}; improvement={improvement:.6g}"
                )
            else:
                details = (
                    f"conditional GCV={float(components.get('conditional_gcv', float('nan'))):.6g}; "
                    f"PSF background={float(background.get('background_fraction', float('nan'))):.6g} "
                    f"± {float(background.get('sigma_fraction', float('nan'))):.3g}; "
                    f"allowed floor=[{float(bounds[0]):.6g}, {float(bounds[1]):.6g}]"
                )
            self.psf_floor_k_result_label.setText(
                f"Joint optimization result (floor pending Apply, K stored now): floor "
                f"{float(result.get('initial_floor_fraction', 0.0)):.8g} → {floor:.8g}; "
                f"K {float(result.get('initial_K', current_k)):.8g} → {kval:.8g}; {details}; "
                f"crop={support_width}x{support_height}; retained PSF mass="
                f"{float(candidate_info.get('retained_mass_fraction', float('nan'))):.4g}; "
                f"nonzero/effective pixels={int(candidate_info.get('nonzero_pixels', 0))}/"
                f"{float(candidate_info.get('effective_pixels', float('nan'))):.4g}; "
                f"normalized sum={float(candidate_info.get('normalized_sum', float('nan'))):.8g}."
            )
            self.threshold_status_label.setText(
                f"PSF-floor/Wiener optimization completed: floor/peak "
                f"{float(result.get('initial_floor_fraction', 0.0)):.6g} → {floor:.6g}, K "
                f"{float(result.get('initial_K', current_k)):.6g} → {kval:.6g}; {details}; "
                f"{int(result.get('evaluations', 0))} evaluations on preview {result.get('preview_shape')}. "
                "For measured data without a reference, GCV is used only to choose K for each fixed PSF; "
                "the floor is restricted by the measured PSF background and collapsed kernels are rejected. "
                "K is active immediately. Press Apply to commit the pending PSF floor."
            )
        except Exception as exc:
            self.psf_floor_k_result_label.setText(f"Joint optimization failed: {type(exc).__name__}: {exc}")
            self.threshold_status_label.setText(f"Joint PSF-floor/Wiener optimization failed: {exc}")
        finally:
            self.optimize_psf_floor_k_button.setEnabled(True)
            QApplication.restoreOverrideCursor()
            _end_numerical_work("tab2_psf_floor_wiener")

    def settings(self) -> Dict[str, Any]:
        return {
            "noise_type": self.noise_type_combo.currentText(),
            "noise_strength": self.noise_spin.value(),
            "image_lower_threshold": self.image_floor_spin.value(),
            "psf_lower_threshold_fraction": self.psf_floor_spin.value(),
            "psf_preview_mode": self.psf_preview_mode_combo.currentText(),
            "psf_calculation_width": self.psf_calc_width_spin.value(),
            "psf_calculation_height": self.psf_calc_height_spin.value(),
            "psf_calculation_size": self.psf_calc_width_spin.value(),
            "psf_center_mode": self._psf_center_mode_key(),
            "psf_center_of_mass": self._psf_center_mode_key() == "center_of_mass",
            "psf_manual_center_x": self.psf_manual_x_spin.value(),
            "psf_manual_center_y": self.psf_manual_y_spin.value(),
        }

    def apply_settings(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        if "noise_type" in data:
            idx = self.noise_type_combo.findText(str(data.get("noise_type")))
            if idx >= 0:
                self.noise_type_combo.setCurrentIndex(idx)
        if "noise_strength" in data:
            self.noise_spin.setValue(_safe_float(data.get("noise_strength"), self.noise_spin.value()))
        if "image_lower_threshold" in data:
            self.image_floor_spin.setValue(_safe_float(data.get("image_lower_threshold"), self.image_floor_spin.value()))
        if "psf_lower_threshold_fraction" in data:
            self.psf_floor_spin.setValue(_safe_float(data.get("psf_lower_threshold_fraction"), self.psf_floor_spin.value()))
        if "psf_preview_mode" in data:
            idx = self.psf_preview_mode_combo.findText(str(data.get("psf_preview_mode")))
            if idx >= 0:
                self.psf_preview_mode_combo.setCurrentIndex(idx)
        legacy_size = data.get("psf_calculation_size")
        if "psf_calculation_width" in data or legacy_size is not None:
            self.psf_calc_width_spin.setValue(_safe_int(data.get("psf_calculation_width", legacy_size), self.psf_calc_width_spin.value()))
        if "psf_calculation_height" in data or legacy_size is not None:
            self.psf_calc_height_spin.setValue(_safe_int(data.get("psf_calculation_height", legacy_size), self.psf_calc_height_spin.value()))
        center_mode = data.get("psf_center_mode")
        if center_mode is None and "psf_center_of_mass" in data:
            center_mode = "center_of_mass" if bool(data.get("psf_center_of_mass")) else "geometric"
        if center_mode is not None:
            idx = self.psf_center_mode_combo.findData(str(center_mode))
            if idx >= 0:
                self.psf_center_mode_combo.setCurrentIndex(idx)
        if "psf_manual_center_x" in data:
            self.psf_manual_x_spin.setValue(_safe_int(data.get("psf_manual_center_x"), self.psf_manual_x_spin.value()))
        if "psf_manual_center_y" in data:
            self.psf_manual_y_spin.setValue(_safe_int(data.get("psf_manual_center_y"), self.psf_manual_y_spin.value()))
        self._update_manual_center_controls()

    @staticmethod
    def _copy_image_record(image: GrayImage) -> Dict[str, Any]:
        return {
            "data": np.asarray(image.data, dtype=np.float64).copy(),
            "name": image.name,
            "metadata": dict(image.metadata or {}),
        }

    @staticmethod
    def _copy_psf_record(psf: PSF) -> Dict[str, Any]:
        return {
            "kernel": np.asarray(psf.kernel, dtype=np.float64).copy(),
            "raw_kernel": np.asarray(psf.raw_kernel if psf.raw_kernel is not None else psf.kernel, dtype=np.float64).copy(),
            "name": psf.name,
            "metadata": dict(getattr(psf, "metadata", {}) or {}),
        }

    def _ensure_threshold_bases(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        degraded: Optional[GrayImage] = self.state.get("degraded")
        psf: Optional[PSF] = self.state.get("psf")
        degradation_psf: Optional[PSF] = self.state.get("degradation_psf")

        # Capture a complete, immutable snapshot only once per thresholding
        # session.  Live slider previews and repeated presses of Apply must
        # always be derived from the same unmodified source data.
        if degraded is not None and "_tab2_threshold_base_degraded" not in self.state:
            self.state["_tab2_threshold_base_degraded"] = self._copy_image_record(degraded)
        if psf is not None and "_tab2_threshold_base_psf" not in self.state:
            self.state["_tab2_threshold_base_psf"] = self._copy_psf_record(psf)
        if degradation_psf is not None and "_tab2_threshold_base_degradation_psf" not in self.state:
            self.state["_tab2_threshold_base_degradation_psf"] = self._copy_psf_record(degradation_psf)
        if "_tab2_threshold_base_psf_selection" not in self.state:
            self.state["_tab2_threshold_base_psf_selection"] = {
                "support_width": int(self.state.get("psf_support_width", self.psf_calc_width_spin.value())),
                "support_height": int(self.state.get("psf_support_height", self.psf_calc_height_spin.value())),
                "center_mode": str(self.state.get("psf_calculation_center_mode", "center_of_mass")),
                "center_x": int(self.state.get("psf_calculation_center_x", self.psf_manual_x_spin.value())),
                "center_y": int(self.state.get("psf_calculation_center_y", self.psf_manual_y_spin.value())),
            }
        return self.state.get("_tab2_threshold_base_degraded"), self.state.get("_tab2_threshold_base_psf")

    @staticmethod
    def _floor_and_rescale(data: np.ndarray, floor: float) -> Tuple[np.ndarray, float]:
        """Apply a lower floor and stretch the surviving range to [0, original maximum].

        For an input maximum ``m`` and floor ``T``, values ``x <= T`` become
        zero, while values above the floor are transformed as
        ``m * (x - T) / (m - T)``. The returned second value is the original
        maximum. If the floor reaches or exceeds the maximum, the result is
        identically zero.
        """
        arr = np.asarray(data, dtype=np.float64)
        arr = np.maximum(np.nan_to_num(arr), 0.0)
        out = np.zeros_like(arr)
        maximum = float(np.max(arr)) if arr.size else 0.0
        threshold = float(max(0.0, floor))
        if maximum <= 1e-18:
            return out, maximum
        if threshold <= 0.0:
            return arr.copy(), maximum
        if threshold >= maximum - 1e-18:
            return out, maximum
        mask = arr > threshold
        out[mask] = maximum * (arr[mask] - threshold) / (maximum - threshold)
        return np.clip(out, 0.0, maximum), maximum

    def apply_lower_thresholds(self) -> None:
        image_base, psf_base = self._ensure_threshold_bases()
        messages: List[str] = []

        if image_base is not None:
            floor = float(self.image_floor_spin.value())
            source = np.asarray(image_base["data"], dtype=np.float64)
            arr, original_max = self._floor_and_rescale(source, floor)
            metadata = dict(image_base.get("metadata", {}))
            metadata.update({
                "_preserve_intensity": True,
                "_thresholded_in_tab2": True,
                "lower_threshold": floor,
                "threshold_rescale_max": original_max,
                "threshold_rescale_mode": "floor_to_zero_and_stretch",
            })
            self.state["degraded"] = GrayImage(arr, name=str(image_base.get("name", "degraded")) + "_thresholded", metadata=metadata)
            messages.append(f"image floor={floor:.5f}, rescaled to [0,{original_max:.5g}]")

        if psf_base is not None:
            fraction = float(self.psf_floor_spin.value())
            source_kernel = np.asarray(psf_base["kernel"], dtype=np.float64)
            peak = float(np.max(source_kernel)) if source_kernel.size else 0.0
            cutoff = fraction * peak
            kernel, original_peak = self._floor_and_rescale(source_kernel, cutoff)
            psf_meta = dict(psf_base.get("metadata", {}))
            center_mode = self._psf_center_mode_key()
            selected_center_y, selected_center_x = self._selected_psf_center(source_kernel)
            selected_height, selected_width = self._selected_psf_shape(source_kernel)
            # Commit the support selection explicitly in full-array coordinates:
            # samples outside the red rectangle become zero.  The compact
            # calculation PSF is then cropped from this masked array and
            # normalized to unit sum by the common PSF preparation path.
            x0 = int(selected_center_x - selected_width // 2)
            y0 = int(selected_center_y - selected_height // 2)
            kernel = zero_outside_psf_rectangle(
                kernel,
                (int(selected_center_y), int(selected_center_x)),
                int(selected_height),
                int(selected_width),
            )
            if float(kernel.sum()) <= 1e-18 and kernel.size:
                # Preserve a valid PSF even when thresholding or the chosen frame
                # removes the complete support. Put an impulse at the selected
                # centre (clipped to the full-array bounds).
                impulse_y = int(np.clip(selected_center_y, 0, kernel.shape[0] - 1))
                impulse_x = int(np.clip(selected_center_x, 0, kernel.shape[1] - 1))
                kernel[impulse_y, impulse_x] = max(original_peak, 1.0)
            self.state["psf_support_height"] = selected_height
            self.state["psf_support_width"] = selected_width
            self.state["psf_calculation_center_mode"] = center_mode
            self.state["psf_calculation_center_x"] = int(selected_center_x)
            self.state["psf_calculation_center_y"] = int(selected_center_y)
            psf_meta.update({
                "_thresholded_in_tab2": True,
                "calculation_center_mode": center_mode,
                "calculation_center": (int(selected_center_y), int(selected_center_x)),
                "calculation_support_height": selected_height,
                "calculation_support_width": selected_width,
                "lower_threshold_fraction": fraction,
                "lower_threshold_absolute": cutoff,
                "threshold_rescale_max": original_peak,
                "threshold_rescale_mode": "floor_to_zero_and_stretch",
                "outside_calculation_frame_zeroed": True,
                "calculation_frame_full_coordinates": (int(x0), int(y0), int(selected_width), int(selected_height)),
            })
            thresholded_psf = PSF(
                kernel,
                name=str(psf_base.get("name", "psf")) + "_thresholded",
                raw_kernel=np.asarray(psf_base.get("raw_kernel", source_kernel), dtype=np.float64).copy(),
                metadata=psf_meta,
            )
            self.state["psf"] = thresholded_psf
            calculation_image = _calculation_image_from_state(self.state)
            calculation_shape = (
                tuple(int(v) for v in calculation_image.data.shape)
                if calculation_image is not None
                else tuple(int(v) for v in thresholded_psf.kernel.shape)
            )
            calculation_psf = _synchronize_calculation_psf(self.state, calculation_shape)
            if calculation_psf is None:
                raise ValueError("The thresholded and cropped calculation PSF could not be created.")
            messages.append(
                f"PSF floor={fraction:.5f} of peak; calculation part={selected_width}x{selected_height} px, "
                f"center={center_mode} at (x={selected_center_x}, y={selected_center_y}); "
                f"calculation kernel={calculation_psf.kernel.shape[1]}x{calculation_psf.kernel.shape[0]} px, "
                f"sum={float(calculation_psf.kernel.sum()):.8g} after crop and normalization"
            )

        if psf_base is None:
            current_image = _calculation_image_from_state(self.state)
            if current_image is not None:
                _synchronize_calculation_psf(self.state, current_image.data.shape)
        if not messages:
            self.threshold_status_label.setText("Load or generate a measured/degraded image or PSF first.")
            return
        self.threshold_status_label.setText("Applied to calculations: " + "; ".join(messages) + ".")
        if psf_base is not None:
            self.calculationPsfSupportChanged.emit(max(
                int(self.state.get("psf_support_width", 1)),
                int(self.state.get("psf_support_height", 1)),
            ))
        self.refresh()
        self.calculationDataChanged.emit()

    def reset_lower_thresholds(self) -> None:
        image_base = self.state.get("_tab2_threshold_base_degraded")
        psf_base = self.state.get("_tab2_threshold_base_psf")
        degradation_psf_base = self.state.get("_tab2_threshold_base_degradation_psf")
        psf_selection_base = self.state.get("_tab2_threshold_base_psf_selection")
        restored = []

        # Restore from the original snapshots unconditionally.  Checking only
        # the current object's metadata was insufficient because another GUI
        # action could replace the thresholded object while leaving the paired
        # degradation PSF or cached preview state modified.
        if image_base is not None:
            metadata = dict(image_base.get("metadata", {}))
            metadata.pop("_thresholded_in_tab2", None)
            metadata.pop("lower_threshold", None)
            metadata.pop("threshold_rescale_max", None)
            metadata.pop("threshold_rescale_mode", None)
            metadata["_preserve_intensity"] = True
            self.state["degraded"] = GrayImage(
                np.asarray(image_base["data"], dtype=np.float64).copy(),
                name=str(image_base.get("name", "degraded")),
                metadata=metadata,
            )
            restored.append("image")

        if psf_base is not None:
            psf_metadata = dict(psf_base.get("metadata", {}))
            for key in (
                "_thresholded_in_tab2",
                "lower_threshold_fraction",
                "lower_threshold_absolute",
                "threshold_rescale_max",
                "threshold_rescale_mode",
            ):
                psf_metadata.pop(key, None)
            restored_psf = PSF(
                np.asarray(psf_base["kernel"], dtype=np.float64).copy(),
                name=str(psf_base.get("name", "psf")),
                raw_kernel=np.asarray(psf_base.get("raw_kernel", psf_base["kernel"]), dtype=np.float64).copy(),
                metadata=psf_metadata,
            )
            self.state["psf"] = restored_psf
            restored.append("PSF")

        if degradation_psf_base is not None:
            degradation_metadata = dict(degradation_psf_base.get("metadata", {}))
            for key in (
                "_thresholded_in_tab2",
                "lower_threshold_fraction",
                "lower_threshold_absolute",
                "threshold_rescale_max",
                "threshold_rescale_mode",
            ):
                degradation_metadata.pop(key, None)
            self.state["degradation_psf"] = PSF(
                np.asarray(degradation_psf_base["kernel"], dtype=np.float64).copy(),
                name=str(degradation_psf_base.get("name", "degradation_psf")),
                raw_kernel=np.asarray(
                    degradation_psf_base.get("raw_kernel", degradation_psf_base["kernel"]),
                    dtype=np.float64,
                ).copy(),
                metadata=degradation_metadata,
            )
        elif psf_base is not None and (
            bool(self.state.get("measured_pair_loaded", False))
            or self.state.get("reference_available") is False
        ):
            # Compatibility fallback for sessions created before the complete
            # degradation-PSF snapshot was introduced.
            restored_psf = self.state.get("psf")
            if isinstance(restored_psf, PSF):
                self.state["degradation_psf"] = PSF(
                    restored_psf.kernel.copy(),
                    name=restored_psf.name + "_paired",
                    raw_kernel=restored_psf.raw_kernel.copy(),
                    metadata=dict(restored_psf.metadata),
                )
        if isinstance(psf_selection_base, dict):
            restored_width = int(psf_selection_base.get("support_width", self.psf_calc_width_spin.value()))
            restored_height = int(psf_selection_base.get("support_height", psf_selection_base.get("support_width", self.psf_calc_height_spin.value())))
            restored_mode = str(psf_selection_base.get("center_mode", "center_of_mass"))
            restored_x = int(psf_selection_base.get("center_x", self.psf_manual_x_spin.value()))
            restored_y = int(psf_selection_base.get("center_y", self.psf_manual_y_spin.value()))
            self.state["psf_support_width"] = restored_width
            self.state["psf_support_height"] = restored_height
            self.state["psf_calculation_center_mode"] = restored_mode
            self.state["psf_calculation_center_x"] = restored_x
            self.state["psf_calculation_center_y"] = restored_y
            widgets = (self.psf_calc_width_spin, self.psf_calc_height_spin, self.psf_center_mode_combo, self.psf_manual_x_spin, self.psf_manual_y_spin)
            old_states = [widget.blockSignals(True) for widget in widgets]
            try:
                self.psf_calc_width_spin.setValue(restored_width)
                self.psf_calc_height_spin.setValue(restored_height)
                mode_index = self.psf_center_mode_combo.findData(restored_mode)
                if mode_index >= 0:
                    self.psf_center_mode_combo.setCurrentIndex(mode_index)
                self.psf_manual_x_spin.setValue(restored_x)
                self.psf_manual_y_spin.setValue(restored_y)
            finally:
                for widget, old in zip(widgets, old_states):
                    widget.blockSignals(old)
            self._update_manual_center_controls()
            restored.append("PSF calculation selection")

        self._threshold_preview_timer.stop()
        self._threshold_syncing = True
        try:
            self.image_floor_spin.setValue(0.0)
            self.image_floor_slider.setValue(0)
            self.psf_floor_spin.setValue(0.0)
            self.psf_floor_slider.setValue(0)
        finally:
            self._threshold_syncing = False
        # A reset ends the current thresholding session.  The next preview
        # starts from the newly restored state and takes a fresh snapshot.
        self.state.pop("_tab2_threshold_base_degraded", None)
        self.state.pop("_tab2_threshold_base_psf", None)
        self.state.pop("_tab2_threshold_base_degradation_psf", None)
        self.state.pop("_tab2_threshold_base_psf_selection", None)
        restored_image = _calculation_image_from_state(self.state)
        restored_shape = restored_image.data.shape if restored_image is not None else None
        _synchronize_calculation_psf(self.state, restored_shape)
        self.threshold_status_label.setText(
            "Restored disk-loaded/generated " + " and ".join(restored) + "."
            if restored else "No applied threshold to reset."
        )
        self.refresh()
        self.calculationDataChanged.emit()

    def refresh(self) -> None:
        reference: Optional[GrayImage] = self.state.get("image")
        psf: Optional[PSF] = self.state.get("psf")
        degraded: Optional[GrayImage] = self.state.get("degraded")
        calculation_shape = degraded.data.shape if degraded is not None else (reference.data.shape if reference is not None else None)
        calculation_psf = _synchronize_calculation_psf(self.state, calculation_shape) if calculation_shape is not None else None

        selection_generation = int(self.state.get("psf_selection_generation", 0))
        if psf is not None and selection_generation != self._last_psf_selection_generation:
            self._last_psf_selection_generation = selection_generation
            width = int(self.state.get("psf_support_width", psf.kernel.shape[1]))
            height = int(self.state.get("psf_support_height", psf.kernel.shape[0]))
            width = min(max(1, width), psf.kernel.shape[1])
            height = min(max(1, height), psf.kernel.shape[0])
            mode = str(self.state.get("psf_calculation_center_mode", "center_of_mass"))
            cx = int(self.state.get("psf_calculation_center_x", PSF.support_center(psf.kernel)[1]))
            cy = int(self.state.get("psf_calculation_center_y", PSF.support_center(psf.kernel)[0]))
            widgets = (self.psf_calc_width_spin, self.psf_calc_height_spin, self.psf_center_mode_combo, self.psf_manual_x_spin, self.psf_manual_y_spin, self.psf_preview_mode_combo)
            blocked = [widget.blockSignals(True) for widget in widgets]
            try:
                self.psf_calc_width_spin.setValue(max(1, width))
                self.psf_calc_height_spin.setValue(max(1, height))
                mode_index = self.psf_center_mode_combo.findData(mode)
                if mode_index >= 0:
                    self.psf_center_mode_combo.setCurrentIndex(mode_index)
                self.psf_manual_x_spin.setValue(cx)
                self.psf_manual_y_spin.setValue(cy)
                full_index = self.psf_preview_mode_combo.findText("Full PSF array")
                if full_index >= 0:
                    self.psf_preview_mode_combo.setCurrentIndex(full_index)
            finally:
                for widget, old in zip(widgets, blocked):
                    widget.blockSignals(old)
            self._update_manual_center_controls()
            auto = self.state.get("psf_automatic_selection", {})
            self.threshold_status_label.setText(
                f"Automatic initial PSF selection: center=(x={cx}, y={cy}), size={width}x{height} px; "
                f"estimated floor/peak={float(auto.get('floor_fraction', 0.0)):.6g}. "
                "Adjust the red frame if desired, then press Apply."
            )
            self.calculationPsfSupportChanged.emit(max(1, width, height))

        self.reference_canvas.setVisible(reference is not None and reference_metrics_available(self.state))
        has_reference = reference is not None and reference_metrics_available(self.state)
        self.reference_canvas.setVisible(has_reference)
        self.reference_canvas.show_image(
            reference.data if has_reference else None,
            "Reference image (metrics only; not reconstruction input)",
        )
        if psf is not None:
            max_width = max(1, psf.kernel.shape[1])
            max_height = max(1, psf.kernel.shape[0])
            old_w = self.psf_calc_width_spin.blockSignals(True)
            old_h = self.psf_calc_height_spin.blockSignals(True)
            self.psf_calc_width_spin.setMaximum(max_width)
            self.psf_calc_height_spin.setMaximum(max_height)
            if self.psf_calc_width_spin.value() > max_width:
                self.psf_calc_width_spin.setValue(max_width)
            if self.psf_calc_height_spin.value() > max_height:
                self.psf_calc_height_spin.setValue(max_height)
            self.psf_calc_width_spin.blockSignals(old_w)
            self.psf_calc_height_spin.blockSignals(old_h)
            old_x = self.psf_manual_x_spin.blockSignals(True)
            old_y = self.psf_manual_y_spin.blockSignals(True)
            self.psf_manual_x_spin.setMaximum(max(0, psf.kernel.shape[1] - 1))
            self.psf_manual_y_spin.setMaximum(max(0, psf.kernel.shape[0] - 1))
            self.psf_manual_x_spin.blockSignals(old_x)
            self.psf_manual_y_spin.blockSignals(old_y)
            full_preview = self.psf_preview_mode_combo.currentText().startswith("Full")
            shown_psf = psf.kernel if full_preview else (calculation_psf.kernel if calculation_psf is not None else None)
            full_rectangle = self._selected_psf_rectangle(psf.kernel)
            selection_rectangle = (
                full_rectangle
                if full_preview
                else ((0, 0, int(shown_psf.shape[1]), int(shown_psf.shape[0])) if shown_psf is not None else None)
            )
            center_mode = self._psf_center_mode_key()
            center_description = {
                "center_of_mass": "center of mass",
                "geometric": "geometric center",
                "manual": f"manual center (x={self.psf_manual_x_spin.value()}, y={self.psf_manual_y_spin.value()})",
            }.get(center_mode, center_mode)
            applied_mask = bool((psf.metadata or {}).get("outside_calculation_frame_zeroed", False))
            psf_title = (
                (f"Full applied PSF {psf.kernel.shape[1]}x{psf.kernel.shape[0]}; outside the last applied frame is zero; "
                 "red frame shows the current pending/applied selection")
                if full_preview and applied_mask
                else (f"Full PSF source {psf.kernel.shape[1]}x{psf.kernel.shape[0]}; red frame is used only after Apply"
                      if full_preview
                      else (f"Exact applied calculation PSF {shown_psf.shape[1]}x{shown_psf.shape[0]} around {center_description}; "
                            f"sum={float(shown_psf.sum()):.8g}" if shown_psf is not None else "Calculation PSF"))
            )
        else:
            shown_psf = None
            psf_title = "PSF"
            selection_rectangle = None
        self.psf_canvas.show_image(
            shown_psf,
            psf_title,
            normalize_display=True,
            selection_rectangle=selection_rectangle,
        )
        self.psf_canvas.set_selection_editing(
            bool(psf is not None and self.psf_preview_mode_combo.currentText().startswith("Full")),
            self._psf_rectangle_edited,
        )
        self.degraded_canvas.show_image(
            degraded.data if degraded else None,
            f"Calculation input {degraded.data.shape[1]}x{degraded.data.shape[0]} px (after thresholding)" if degraded else "Calculation input",
        )

        self.image_histogram.show_histogram(
            degraded.data if degraded is not None else None,
            "Applied calculation-input histogram (256 bins)",
            threshold=float(self.image_floor_spin.value()),
            relative_to_peak=False,
        )
        self.psf_histogram.show_histogram(
            calculation_psf.kernel if calculation_psf is not None else None,
            "Applied calculation-PSF histogram (256 bins; cropped and normalized)",
            threshold=float(self.psf_floor_spin.value()),
            relative_to_peak=True,
        )
        self.calculation_info_label.setText(_calculation_data_summary(self.state))
        self.update_metrics()

    def generate_degraded_input(self) -> None:
        image: Optional[GrayImage] = self.state.get("image")
        psf: Optional[PSF] = self.state.get("psf")
        if image is None or psf is None:
            QMessageBox.warning(self, "Missing data", "Load/generate a reference image and PSF first.")
            return
        run_psf = _synchronize_calculation_psf(self.state, image.data.shape)
        if run_psf is None:
            QMessageBox.warning(self, "Missing PSF", "No valid thresholded/cropped calculation PSF is available.")
            return
        self.state["degradation_psf"] = PSF(run_psf.kernel.copy(), name=run_psf.name + "_forward", metadata=dict(run_psf.metadata or {}))
        self.state["degraded"] = degrade_image(image, run_psf, self.noise_spin.value(), self.noise_type_combo.currentText())
        self.state.pop("_tab2_threshold_base_degraded", None)
        self.state.pop("_tab2_threshold_base_psf", None)
        self.state.pop("_tab2_threshold_base_degradation_psf", None)
        self.state.pop("_tab2_threshold_base_psf_selection", None)
        self.refresh()
        self.calculationDataChanged.emit()

    def update_metrics(self) -> None:
        degraded: Optional[GrayImage] = self.state.get("degraded")
        if degraded is None:
            self.metrics_label.setText("Metrics in original (non-padded) region: PSNR: -    SSIM: -    TV: -")
            return
        allow_ref = reference_metrics_available(self.state)
        metrics = compute_metrics(self.state.get("image"), degraded, allow_reference_metrics=allow_ref, roi_source=degraded)
        tv_text = f"TV: {metrics.get('TV', float('nan')):.6f}" if np.isfinite(metrics.get('TV', float('nan'))) else "TV: n/a"
        if allow_ref and np.isfinite(metrics.get("PSNR", float("nan"))) and np.isfinite(metrics.get("SSIM", float("nan"))):
            self.metrics_label.setText(
                f"Original region only — PSNR: {metrics['PSNR']:.3f} dB    SSIM: {metrics['SSIM']:.4f}    {tv_text}"
            )
        else:
            self.metrics_label.setText(f"Original region only — measured input: PSNR/SSIM not computed    {tv_text}")


class AutoTuneWorker(QObject):
    """Runs Auto/Auto All outside the GUI thread."""

    progress = pyqtSignal(str)
    algorithm_finished = pyqtSignal(str, dict, float, str)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        alg_tab: "AlgorithmTab",
        jobs: List[Tuple[str, Dict[str, Any]]],
        numerical_payload: Dict[str, Any],
        cancel_grace_seconds: float = 5.0,
    ) -> None:
        super().__init__()
        self.alg_tab = alg_tab
        self.jobs = jobs
        self.numerical_payload = dict(numerical_payload)
        self.cancel_grace_seconds = float(cancel_grace_seconds)
        self._cancelled = False
        self._numerical_client: Optional[AutoNumericalProcessClient] = None

    def cancel(self) -> None:
        self._cancelled = True
        try:
            self.alg_tab._auto_cancel_requested = True
        except Exception:
            pass
        client = self._numerical_client
        if client is not None:
            try:
                client.cancel()
            except Exception:
                pass

    def run(self) -> None:
        owner = f"auto:{id(self)}"
        if not _try_begin_numerical_work(owner):
            active = _current_numerical_owner() or "another numerical task"
            self.failed.emit(f"Auto cannot start while {active} is still running.")
            self.finished.emit()
            return
        try:
            with _NUMERICAL_WORK_LOCK:
                if self._cancelled:
                    self.progress.emit("Auto cancelled before the numerical process was started.")
                    return
                try:
                    self.progress.emit("Auto: starting isolated numerical process ...")
                    self._numerical_client = AutoNumericalProcessClient(
                        self.numerical_payload, cancel_grace_seconds=self.cancel_grace_seconds
                    )
                    self.alg_tab._auto_numerical_client = self._numerical_client
                except Exception as exc:
                    self._numerical_client = None
                    self.alg_tab._auto_numerical_client = None
                    self.failed.emit(
                        f"Auto cannot start the isolated numerical process: {type(exc).__name__}: {exc}"
                    )
                    return

                for index, (alg_name, params) in enumerate(self.jobs, start=1):
                    if self._cancelled:
                        self.progress.emit("Auto cancelled.")
                        break
                    self.progress.emit(f"Auto {index}/{len(self.jobs)}: tuning {alg_name} ...")
                    best_params, best_score, status = self.alg_tab._auto_tune_algorithm_sync(alg_name, dict(params))
                    if self._cancelled or getattr(self.alg_tab, "_auto_cancel_requested", False):
                        client = self._numerical_client
                        if client is not None and client.forced:
                            self.progress.emit(
                                f"Auto cancelled: current numerical iteration was force-stopped after "
                                f"{self.cancel_grace_seconds:.1f} s."
                            )
                        else:
                            self.progress.emit(status if "cancel" in status.lower() else "Auto cancelled.")
                        break
                    self.algorithm_finished.emit(alg_name, best_params, float(best_score), status)
        except AutoCancelledError as exc:
            self._cancelled = True
            self.progress.emit(f"Auto cancelled: {exc}")
        except Exception as exc:
            self.failed.emit(f"Auto worker error: {type(exc).__name__}: {exc}")
        finally:
            client = self._numerical_client
            self.alg_tab._auto_numerical_client = None
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            self._numerical_client = None
            _safe_torch_worker_cleanup()
            _end_numerical_work(owner)
            self.finished.emit()

class AlgorithmTab(QWidget):
    _TORCH_BATCH_PAIRS = {
        "Wiener": "Torch batch Wiener",
        "Richardson-Lucy": "Torch batch Richardson-Lucy",
        "Richardson-Lucy-Wiener": "Torch batch Richardson-Lucy-Wiener",
        "Richardson-Lucy-Rosen": "Torch batch Richardson-Lucy-Rosen",
        "Landweber": "Torch batch Landweber",
    }
    _TORCH_BATCH_REVERSE = {v: k for k, v in _TORCH_BATCH_PAIRS.items()}

    def _is_torch_batch_algorithm(self, name: str) -> bool:
        return str(name) in self._TORCH_BATCH_REVERSE

    def _display_algorithm_names(self) -> List[str]:
        """Algorithms shown in the combo box.

        Torch-batch variants are deliberately hidden and controlled by the
        per-algorithm "PyTorch batch" checkbox instead.  The registry still
        contains the actual Torch implementations for execution and Auto.
        """
        hidden = set(self._TORCH_BATCH_REVERSE)
        return [name for name in self.registry.names() if name not in hidden]

    def _has_torch_batch_equivalent(self, name: str) -> bool:
        torch_name = self._TORCH_BATCH_PAIRS.get(str(name))
        if not torch_name:
            return False
        try:
            self.registry.get(torch_name)
            return True
        except Exception:
            return False

    def __init__(self, app_state: Dict[str, Any], registry: AlgorithmRegistry) -> None:
        super().__init__()
        self.state = app_state
        self.registry = registry
        self._resolution_default_psf_support = int(
            self.state.get("resolution_linked_psf_support", resolution_linked_psf_support((1024, 1280), 0.45))
        )
        self._loading_algorithm_params = False
        self._previous_algorithm_name = ""
        self._algorithm_param_values: Dict[str, Dict[str, Any]] = {}
        layout = QVBoxLayout(self)

        box = QGroupBox("Algorithm and parameters")
        form = QFormLayout(box)
        self.combo = QComboBox()
        self.combo.addItems(self._display_algorithm_names())
        self.use_torch_batch_check = QCheckBox("PyTorch batch")
        self.use_torch_batch_check.setChecked(True)
        self.use_torch_batch_check.setToolTip("Use the PyTorch batched implementation for this algorithm when available. Turn it off to run the ordinary/reference implementation.")
        self.psf_policy_label = QLabel()
        self.psf_policy_label.setWordWrap(True)
        self.psf_policy_label.setToolTip(
            "Non-blind methods use the known PSF selection controlled in Tab 2. "
            "Blind methods use the known PSF only as an optional initial estimate and constrain the estimated PSF with the controls shown below."
        )
        self.blind_psf_support_info_label = QLabel()
        self.blind_psf_support_info_label.setWordWrap(True)
        self.k_spin = QDoubleSpinBox()
        self.k_spin.setRange(1e-12, 1e4)
        self.k_spin.setDecimals(12)
        self.k_spin.setValue(0.01)
        self.k_spin.setToolTip(
            "Wiener regularization K. Auto searches K logarithmically across many orders of magnitude, "
            "then performs two local log-space refinement passes."
        )
        self.wiener_k_scan_check = QCheckBox("Generate logarithmic Wiener K scan")
        self.wiener_k_scan_check.setChecked(False)
        self.wiener_k_scan_check.setToolTip(
            "For the plain Wiener filter, run several independent reconstructions with logarithmically spaced K values. "
            "The results are stored as consecutive frames in the Test tab for visual comparison."
        )
        self.wiener_k_scan_min_spin = QDoubleSpinBox()
        self.wiener_k_scan_min_spin.setRange(1e-12, 1e4)
        self.wiener_k_scan_min_spin.setDecimals(12)
        self.wiener_k_scan_min_spin.setValue(1e-10)
        self.wiener_k_scan_min_spin.setToolTip("Smallest K included in the logarithmic scan.")
        self.wiener_k_scan_max_spin = QDoubleSpinBox()
        self.wiener_k_scan_max_spin.setRange(1e-12, 1e4)
        self.wiener_k_scan_max_spin.setDecimals(12)
        self.wiener_k_scan_max_spin.setValue(1e-1)
        self.wiener_k_scan_max_spin.setToolTip("Largest K included in the logarithmic scan.")
        self.wiener_k_scan_points_spin = QSpinBox()
        self.wiener_k_scan_points_spin.setRange(2, 200)
        self.wiener_k_scan_points_spin.setValue(31)
        self.wiener_k_scan_points_spin.setToolTip(
            "Number of logarithmically spaced K values. Each value appears as one browsable frame in the Test tab."
        )
        self.copy_wiener_settings_button = QPushButton("Copy Wiener settings to all applicable algorithms")
        self.copy_wiener_settings_button.setToolTip(
            "Copy the saved Wiener profile values K and noise-PSD use "
            "to every algorithm that uses a Wiener stage. All methods always use the current "
            "thresholded, cropped and normalized PSF from Tab 2; Begin with Wiener is left unchanged."
        )
        self.copy_wiener_settings_button.clicked.connect(self.copy_wiener_settings_to_all)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 500)
        self.iter_spin.setValue(20)
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(1e-12, 1e-2)
        self.epsilon_spin.setDecimals(12)
        self.epsilon_spin.setSingleStep(1e-8)
        self.epsilon_spin.setValue(1e-8)
        self.landweber_step_spin = QDoubleSpinBox()
        self.landweber_step_spin.setRange(1e-6, 5.0)
        self.landweber_step_spin.setDecimals(6)
        self.landweber_step_spin.setSingleStep(0.05)
        self.landweber_step_spin.setValue(0.8)
        self.kaczmarz_relaxation_spin = QDoubleSpinBox()
        self.kaczmarz_relaxation_spin.setRange(1e-6, 5.0)
        self.kaczmarz_relaxation_spin.setDecimals(6)
        self.kaczmarz_relaxation_spin.setSingleStep(0.01)
        self.kaczmarz_relaxation_spin.setValue(0.15)
        self.kaczmarz_block_size_spin = QSpinBox()
        self.kaczmarz_block_size_spin.setRange(4, 256)
        self.kaczmarz_block_size_spin.setValue(32)
        self.kaczmarz_blocks_per_iteration_spin = QSpinBox()
        self.kaczmarz_blocks_per_iteration_spin.setRange(1, 4096)
        self.kaczmarz_blocks_per_iteration_spin.setValue(16)
        self.kaczmarz_full_sweep_check = QCheckBox("Kaczmarz full sweep over all blocks each iteration")
        self.kaczmarz_full_sweep_check.setChecked(True)
        self.kaczmarz_overlap_check = QCheckBox("Use overlapping Kaczmarz blocks")
        self.kaczmarz_overlap_check.setChecked(True)
        self.kaczmarz_randomized_check = QCheckBox("Visit Kaczmarz blocks in random order")
        self.kaczmarz_randomized_check.setChecked(False)
        self.kaczmarz_shift_grid_check = QCheckBox("Shift Kaczmarz block grid between iterations")
        self.kaczmarz_shift_grid_check.setChecked(True)
        self.kaczmarz_window_check = QCheckBox("Use smooth Kaczmarz block window")
        self.kaczmarz_window_check.setChecked(True)
        self.kaczmarz_stabilized_check = QCheckBox("Use stabilized Kaczmarz full-sweep update")
        self.kaczmarz_stabilized_check.setChecked(True)
        self.kaczmarz_damping_spin = QDoubleSpinBox()
        self.kaczmarz_damping_spin.setRange(0.01, 1.0)
        self.kaczmarz_damping_spin.setDecimals(3)
        self.kaczmarz_damping_spin.setSingleStep(0.05)
        self.kaczmarz_damping_spin.setValue(0.50)
        self.kaczmarz_max_update_spin = QDoubleSpinBox()
        self.kaczmarz_max_update_spin.setRange(0.0, 2.0)
        self.kaczmarz_max_update_spin.setDecimals(3)
        self.kaczmarz_max_update_spin.setSingleStep(0.05)
        self.kaczmarz_max_update_spin.setValue(0.25)
        self.rosen_l_spin = QDoubleSpinBox()
        self.rosen_l_spin.setRange(0.0, 5.0)
        self.rosen_l_spin.setDecimals(3)
        self.rosen_l_spin.setSingleStep(0.05)
        self.rosen_l_spin.setValue(0.5)
        self.rosen_m_spin = QDoubleSpinBox()
        self.rosen_m_spin.setRange(0.0, 5.0)
        self.rosen_m_spin.setDecimals(3)
        self.rosen_m_spin.setSingleStep(0.05)
        self.rosen_m_spin.setValue(0.5)
        self.rosen_relax_check = QCheckBox("Slowly relax Rosen L and M toward 1 after each iteration")
        self.rosen_relax_check.setChecked(False)
        self.rosen_relax_factor_spin = QDoubleSpinBox()
        self.rosen_relax_factor_spin.setRange(0.0, 1.0)
        self.rosen_relax_factor_spin.setDecimals(4)
        self.rosen_relax_factor_spin.setSingleStep(0.01)
        self.rosen_relax_factor_spin.setValue(0.98)
        self.rosen_match_rl_button = QPushButton("Set Rosen from Richardson–Lucy (L=M=1)")
        self.rosen_match_rl_button.setToolTip(
            "Copy the common Richardson–Lucy settings to Richardson–Lucy–Rosen, "
            "set both nonlinear exponents to 1, and disable relaxation toward 1."
        )
        self.rosen_match_rl_button.clicked.connect(self.set_rosen_from_richardson_lucy)
        self.psf_sigma_spin = QDoubleSpinBox()
        self.psf_sigma_spin.setRange(0.1, 30.0)
        self.psf_sigma_spin.setDecimals(3)
        self.psf_sigma_spin.setValue(3.0)
        self.non_negative_check = QCheckBox("Replace negative values with zero after each iteration")
        self.non_negative_check.setChecked(True)
        self.begin_with_wiener_check = QCheckBox("Apply one Wiener filter before the selected algorithm")
        self.begin_with_wiener_check.setChecked(False)
        self.wiener_use_noise_psd_check = QCheckBox("Use generated noise power spectrum in Wiener denominator")
        self.wiener_use_noise_psd_check.setChecked(False)
        self.tv_preconditioning_check = QCheckBox("Apply a TV proximal step after each iteration")
        self.tv_preconditioning_check.setChecked(False)
        self.tv_weight_spin = QDoubleSpinBox()
        self.tv_weight_spin.setRange(0.0, 1.0)
        self.tv_weight_spin.setDecimals(6)
        self.tv_weight_spin.setSingleStep(0.001)
        self.tv_weight_spin.setValue(0.005)
        self.tv_iterations_spin = QSpinBox()
        self.tv_iterations_spin.setRange(1, 200)
        self.tv_iterations_spin.setValue(5)
        self.denoiser_type_combo = QComboBox()
        self.denoiser_type_combo.addItems([
            "TV only",
            "Gaussian",
            "Bilateral",
            "Non-local Means",
            "Wavelet",
            "DnCNN",
            "FFDNet",
            "DRUNet",
            "SCUNet",
            "Model zoo / custom manifest",
            "Neural CNN loaded from file",
            "Lightweight CNN fallback",
        ])
        self.denoiser_type_combo.setCurrentText("TV only")
        self.neural_denoiser_mode_combo = QComboBox()
        self.neural_denoiser_mode_combo.addItems(["Off", "Before algorithm", "After each iteration"])
        self.neural_denoiser_strength_spin = QDoubleSpinBox()
        self.neural_denoiser_strength_spin.setRange(0.0, 1.0)
        self.neural_denoiser_strength_spin.setDecimals(3)
        self.neural_denoiser_strength_spin.setSingleStep(0.05)
        self.neural_denoiser_strength_spin.setValue(0.15)
        self.neural_denoiser_weights_edit = QLineEdit()
        self.neural_denoiser_weights_edit.setPlaceholderText(".pt/.pth weights; optional sidecar .json manifest")
        self.neural_denoiser_browse_button = QPushButton("Browse weights")
        weights_row = QHBoxLayout()
        weights_row.addWidget(self.neural_denoiser_weights_edit)
        weights_row.addWidget(self.neural_denoiser_browse_button)
        weights_widget = QWidget()
        weights_widget.setLayout(weights_row)
        self.neural_weights_widget = weights_widget
        self.neural_denoiser_browse_button.clicked.connect(self.browse_neural_weights)
        self.torch_lr_spin = QDoubleSpinBox()
        self.torch_lr_spin.setRange(1e-6, 10.0)
        self.torch_lr_spin.setDecimals(6)
        self.torch_lr_spin.setSingleStep(0.005)
        self.torch_lr_spin.setValue(0.05)
        self.blind_psf_lr_spin = QDoubleSpinBox()
        self.blind_psf_lr_spin.setRange(1e-6, 10.0)
        self.blind_psf_lr_spin.setDecimals(6)
        self.blind_psf_lr_spin.setSingleStep(0.001)
        self.blind_psf_lr_spin.setValue(0.01)
        self.blind_psf_tv_weight_spin = QDoubleSpinBox()
        self.blind_psf_tv_weight_spin.setRange(0.0, 1.0)
        self.blind_psf_tv_weight_spin.setDecimals(6)
        self.blind_psf_tv_weight_spin.setSingleStep(0.0005)
        self.blind_psf_tv_weight_spin.setValue(0.0005)
        self.blind_psf_rot_sym_check = QCheckBox("Constrain estimated PSF to rotational symmetry")
        self.blind_psf_rot_sym_check.setChecked(False)
        self.blind_use_known_psf_init_check = QCheckBox("Initialize estimated PSF from current known PSF")
        self.blind_use_known_psf_init_check.setChecked(True)
        self.blind_use_known_psf_init_check.setToolTip(
            "When a PSF has been loaded or generated, use its calculation support as the initial blind estimate. "
            "When disabled, or when no PSF is known, initialize with a Gaussian PSF."
        )
        self.prefer_cuda_check = QCheckBox("Use CUDA when available")
        self.prefer_cuda_check.setChecked(True)
        self.torch_float64_check = QCheckBox("Use float64 for Torch computations (default: float32)")
        self.torch_float64_check.setChecked(False)
        self.torch_record_every_spin = QSpinBox()
        self.torch_record_every_spin.setRange(1, 100)
        self.torch_record_every_spin.setValue(1)
        self.auto_batch_size_spin = QSpinBox()
        self.auto_batch_size_spin.setRange(1, 8192)
        self.auto_batch_size_spin.setValue(32)
        self.auto_batch_size_spin.setToolTip("Actual Auto batch used for the next optimization. It can be raised automatically by the GPU benchmark.")
        self.auto_max_batch_size_spin = QSpinBox()
        self.auto_max_batch_size_spin.setRange(1, 8192)
        self.auto_max_batch_size_spin.setValue(2048)
        self.auto_max_batch_size_spin.setToolTip("Upper limit tested by the CUDA batch-size benchmark.")
        self.auto_batch_policy_combo = QComboBox()
        self.auto_batch_policy_combo.addItems(["Conservative (50%)", "Balanced (75%)", "Aggressive (95%)"] )
        self.auto_batch_policy_combo.setCurrentText("Balanced (75%)")
        self.auto_batch_policy_combo.setToolTip("Safety margin applied to the largest CUDA batch that passes the memory probe.")
        self.auto_batch_detected_label = QLabel("Detected CUDA batch: not benchmarked")
        self._auto_batch_cache: Dict[str, int] = {}
        self.auto_strategy_combo = QComboBox()
        self.auto_strategy_combo.addItems([
            "Quadratic coordinate (fast)",
            "Full local grid (thorough)",
        ])
        self.auto_strategy_combo.setCurrentText("Quadratic coordinate (fast)")
        self.auto_strategy_combo.setToolTip(
            "Quadratic coordinate evaluates a few points per numeric parameter and fits a local parabola. "
            "Full local grid evaluates the Cartesian product of local candidates and is more expensive."
        )
        self.auto_tune_numeric_check = QCheckBox("Tune numeric parameters")
        self.auto_tune_numeric_check.setChecked(True)
        self.auto_tune_numeric_check.setToolTip("Tune scalar numeric hyperparameters within about +/-50% of the current value.")
        self.auto_tune_boolean_check = QCheckBox("Tune boolean parameters")
        self.auto_tune_boolean_check.setChecked(False)
        self.auto_tune_boolean_check.setToolTip("Try both unchecked/checked for safe algorithmic checkboxes. Disabled by default to avoid changing qualitative assumptions silently.")
        self.auto_tune_categorical_check = QCheckBox("Tune categorical options")
        self.auto_tune_categorical_check.setChecked(False)
        self.auto_tune_categorical_check.setToolTip("Try safe categorical choices. Currently used mainly for denoiser timing/type when denoiser tuning is enabled.")
        self.auto_tune_denoiser_check = QCheckBox("Tune denoiser type/timing")
        self.auto_tune_denoiser_check.setChecked(False)
        self.auto_tune_denoiser_check.setToolTip("Try safe denoiser choices: Off/Before/After and TV/Gaussian/Bilateral/NLM/Wavelet. External neural weights are not changed.")
        self.auto_tune_denoiser_strength_check = QCheckBox("Tune denoiser strength")
        self.auto_tune_denoiser_strength_check.setChecked(True)
        self.auto_tune_denoiser_strength_check.setToolTip("Tune the numeric denoiser strength when a denoiser is active or denoiser tuning is enabled.")
        self.auto_tune_tv_check = QCheckBox("Tune TV parameters")
        self.auto_tune_tv_check.setChecked(True)
        self.auto_tune_tv_check.setToolTip("Tune TV weight/iterations and, when boolean tuning is enabled, the TV preconditioning checkbox.")
        self.auto_tune_wiener_init_check = QCheckBox("Tune Wiener initialization")
        self.auto_tune_wiener_init_check.setChecked(False)
        self.auto_tune_wiener_init_check.setToolTip("Allow Auto to toggle Begin with Wiener filter and tune Wiener K for initialization-related choices.")
        self.auto_max_candidates_spin = QSpinBox()
        self.auto_max_candidates_spin.setRange(16, 4096)
        self.auto_max_candidates_spin.setValue(256)
        self.auto_max_candidates_spin.setToolTip("Maximum number of candidate parameter sets generated by full batched Auto.")
        self.torch_status_label = QLabel(f"PyTorch: {'available' if TORCH_AVAILABLE else 'not installed'}; device: {torch_device_name(True)}")
        self.auto_button = QPushButton("Auto")
        self.auto_all_button = QPushButton("Auto All")
        self.cancel_auto_button = QPushButton("Cancel Auto")
        self.cancel_auto_button.setEnabled(False)
        self.cancel_auto_button.setToolTip("Request cooperative cancellation immediately. If the current numerical iteration has not returned after 5 seconds, the isolated Auto process is force-stopped.")
        self._auto_cancel_requested = False
        self._auto_numerical_client: Optional[AutoNumericalProcessClient] = None
        self._auto_process_failed = False
        self._auto_thread = None
        self._auto_worker = None
        self._auto_threads: List[Tuple[QThread, AutoTuneWorker]] = []
        self._auto_running = False
        self.auto_status = QLabel("Auto: not run")
        form.addRow("Algorithm", self.combo)
        form.addRow("Computation backend", self.use_torch_batch_check)
        form.addRow("PSF support used by selected method", self.psf_policy_label)
        form.addRow("Wiener K", self.k_spin)
        form.addRow("Manual K scan", self.wiener_k_scan_check)
        form.addRow("K scan minimum", self.wiener_k_scan_min_spin)
        form.addRow("K scan maximum", self.wiener_k_scan_max_spin)
        form.addRow("K scan points", self.wiener_k_scan_points_spin)
        form.addRow("Wiener profile", self.copy_wiener_settings_button)
        form.addRow("Iterations", self.iter_spin)
        form.addRow("Epsilon", self.epsilon_spin)
        form.addRow("Landweber step", self.landweber_step_spin)
        form.addRow("Kaczmarz relaxation", self.kaczmarz_relaxation_spin)
        form.addRow("Kaczmarz block size", self.kaczmarz_block_size_spin)
        form.addRow("Kaczmarz blocks per iteration", self.kaczmarz_blocks_per_iteration_spin)
        form.addRow("Kaczmarz full sweep", self.kaczmarz_full_sweep_check)
        form.addRow("Kaczmarz overlapping blocks", self.kaczmarz_overlap_check)
        form.addRow("Kaczmarz randomized blocks", self.kaczmarz_randomized_check)
        form.addRow("Kaczmarz shifted grid", self.kaczmarz_shift_grid_check)
        form.addRow("Kaczmarz smooth block window", self.kaczmarz_window_check)
        form.addRow("Kaczmarz stabilized sweep", self.kaczmarz_stabilized_check)
        form.addRow("Kaczmarz update damping", self.kaczmarz_damping_spin)
        form.addRow("Kaczmarz max update fraction", self.kaczmarz_max_update_spin)
        form.addRow("Rosen L exponent", self.rosen_l_spin)
        form.addRow("Rosen M exponent", self.rosen_m_spin)
        form.addRow("Rosen relax L/M toward 1", self.rosen_relax_check)
        form.addRow("Rosen relax factor", self.rosen_relax_factor_spin)
        form.addRow("Rosen preset", self.rosen_match_rl_button)
        form.addRow("Blind Gaussian initial PSF sigma", self.psf_sigma_spin)
        form.addRow("Blind effective PSF support", self.blind_psf_support_info_label)
        form.addRow("Non-negative", self.non_negative_check)
        form.addRow("Begin with Wiener filter", self.begin_with_wiener_check)
        form.addRow("Wiener noise PSD", self.wiener_use_noise_psd_check)
        form.addRow("TV preconditioning", self.tv_preconditioning_check)
        form.addRow("TV weight", self.tv_weight_spin)
        form.addRow("TV iterations", self.tv_iterations_spin)
        form.addRow("Denoiser type", self.denoiser_type_combo)
        form.addRow("Denoiser timing", self.neural_denoiser_mode_combo)
        form.addRow("Denoiser strength", self.neural_denoiser_strength_spin)
        form.addRow("Neural denoiser weights", weights_widget)
        form.addRow("PyTorch Adam image learning rate", self.torch_lr_spin)
        form.addRow("Blind Adam PSF learning rate", self.blind_psf_lr_spin)
        form.addRow("Blind Adam PSF TV weight", self.blind_psf_tv_weight_spin)
        form.addRow("Blind PSF rotational symmetry", self.blind_psf_rot_sym_check)
        form.addRow("Blind PSF initialization", self.blind_use_known_psf_init_check)
        form.addRow("Prefer CUDA", self.prefer_cuda_check)
        form.addRow("Torch numerical precision", self.torch_float64_check)
        form.addRow("Record every N torch iterations", self.torch_record_every_spin)
        form.addRow("PyTorch backend", self.torch_status_label)

        auto_box = QGroupBox("Auto optimization")
        auto_form = QFormLayout(auto_box)
        auto_form.addRow("Strategy", self.auto_strategy_combo)
        auto_form.addRow("Auto batch size", self.auto_batch_size_spin)
        auto_form.addRow("Max batch size", self.auto_max_batch_size_spin)
        auto_form.addRow("GPU batch policy", self.auto_batch_policy_combo)
        auto_form.addRow("GPU benchmark", self.auto_batch_detected_label)
        auto_form.addRow("Max candidates", self.auto_max_candidates_spin)
        auto_form.addRow("Numeric", self.auto_tune_numeric_check)
        auto_form.addRow("Boolean", self.auto_tune_boolean_check)
        auto_form.addRow("Categorical", self.auto_tune_categorical_check)
        auto_form.addRow("Denoiser type/timing", self.auto_tune_denoiser_check)
        auto_form.addRow("Denoiser strength", self.auto_tune_denoiser_strength_check)
        auto_form.addRow("TV", self.auto_tune_tv_check)
        auto_form.addRow("Wiener initialization", self.auto_tune_wiener_init_check)
        auto_buttons = QHBoxLayout()
        auto_buttons.addWidget(self.auto_button)
        auto_buttons.addWidget(self.auto_all_button)
        auto_buttons.addWidget(self.cancel_auto_button)
        auto_buttons_widget = QWidget()
        auto_buttons_widget.setLayout(auto_buttons)
        auto_form.addRow("Actions", auto_buttons_widget)
        auto_form.addRow("Status", self.auto_status)
        form.addRow(auto_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(box)
        layout.addWidget(scroll)
        self.auto_button.clicked.connect(self.auto_tune)
        self.auto_all_button.clicked.connect(self.auto_tune_all)
        self.cancel_auto_button.clicked.connect(self.request_cancel_auto)

        self._field_widgets: Dict[str, QWidget] = {
            "use_torch_batch": self.use_torch_batch_check,
            "K": self.k_spin,
            "wiener_k_scan_enabled": self.wiener_k_scan_check,
            "wiener_k_scan_min": self.wiener_k_scan_min_spin,
            "wiener_k_scan_max": self.wiener_k_scan_max_spin,
            "wiener_k_scan_points": self.wiener_k_scan_points_spin,
            "copy_wiener_settings_preset": self.copy_wiener_settings_button,
            "iterations": self.iter_spin,
            "epsilon": self.epsilon_spin,
            "step": self.landweber_step_spin,
            "kaczmarz_relaxation": self.kaczmarz_relaxation_spin,
            "kaczmarz_block_size": self.kaczmarz_block_size_spin,
            "kaczmarz_blocks_per_iteration": self.kaczmarz_blocks_per_iteration_spin,
            "use_torch_batch": self.use_torch_batch_check,
            "kaczmarz_full_sweep": self.kaczmarz_full_sweep_check,
            "kaczmarz_overlap": self.kaczmarz_overlap_check,
            "kaczmarz_randomized": self.kaczmarz_randomized_check,
            "kaczmarz_shift_grid": self.kaczmarz_shift_grid_check,
            "kaczmarz_window": self.kaczmarz_window_check,
            "kaczmarz_stabilized_sweep": self.kaczmarz_stabilized_check,
            "kaczmarz_update_damping": self.kaczmarz_damping_spin,
            "kaczmarz_max_update_fraction": self.kaczmarz_max_update_spin,
            "rosen_L": self.rosen_l_spin,
            "rosen_M": self.rosen_m_spin,
            "rosen_relax_to_one": self.rosen_relax_check,
            "rosen_relax_factor": self.rosen_relax_factor_spin,
            "rosen_match_rl_preset": self.rosen_match_rl_button,
            "psf_sigma": self.psf_sigma_spin,
            "blind_psf_support_info": self.blind_psf_support_info_label,
            "non_negative": self.non_negative_check,
            "begin_with_wiener": self.begin_with_wiener_check,
            "wiener_use_noise_psd": self.wiener_use_noise_psd_check,
            "use_tv_preconditioning": self.tv_preconditioning_check,
            "tv_weight": self.tv_weight_spin,
            "tv_iterations": self.tv_iterations_spin,
            "denoiser_type": self.denoiser_type_combo,
            "neural_denoiser_mode": self.neural_denoiser_mode_combo,
            "neural_denoiser_strength": self.neural_denoiser_strength_spin,
            "neural_denoiser_weights": self.neural_weights_widget,
            "torch_lr": self.torch_lr_spin,
            "blind_psf_lr": self.blind_psf_lr_spin,
            "blind_psf_tv_weight": self.blind_psf_tv_weight_spin,
            "blind_psf_rotational_symmetry": self.blind_psf_rot_sym_check,
            "blind_use_known_psf_init": self.blind_use_known_psf_init_check,
            "prefer_cuda": self.prefer_cuda_check,
            "torch_float64": self.torch_float64_check,
            "torch_record_every": self.torch_record_every_spin,
        }
        self._row_widgets: Dict[str, Tuple[Optional[QWidget], QWidget]] = {}
        for key, field in self._field_widgets.items():
            label = form.labelForField(field)
            self._row_widgets[key] = (label, field)

        self._previous_algorithm_name = self.combo.currentText()
        base_params = dict(self.params())
        for name in self.registry.names():
            alg_defaults = getattr(self.registry.get(name), "default_params", {})
            merged = dict(base_params)
            merged.update(alg_defaults)
            self._algorithm_param_values[name] = merged
        current_saved = self._algorithm_param_values.get(self._previous_algorithm_name)
        if current_saved:
            self._loading_algorithm_params = True
            try:
                self._apply_params_to_widgets(current_saved)
            finally:
                self._loading_algorithm_params = False
        self.combo.currentIndexChanged.connect(lambda _=None: self._on_algorithm_changed(self.combo.currentText()))
        self.blind_use_known_psf_init_check.toggled.connect(lambda _=None: self._update_psf_policy_label())
        for widget in self._field_widgets.values():
            self._connect_widget_change(widget)
        self.neural_denoiser_weights_edit.textChanged.connect(lambda _=None: self._remember_current_algorithm_params())
        self.denoiser_type_combo.currentTextChanged.connect(lambda _=None: self._update_visible_parameter_rows())
        self.neural_denoiser_mode_combo.currentTextChanged.connect(lambda _=None: self._update_visible_parameter_rows())
        self.tv_preconditioning_check.toggled.connect(lambda _=None: self._update_visible_parameter_rows())
        self.use_torch_batch_check.toggled.connect(lambda _=None: self._update_visible_parameter_rows())
        self.use_torch_batch_check.toggled.connect(lambda _=None: self._update_psf_policy_label())
        self._update_visible_parameter_rows()
        self._update_psf_policy_label()

    def _connect_widget_change(self, widget: QWidget) -> None:
        try:
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(lambda _=None: self._remember_current_algorithm_params())
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(lambda _=None: self._remember_current_algorithm_params())
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda _=None: self._remember_current_algorithm_params())
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(lambda _=None: self._remember_current_algorithm_params())
        except Exception:
            pass

    def _remember_current_algorithm_params(self) -> None:
        if self._loading_algorithm_params:
            return
        name = self.combo.currentText()
        if name:
            remembered = dict(self.params())
            self._algorithm_param_values[name] = remembered
            if name == "Wiener":
                self.state["wiener_profile_k"] = float(remembered.get("K", self.k_spin.value()))

    def _on_algorithm_changed(self, new_name: str) -> None:
        if self._previous_algorithm_name:
            self._algorithm_param_values[self._previous_algorithm_name] = dict(self.params())
        self._loading_algorithm_params = True
        try:
            saved = self._algorithm_param_values.get(new_name)
            if saved:
                self._apply_params_to_widgets(saved)
        finally:
            self._loading_algorithm_params = False
        self._previous_algorithm_name = new_name
        self._update_visible_parameter_rows()
        self._update_psf_policy_label()

    def _set_row_visible(self, key: str, visible: bool) -> None:
        row = self._row_widgets.get(key)
        if not row:
            return
        label, field = row
        if label is not None:
            label.setVisible(visible)
        field.setVisible(visible)

    def _update_visible_parameter_rows(self) -> None:
        alg_name = self.combo.currentText()
        active = set(self._active_parameter_names(alg_name))
        active.add("auto_tune_categorical")
        # Denoiser details are useful only when a denoiser is actually inserted.
        denoiser_mode = self.neural_denoiser_mode_combo.currentText()
        denoiser_type = self.denoiser_type_combo.currentText()
        if denoiser_mode == "Off":
            active.discard("denoiser_type")
            active.discard("neural_denoiser_strength")
            active.discard("neural_denoiser_weights")
        elif denoiser_type not in ("Neural CNN loaded from file", "DnCNN", "FFDNet", "DRUNet", "SCUNet", "Model zoo / custom manifest"):
            active.discard("neural_denoiser_weights")
        if not self.tv_preconditioning_check.isChecked() and denoiser_type != "TV only":
            active.discard("tv_weight")
            active.discard("tv_iterations")
        for key in self._field_widgets:
            self._set_row_visible(key, key in active)

    def update_known_psf_support_info(self, support: int) -> None:
        """Update the scalar support summary without overwriting the Tab-2 rectangle."""
        self.state["psf_support_extent"] = int(max(1, support))
        self._update_psf_policy_label()

    def _update_psf_policy_label(self) -> None:
        """Describe the one calculation PSF used by the selected method."""
        if not hasattr(self, "combo"):
            return
        alg_name = self.combo.currentText()
        selected_name = self._torch_auto_equivalent(alg_name) if self.use_torch_batch_check.isChecked() else alg_name
        is_blind = selected_name in {"Blind Richardson-Lucy", "PyTorch Blind Adam TV-MAP"}
        image = _calculation_image_from_state(self.state)
        image_shape = image.data.shape if isinstance(image, GrayImage) else tuple(self.state.get("calculation_image_shape", (256, 256)))
        calculation_psf = _synchronize_calculation_psf(self.state, image_shape)
        if isinstance(calculation_psf, PSF):
            kh, kw = calculation_psf.kernel.shape
            psf_desc = f"{kw} × {kh} px; sum={float(calculation_psf.kernel.sum()):.8g}"
        else:
            kw = int(self.state.get("psf_support_width", 1))
            kh = int(self.state.get("psf_support_height", 1))
            psf_desc = "not available"
        if is_blind:
            init_mode = (
                "initialized from the current calculation PSF"
                if self.blind_use_known_psf_init_check.isChecked() and isinstance(calculation_psf, PSF)
                else "initialized with a Gaussian"
            )
            self.psf_policy_label.setText(
                f"Blind method: estimated PSF array size is taken directly from the Tab 2 rectangle: "
                f"{kw} × {kh} px; {init_mode}. The estimate is normalized during optimization."
            )
            self.blind_psf_support_info_label.setText(
                f"Tab 2 controls the blind PSF width and height: {kw} × {kh} px."
            )
        elif selected_name in {"Wiener", "Torch batch Wiener"}:
            self.psf_policy_label.setText(
                f"Wiener uses only the current Tab 2 calculation PSF ({psf_desc}). "
                "The inverse is evaluated explicitly with FFT and IFFT using the circular-FFT Wiener model."
            )
            self.blind_psf_support_info_label.setText("")
        else:
            self.psf_policy_label.setText(
                f"Non-blind method uses only the current Tab 2 calculation PSF ({psf_desc}), "
                "after thresholding, rectangular cropping and unit-sum normalization."
            )
            self.blind_psf_support_info_label.setText("")

    def set_wiener_k_from_tab2(self, value: float) -> None:
        """Store a K value optimized jointly with the Tab-2 PSF floor."""
        kval = max(1e-12, float(value))
        self._remember_current_algorithm_params()
        profile = dict(self._algorithm_param_values.get("Wiener", {}))
        profile["K"] = kval
        self._algorithm_param_values["Wiener"] = profile
        self.state["wiener_profile_k"] = kval
        current = self.combo.currentText()
        current_profile = dict(self._algorithm_param_values.get(current, {}))
        if "K" in set(self._active_parameter_names(current, current_profile)):
            current_profile["K"] = kval
            self._algorithm_param_values[current] = current_profile
            old = self.k_spin.blockSignals(True)
            self.k_spin.setValue(kval)
            self.k_spin.blockSignals(old)
        self.auto_status.setText(
            f"Wiener K={kval:.6g} received from joint PSF-floor optimization in Tab 2. "
            "Use the Wiener-profile copy button to propagate it to all applicable methods."
        )

    def copy_wiener_settings_to_all(self) -> None:
        """Copy the saved plain-Wiener profile to every compatible algorithm.

        Only settings that describe an actual Wiener stage are propagated.  The
        full-loaded-array PSF mode remains exclusive to the plain Wiener filter,
        because iterative algorithms use a linear zero-boundary forward model.
        Whether an optional Wiener initializer is enabled is deliberately not
        changed.
        """
        if self._active_auto_threads_running():
            QMessageBox.information(
                self,
                "Auto cancellation in progress",
                "Wait until the current Auto worker has stopped before copying Wiener settings. "
                "Auto and this preset modify the same per-algorithm parameter profiles.",
            )
            return

        self._remember_current_algorithm_params()
        source = dict(self._algorithm_param_values.get("Wiener", {}))
        if self.combo.currentText() == "Wiener":
            source.update(self.params())
        keys = ("K", "wiener_use_noise_psd")
        copied_to: List[str] = []
        for alg_name in self.registry.names():
            target = dict(self._algorithm_param_values.get(alg_name, {}))
            active = set(self._active_parameter_names(alg_name, target))
            if "K" not in active:
                continue
            changed = False
            for key in keys:
                if key in active and key in source:
                    target[key] = source[key]
                    changed = True
            if changed:
                self._algorithm_param_values[alg_name] = target
                copied_to.append(alg_name)

        current_name = self.combo.currentText()
        if current_name in copied_to:
            self._loading_algorithm_params = True
            try:
                self._apply_params_to_widgets(self._algorithm_param_values[current_name])
            finally:
                self._loading_algorithm_params = False
            self._update_visible_parameter_rows()
            self._update_psf_policy_label()

        k_value = float(source.get("K", self.k_spin.value()))
        self.auto_status.setText(
            f"Wiener profile copied to {len(copied_to)} applicable algorithm profiles: K={k_value:.6g}. "
            "Full-array PSF mode and Begin with Wiener were not changed."
        )

    def set_rosen_from_richardson_lucy(self) -> None:
        """Copy the ordinary RL profile and set the Rosen exponents to one."""
        if self._active_auto_threads_running():
            QMessageBox.information(
                self,
                "Auto cancellation in progress",
                "Wait until the current Auto worker has stopped before applying the Rosen preset. "
                "The preset and Auto modify the same per-algorithm parameter profile.",
            )
            return
        rl_params = dict(self._algorithm_param_values.get("Richardson-Lucy", {}))
        current = dict(self.params())
        common_keys = {
            "use_torch_batch", "K", "iterations", "epsilon", "non_negative",
            "begin_with_wiener",
            "use_tv_preconditioning", "tv_weight", "tv_iterations",
            "denoiser_type", "neural_denoiser_mode", "neural_denoiser_strength",
            "neural_denoiser_weights", "prefer_cuda", "torch_float64",
        }
        for key in common_keys:
            if key in rl_params:
                current[key] = rl_params[key]
        current["rosen_L"] = 1.0
        current["rosen_M"] = 1.0
        current["rosen_relax_to_one"] = False
        self._loading_algorithm_params = True
        try:
            self._apply_params_to_widgets(current)
        finally:
            self._loading_algorithm_params = False
        self._algorithm_param_values["Richardson-Lucy-Rosen"] = dict(self.params())
        self.auto_status.setText("Rosen preset applied: copied Richardson–Lucy settings; L=1, M=1; relaxation disabled.")
        self._update_visible_parameter_rows()

    def apply_resolution_linked_psf_defaults(self, support: int) -> None:
        """Retained compatibility hook; blind PSF dimensions now come from Tab 2."""
        self._resolution_default_psf_support = max(1, int(support))
        self.state["resolution_linked_psf_support"] = self._resolution_default_psf_support
        self._update_psf_policy_label()

    def browse_neural_weights(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open neural denoiser weights", "", "PyTorch weights (*.pt *.pth);;All files (*)")
        if path:
            self.neural_denoiser_weights_edit.setText(path)

    def settings(self) -> Dict[str, Any]:
        self._remember_current_algorithm_params()
        obsolete_profile_keys = {"use_exact_degradation_psf", "wiener_psf_mode", "wiener_absolute_output", "psf_size", "max_psf_width", "blind_psf_width", "blind_psf_height"}
        clean_profiles = {
            name: {k: v for k, v in dict(values).items() if k not in obsolete_profile_keys}
            for name, values in self._algorithm_param_values.items()
        }
        self._algorithm_param_values = clean_profiles
        data = self.params()
        for key in ("blind_psf_width", "blind_psf_height", "psf_size", "max_psf_width"):
            data.pop(key, None)
        data["algorithm"] = self.combo.currentText()
        data["algorithm_params_by_algorithm"] = clean_profiles
        data["auto_batch_cache"] = dict(getattr(self, "_auto_batch_cache", {}))
        return data

    def apply_settings(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        saved_by_algorithm = data.get("algorithm_params_by_algorithm")
        obsolete_profile_keys = {"use_exact_degradation_psf", "wiener_psf_mode", "wiener_absolute_output", "psf_size", "max_psf_width", "blind_psf_width", "blind_psf_height"}
        if isinstance(saved_by_algorithm, dict):
            for name, params in saved_by_algorithm.items():
                if isinstance(params, dict):
                    clean = {k: v for k, v in params.items() if k not in obsolete_profile_keys}
                    self._algorithm_param_values[str(name)] = clean
        if "algorithm" in data:
            saved_alg = str(data.get("algorithm"))
            if self._is_torch_batch_algorithm(saved_alg):
                plain = self._plain_equivalent_for_torch_auto(saved_alg) or saved_alg
                idx = self.combo.findText(plain)
                if idx >= 0:
                    self.combo.setCurrentIndex(idx)
                    self.use_torch_batch_check.setChecked(True)
            else:
                idx = self.combo.findText(saved_alg)
                if idx >= 0:
                    self.combo.setCurrentIndex(idx)
        numeric_widgets = {
            "K": self.k_spin,
            "wiener_k_scan_min": self.wiener_k_scan_min_spin,
            "wiener_k_scan_max": self.wiener_k_scan_max_spin,
            "wiener_k_scan_points": self.wiener_k_scan_points_spin,
            "iterations": self.iter_spin,
            "epsilon": self.epsilon_spin,
            "step": self.landweber_step_spin,
            "kaczmarz_relaxation": self.kaczmarz_relaxation_spin,
            "kaczmarz_block_size": self.kaczmarz_block_size_spin,
            "kaczmarz_blocks_per_iteration": self.kaczmarz_blocks_per_iteration_spin,
            "kaczmarz_update_damping": self.kaczmarz_damping_spin,
            "kaczmarz_max_update_fraction": self.kaczmarz_max_update_spin,
            "rosen_L": self.rosen_l_spin,
            "rosen_M": self.rosen_m_spin,
            "rosen_relax_factor": self.rosen_relax_factor_spin,
            "psf_sigma": self.psf_sigma_spin,
            "tv_weight": self.tv_weight_spin,
            "tv_iterations": self.tv_iterations_spin,
            "neural_denoiser_strength": self.neural_denoiser_strength_spin,
            "torch_lr": self.torch_lr_spin,
            "blind_psf_lr": self.blind_psf_lr_spin,
            "blind_psf_tv_weight": self.blind_psf_tv_weight_spin,
            "torch_record_every": self.torch_record_every_spin,
            "auto_batch_size": self.auto_batch_size_spin,
            "auto_max_batch_size": self.auto_max_batch_size_spin,
            "auto_max_candidates": self.auto_max_candidates_spin,
        }
        for key, widget in numeric_widgets.items():
            if key not in data:
                continue
            if isinstance(widget, QSpinBox):
                widget.setValue(_safe_int(data.get(key), widget.value()))
            else:
                widget.setValue(_safe_float(data.get(key), widget.value()))
        checkboxes = {
            "use_torch_batch": self.use_torch_batch_check,
            "wiener_k_scan_enabled": self.wiener_k_scan_check,
            "kaczmarz_full_sweep": self.kaczmarz_full_sweep_check,
            "kaczmarz_overlap": self.kaczmarz_overlap_check,
            "kaczmarz_randomized": self.kaczmarz_randomized_check,
            "kaczmarz_shift_grid": self.kaczmarz_shift_grid_check,
            "kaczmarz_window": self.kaczmarz_window_check,
            "kaczmarz_stabilized_sweep": self.kaczmarz_stabilized_check,
            "rosen_relax_to_one": self.rosen_relax_check,
            "non_negative": self.non_negative_check,
            "begin_with_wiener": self.begin_with_wiener_check,
            "wiener_use_noise_psd": self.wiener_use_noise_psd_check,
            "use_tv_preconditioning": self.tv_preconditioning_check,
            "prefer_cuda": self.prefer_cuda_check,
            "blind_psf_rotational_symmetry": self.blind_psf_rot_sym_check,
            "torch_float64": self.torch_float64_check,
            "auto_tune_numeric": self.auto_tune_numeric_check,
            "auto_tune_boolean": self.auto_tune_boolean_check,
            "auto_tune_categorical": self.auto_tune_categorical_check,
            "auto_tune_denoiser": self.auto_tune_denoiser_check,
            "auto_tune_denoiser_strength": self.auto_tune_denoiser_strength_check,
            "auto_tune_tv": self.auto_tune_tv_check,
            "auto_tune_wiener_init": self.auto_tune_wiener_init_check,
        }
        for key, widget in checkboxes.items():
            if key in data:
                widget.setChecked(bool(data.get(key)))
        if "denoiser_type" in data:
            idx = self.denoiser_type_combo.findText(str(data.get("denoiser_type")))
            if idx >= 0:
                self.denoiser_type_combo.setCurrentIndex(idx)
        if "neural_denoiser_mode" in data:
            idx = self.neural_denoiser_mode_combo.findText(str(data.get("neural_denoiser_mode")))
            if idx >= 0:
                self.neural_denoiser_mode_combo.setCurrentIndex(idx)
        if "auto_batch_policy" in data:
            idx = self.auto_batch_policy_combo.findText(str(data.get("auto_batch_policy")))
            if idx >= 0:
                self.auto_batch_policy_combo.setCurrentIndex(idx)
        if "auto_strategy" in data:
            idx = self.auto_strategy_combo.findText(str(data.get("auto_strategy")))
            if idx >= 0:
                self.auto_strategy_combo.setCurrentIndex(idx)
        if isinstance(data.get("auto_batch_cache"), dict):
            self._auto_batch_cache = {str(k): int(v) for k, v in data.get("auto_batch_cache", {}).items() if str(v).isdigit() or isinstance(v, int)}
        if "neural_denoiser_weights" in data:
            self.neural_denoiser_weights_edit.setText(str(data.get("neural_denoiser_weights") or ""))
        self._algorithm_param_values[self.combo.currentText()] = dict(self.params())
        wiener_profile = self._algorithm_param_values.get("Wiener", {})
        self.state["wiener_profile_k"] = float(wiener_profile.get("K", data.get("K", self.k_spin.value())))
        self._update_visible_parameter_rows()

    def important_params_summary(self, params: Optional[Dict[str, Any]] = None) -> str:
        alg_name = self.combo.currentText()
        params = params or self.params()
        try:
            names = self._active_parameter_names(alg_name)
        except Exception:
            names = []
        keep = ["iterations", "K", "step", "non_negative", "begin_with_wiener", "use_tv_preconditioning"]
        for name in keep:
            if name in params and name not in names:
                names.append(name)
        parts = []
        for name in names:
            if name not in params:
                continue
            value = params[name]
            if isinstance(value, float):
                parts.append(f"{name}={value:.4g}")
            else:
                parts.append(f"{name}={value}")
        actual = self._torch_auto_equivalent(alg_name) if bool(params.get("use_torch_batch", True)) else alg_name
        backend = "PyTorch batch" if actual != alg_name else "reference"
        return f"{alg_name} [{backend}]" + (" | " + ", ".join(parts[:12]) if parts else "")

    def selected_algorithm_name(self) -> str:
        """Return the actual implementation used by Run deconvolution."""
        selected = self.combo.currentText()
        params = self.params() if hasattr(self, "use_torch_batch_check") else {}
        if bool(params.get("use_torch_batch", True)):
            return self._torch_auto_equivalent(selected)
        return selected

    def selected_algorithm(self) -> DeconvolutionAlgorithm:
        return self.registry.get(self.selected_algorithm_name())

    def params(self) -> Dict[str, Any]:
        blind_w = max(1, int(self.state.get("psf_support_width", 1)))
        blind_h = max(1, int(self.state.get("psf_support_height", 1)))
        return {
            "use_torch_batch": self.use_torch_batch_check.isChecked(),
            "K": self.k_spin.value(),
            "wiener_k_scan_enabled": self.wiener_k_scan_check.isChecked(),
            "wiener_k_scan_min": self.wiener_k_scan_min_spin.value(),
            "wiener_k_scan_max": self.wiener_k_scan_max_spin.value(),
            "wiener_k_scan_points": self.wiener_k_scan_points_spin.value(),
            "iterations": self.iter_spin.value(),
            "epsilon": self.epsilon_spin.value(),
            "step": self.landweber_step_spin.value(),
            "kaczmarz_relaxation": self.kaczmarz_relaxation_spin.value(),
            "kaczmarz_block_size": self.kaczmarz_block_size_spin.value(),
            "kaczmarz_blocks_per_iteration": self.kaczmarz_blocks_per_iteration_spin.value(),
            "kaczmarz_full_sweep": self.kaczmarz_full_sweep_check.isChecked(),
            "kaczmarz_overlap": self.kaczmarz_overlap_check.isChecked(),
            "kaczmarz_randomized": self.kaczmarz_randomized_check.isChecked(),
            "kaczmarz_shift_grid": self.kaczmarz_shift_grid_check.isChecked(),
            "kaczmarz_window": self.kaczmarz_window_check.isChecked(),
            "kaczmarz_stabilized_sweep": self.kaczmarz_stabilized_check.isChecked(),
            "kaczmarz_update_damping": self.kaczmarz_damping_spin.value(),
            "kaczmarz_max_update_fraction": self.kaczmarz_max_update_spin.value(),
            "rosen_L": self.rosen_l_spin.value(),
            "rosen_M": self.rosen_m_spin.value(),
            "rosen_relax_to_one": self.rosen_relax_check.isChecked(),
            "rosen_relax_factor": self.rosen_relax_factor_spin.value(),
            "psf_sigma": self.psf_sigma_spin.value(),
            "blind_psf_width": blind_w,
            "blind_psf_height": blind_h,
            "non_negative": self.non_negative_check.isChecked(),
            "begin_with_wiener": self.begin_with_wiener_check.isChecked(),
            "wiener_use_noise_psd": self.wiener_use_noise_psd_check.isChecked(),
            "use_tv_preconditioning": self.tv_preconditioning_check.isChecked(),
            "tv_weight": self.tv_weight_spin.value(),
            "tv_iterations": self.tv_iterations_spin.value(),
            "denoiser_type": self.denoiser_type_combo.currentText(),
            "neural_denoiser_mode": self.neural_denoiser_mode_combo.currentText(),
            "neural_denoiser_strength": self.neural_denoiser_strength_spin.value(),
            "neural_denoiser_weights": self.neural_denoiser_weights_edit.text().strip(),
            "torch_lr": self.torch_lr_spin.value(),
            "blind_psf_lr": self.blind_psf_lr_spin.value(),
            "blind_psf_tv_weight": self.blind_psf_tv_weight_spin.value(),
            "blind_psf_rotational_symmetry": self.blind_psf_rot_sym_check.isChecked(),
            "blind_use_known_psf_init": self.blind_use_known_psf_init_check.isChecked(),
            "prefer_cuda": self.prefer_cuda_check.isChecked(),
            "torch_float64": self.torch_float64_check.isChecked(),
            "torch_record_every": self.torch_record_every_spin.value(),
            "auto_batch_size": self.auto_batch_size_spin.value(),
            "auto_max_batch_size": self.auto_max_batch_size_spin.value(),
            "auto_batch_policy": self.auto_batch_policy_combo.currentText(),
            "auto_strategy": self.auto_strategy_combo.currentText(),
            "auto_max_candidates": self.auto_max_candidates_spin.value(),
            "auto_tune_numeric": self.auto_tune_numeric_check.isChecked(),
            "auto_tune_boolean": self.auto_tune_boolean_check.isChecked(),
            "auto_tune_categorical": self.auto_tune_categorical_check.isChecked(),
            "auto_tune_denoiser": self.auto_tune_denoiser_check.isChecked(),
            "auto_tune_denoiser_strength": self.auto_tune_denoiser_strength_check.isChecked(),
            "auto_tune_tv": self.auto_tune_tv_check.isChecked(),
            "auto_tune_wiener_init": self.auto_tune_wiener_init_check.isChecked(),
        }

    def _auto_candidate_values(self, name: str, value: Any, params: Dict[str, Any]) -> List[Any]:
        """Return candidate values respecting the explicit Auto optimization options.

        This is the central gate that decides what Auto is allowed to change.
        Numeric, boolean, categorical, TV, denoiser, and Wiener-initialization
        choices are controlled independently by the visible Auto optimization
        checkboxes.  Technical execution controls and file paths are never tuned.
        """
        frozen_allowed = params.get("__auto_allowed_parameter_names")
        if isinstance(frozen_allowed, (tuple, list, set)) and name not in set(frozen_allowed):
            return []
        technical = {
            "auto_batch_size", "auto_max_batch_size", "auto_batch_policy", "auto_strategy", "auto_max_candidates", "prefer_cuda", "torch_float64",
            "torch_record_every", "neural_denoiser_weights",
            "wiener_k_scan_enabled", "wiener_k_scan_min", "wiener_k_scan_max", "wiener_k_scan_points",
            "auto_tune_numeric", "auto_tune_boolean", "auto_tune_categorical",
            "auto_tune_denoiser", "auto_tune_denoiser_strength", "auto_tune_tv",
            "auto_tune_wiener_init",
        }
        if name in technical:
            return []
        tv_names = {"use_tv_preconditioning", "tv_weight", "tv_iterations"}
        denoiser_names = {"neural_denoiser_mode", "denoiser_type", "neural_denoiser_strength"}
        wiener_init_names = {"begin_with_wiener"}

        tune_numeric = bool(params.get("auto_tune_numeric", True))
        tune_boolean = bool(params.get("auto_tune_boolean", False))
        tune_categorical = bool(params.get("auto_tune_categorical", False))
        tune_denoiser = bool(params.get("auto_tune_denoiser", False))
        tune_denoiser_strength = bool(params.get("auto_tune_denoiser_strength", True))
        tune_tv = bool(params.get("auto_tune_tv", True))
        tune_wiener_init = bool(params.get("auto_tune_wiener_init", False))

        if name in tv_names and not tune_tv:
            return []
        if name in wiener_init_names and not tune_wiener_init:
            return []
        if name in {"K", "wiener_use_noise_psd"} and not tune_numeric and not tune_wiener_init:
            return []
        if name in denoiser_names:
            if name == "neural_denoiser_strength":
                if not (tune_denoiser or tune_denoiser_strength):
                    return []
            elif not (tune_denoiser or tune_categorical):
                return []

        if isinstance(value, bool):
            allow_bool = tune_boolean or (name in tv_names and tune_tv and tune_boolean) or (name in wiener_init_names and tune_wiener_init)
            return self._candidate_values(name, value, allow_bool)
        if isinstance(value, str):
            allow_cat = tune_categorical or (name in denoiser_names and tune_denoiser)
            return self._candidate_values(name, value, allow_cat)
        if not tune_numeric and name not in tv_names and name not in denoiser_names and name not in {"K"}:
            return []
        return self._candidate_values(name, value, tune_categorical or tune_denoiser)


    def _candidate_values(self, name: str, value: Any, tune_categorical: bool = False) -> List[Any]:
        """Return local Auto candidates.

        Numeric parameters are searched within about +/-50%.  Categorical
        parameters and checkboxes are changed only when the user enables
        "Auto tune categorical options".  This keeps default Auto fast and
        avoids silently switching qualitative choices such as the denoiser.
        """
        if value is None:
            return []
        if isinstance(value, bool):
            return [value, not value] if tune_categorical else [value]
        if isinstance(value, str):
            if not tune_categorical:
                return [value]
            if name == "neural_denoiser_mode":
                vals = ["Off", "Before algorithm", "After each iteration"]
                return vals if value in vals else [value] + vals
            if name == "denoiser_type":
                # Safe denoisers do not require external weights or heavy custom models.
                vals = ["TV only", "Gaussian", "Bilateral", "Non-local Means", "Wavelet"]
                if value not in vals:
                    vals = [value] + vals
                return vals
            return [value]
        if name in {"iterations", "psf_size", "max_psf_width", "kaczmarz_block_size", "kaczmarz_blocks_per_iteration", "tv_iterations", "torch_record_every"}:
            vals = sorted({max(1, int(round(float(value) * f))) for f in (0.5, 0.75, 1.0, 1.25, 1.5)})
            if name in {"psf_size", "max_psf_width"}:
                vals = [v + 1 if v % 2 == 0 else v for v in vals]
                vals = [max(3, v) for v in vals]
            return vals
        try:
            v = float(value)
        except (TypeError, ValueError):
            return [value]
        if v == 0.0:
            return [0.0]
        return [v * f for f in (0.5, 0.75, 1.0, 1.25, 1.5)]

    def _wiener_k_bounds(self) -> Tuple[float, float]:
        # Fixed numerical bounds keep background Auto independent of Qt widgets.
        return 1e-12, 1e4

    @staticmethod
    def _unique_float_values(values: List[float], lo: float, hi: float) -> List[float]:
        result: List[float] = []
        for value in values:
            v = float(np.clip(float(value), lo, hi))
            if not np.isfinite(v) or v <= 0:
                continue
            if not any(abs(np.log10(v) - np.log10(old)) < 1e-10 for old in result):
                result.append(v)
        return sorted(result)

    def _wiener_k_coarse_values(self, current: Any) -> List[float]:
        """Return a broad logarithmic K grid spanning the allowed range.

        The grid contains one and sqrt(10) times each decade.  This samples the
        full range densely enough to locate the correct order of magnitude while
        keeping the number of expensive reconstruction trials moderate.
        """
        lo, hi = self._wiener_k_bounds()
        try:
            current_value = float(current)
        except (TypeError, ValueError):
            current_value = 0.01
        e0 = int(np.floor(np.log10(lo)))
        e1 = int(np.ceil(np.log10(hi)))
        values: List[float] = [current_value]
        for exponent in range(e0, e1 + 1):
            base = 10.0 ** exponent
            values.extend([base, np.sqrt(10.0) * base])
        return self._unique_float_values(values, lo, hi)

    def _wiener_k_fine_values(self, center: float, half_span_decades: float, count: int) -> List[float]:
        lo, hi = self._wiener_k_bounds()
        center = float(np.clip(center, lo, hi))
        factors = 10.0 ** np.linspace(-abs(float(half_span_decades)), abs(float(half_span_decades)), max(3, int(count)))
        return self._unique_float_values([center * float(f) for f in factors] + [center], lo, hi)

    def _auto_optimize_wiener_k_for_algorithm(
        self,
        alg_name: str,
        params: Dict[str, Any],
        initial_score: float,
    ) -> Tuple[Dict[str, Any], float, int]:
        """Tune Wiener K in log space, then refine it locally.

        This dedicated stage is used for every algorithm that exposes K,
        including algorithms that use a Wiener filter only for initialization or
        preconditioning.  A broad logarithmic scan first locates the correct
        order of magnitude.  Two progressively narrower logarithmic scans then
        provide precise tuning around the best value.
        """
        if "K" not in self._active_parameter_names(alg_name, params):
            return dict(params), float(initial_score), 0
        if not self._auto_candidate_values("K", params.get("K"), params):
            return dict(params), float(initial_score), 0
        direct_k_algorithms = {
            "Wiener", "Torch batch Wiener",
            "Richardson-Lucy-Wiener", "Torch batch Richardson-Lucy-Wiener",
            "Landweber Wiener-preconditioned",
        }
        if alg_name not in direct_k_algorithms and not bool(params.get("begin_with_wiener", False)):
            return dict(params), float(initial_score), 0

        best_params = dict(params)
        best_score = float(initial_score)
        tested = 0

        def evaluate(values: List[float]) -> None:
            nonlocal best_params, best_score, tested
            if getattr(self, "_auto_cancel_requested", False):
                return
            current_k = float(best_params.get("K", 0.01))
            trials: List[Dict[str, Any]] = []
            for value in values:
                if abs(np.log10(float(value)) - np.log10(max(current_k, 1e-300))) < 1e-12:
                    continue
                trial = dict(best_params)
                trial["K"] = float(value)
                trials.append(trial)
            if not trials:
                return
            scores = self._score_params_batch_for_algorithm(alg_name, trials)
            tested += len(trials)
            for trial, score in zip(trials, scores):
                score = float(score)
                if np.isfinite(score) and (not np.isfinite(best_score) or score > best_score):
                    best_score = score
                    best_params = trial

        evaluate(self._wiener_k_coarse_values(best_params.get("K", 0.01)))
        if not getattr(self, "_auto_cancel_requested", False):
            evaluate(self._wiener_k_fine_values(float(best_params.get("K", 0.01)), 0.50, 11))
        if not getattr(self, "_auto_cancel_requested", False):
            evaluate(self._wiener_k_fine_values(float(best_params.get("K", 0.01)), 0.12, 9))
        return best_params, best_score, tested

    def _active_parameter_names(self, alg_name: str, params: Optional[Dict[str, Any]] = None) -> List[str]:
        torch_backend_controls = ["use_torch_batch"]
        use_torch_batch = bool((params or {}).get("use_torch_batch", True))
        if self._has_torch_batch_equivalent(alg_name) and use_torch_batch:
            torch_backend_controls += ["prefer_cuda", "torch_float64", "auto_batch_size", "auto_max_batch_size", "auto_batch_policy"]
        denoiser = ["neural_denoiser_mode", "denoiser_type", "neural_denoiser_strength", "neural_denoiser_weights"]
        tv = ["use_tv_preconditioning", "tv_weight", "tv_iterations"]
        wiener_controls = ["wiener_use_noise_psd"]
        common_iter = ["iterations", "epsilon", "begin_with_wiener", "K", "non_negative"] + tv + denoiser
        mapping = {
            "Wiener": ["K", "wiener_k_scan_enabled", "wiener_k_scan_min", "wiener_k_scan_max", "wiener_k_scan_points", "copy_wiener_settings_preset", "non_negative"] + wiener_controls + tv + denoiser,
            "Richardson-Lucy": common_iter + wiener_controls,
            "Richardson-Lucy-Wiener": ["iterations", "epsilon", "K", "begin_with_wiener", "non_negative"] + wiener_controls + tv + denoiser,
            "Blind Richardson-Lucy": ["iterations", "epsilon", "psf_sigma", "blind_psf_support_info", "blind_psf_rotational_symmetry", "blind_use_known_psf_init", "begin_with_wiener", "K", "non_negative"] + wiener_controls + tv + denoiser,
            "Landweber": ["iterations", "step", "begin_with_wiener", "K", "non_negative"] + wiener_controls + tv + denoiser,
            "Landweber Wiener-preconditioned": ["iterations", "step", "K", "begin_with_wiener", "non_negative"] + wiener_controls + tv + denoiser,
            "Block Kaczmarz": ["iterations", "kaczmarz_relaxation", "kaczmarz_block_size", "kaczmarz_blocks_per_iteration", "kaczmarz_full_sweep", "kaczmarz_overlap", "kaczmarz_randomized", "kaczmarz_shift_grid", "kaczmarz_window", "kaczmarz_stabilized_sweep", "kaczmarz_update_damping", "kaczmarz_max_update_fraction", "begin_with_wiener", "K", "non_negative"] + wiener_controls + tv + denoiser,
            "Richardson-Lucy-Rosen": ["iterations", "epsilon", "K", "rosen_L", "rosen_M", "rosen_relax_to_one", "rosen_relax_factor", "rosen_match_rl_preset", "begin_with_wiener", "non_negative"] + wiener_controls + tv + denoiser,
            "PyTorch Adam TV-MAP": ["iterations", "torch_lr", "tv_weight", "begin_with_wiener", "K", "non_negative", "prefer_cuda", "torch_float64", "torch_record_every", "auto_batch_size", "auto_max_batch_size", "auto_batch_policy"] + wiener_controls + denoiser,
            "PyTorch Blind Adam TV-MAP": ["iterations", "torch_lr", "blind_psf_lr", "tv_weight", "blind_psf_tv_weight", "psf_sigma", "blind_psf_support_info", "blind_psf_rotational_symmetry", "blind_use_known_psf_init", "begin_with_wiener", "K", "non_negative", "prefer_cuda", "torch_float64", "torch_record_every", "auto_batch_size", "auto_max_batch_size", "auto_batch_policy"] + wiener_controls + denoiser,
            "Torch batch Wiener": ["K", "wiener_k_scan_enabled", "wiener_k_scan_min", "wiener_k_scan_max", "wiener_k_scan_points", "non_negative", "prefer_cuda", "torch_float64", "auto_batch_size", "auto_max_batch_size", "auto_batch_policy"] + wiener_controls,
            "Torch batch Richardson-Lucy": ["iterations", "epsilon", "begin_with_wiener", "K", "non_negative", "prefer_cuda", "torch_float64", "auto_batch_size", "auto_max_batch_size", "auto_batch_policy"] + wiener_controls + denoiser,
            "Torch batch Richardson-Lucy-Wiener": ["iterations", "epsilon", "K", "begin_with_wiener", "non_negative", "prefer_cuda", "torch_float64", "auto_batch_size", "auto_max_batch_size", "auto_batch_policy"] + wiener_controls + denoiser,
            "Torch batch Richardson-Lucy-Rosen": ["iterations", "epsilon", "K", "rosen_L", "rosen_M", "rosen_relax_to_one", "rosen_relax_factor", "begin_with_wiener", "non_negative", "prefer_cuda", "torch_float64", "auto_batch_size", "auto_max_batch_size", "auto_batch_policy"] + wiener_controls + tv + denoiser,
            "Torch batch Landweber": ["iterations", "step", "K", "begin_with_wiener", "non_negative", "prefer_cuda", "torch_float64", "auto_batch_size", "auto_max_batch_size", "auto_batch_policy"] + wiener_controls + denoiser,
        }
        names = list(mapping.get(alg_name, common_iter))
        if self._has_torch_batch_equivalent(alg_name):
            for key in torch_backend_controls:
                if key not in names:
                    names.append(key)
        return names

    def _calculation_psf(
        self,
        psf: Optional[PSF],
        image_shape: Tuple[int, int],
        params: Dict[str, Any],
        algorithm_name: Optional[str] = None,
    ) -> Optional[PSF]:
        """Return the single current Tab-2 calculation PSF.

        There is deliberately no stored degradation/paired-snapshot branch.
        The numerical kernel is always rebuilt from the thresholded full PSF,
        current rectangular selection and unit-sum normalization visible in
        Tabs 1 and 2.
        """
        selected_name = str(algorithm_name or params.get("__auto_selected_algorithm") or params.get("_algorithm_name") or "")
        if not selected_name:
            try:
                selected_name = self.selected_algorithm_name()
            except Exception:
                selected_name = ""
        is_blind = selected_name in {"Blind Richardson-Lucy", "PyTorch Blind Adam TV-MAP"}
        if is_blind:
            blind_w = max(1, int(self.state.get("psf_support_width", 1)))
            blind_h = max(1, int(self.state.get("psf_support_height", 1)))
            params["blind_psf_width"] = blind_w
            params["blind_psf_height"] = blind_h
            # Compatibility scalars cannot override the rectangular Tab-2 size.
            params["psf_size"] = max(blind_w, blind_h)
            params["max_psf_width"] = max(blind_w, blind_h)
        model = CIRCULAR_FFT if selected_name in {"Wiener", "Torch batch Wiener"} else LINEAR_SAME
        current = _synchronize_calculation_psf(self.state, image_shape, algorithm_model=model)
        if current is None and isinstance(psf, PSF):
            current = calculation_psf_for_image(psf, image_shape, algorithm_convolution_model=model)
            if current is not None:
                self.state["calculation_psf"] = current
        if is_blind and not bool(params.get("blind_use_known_psf_init", True)):
            return None
        if current is None:
            return None
        meta = dict(current.metadata or {})
        meta.update({
            "algorithm_convolution_model": model,
            "wiener_full_frame_psf": False,
            "wiener_kernel_source": "current_tab2_calculation_psf",
        })
        return PSF(current.kernel.copy(), name=current.name, raw_kernel=current.raw_kernel, metadata=meta)

    @staticmethod
    def _quality_psf_from_frame(frame: GrayImage, fallback: Optional[PSF]) -> Optional[PSF]:
        if frame is not None and isinstance(getattr(frame, "metadata", None), dict):
            estimated = frame.metadata.get("estimated_psf")
            if estimated is not None:
                try:
                    return PSF(np.asarray(estimated, dtype=np.float64), name="estimated_psf_for_quality")
                except Exception:
                    pass
        return fallback

    def _reconstruction_metrics(
        self,
        reference: Optional[GrayImage],
        degraded: Optional[GrayImage],
        frame: GrayImage,
        fallback_psf: Optional[PSF],
    ) -> Dict[str, float]:
        allow_ref = reference_metrics_available(self.state)
        quality_psf = self._quality_psf_from_frame(frame, fallback_psf)
        return compute_metrics(
            reference,
            frame,
            allow_reference_metrics=allow_ref,
            roi_source=degraded,
            measured=degraded,
            psf=quality_psf,
        )

    def _plain_wiener_gcv_score(
        self,
        alg_name: str,
        degraded: Optional[GrayImage],
        run_psf: Optional[PSF],
        params: Dict[str, Any],
    ) -> Optional[float]:
        """Return a score to maximize for no-reference plain Wiener Auto.

        The generic image-domain no-reference cost contains a smoothness term
        and can therefore prefer an over-regularized, edge-free Wiener result.
        Plain Wiener K is instead selected with generalized cross-validation
        directly from the measured data and transfer function.
        """
        if alg_name not in {"Wiener", "Torch batch Wiener"}:
            return None
        if reference_metrics_available(self.state) or degraded is None or run_psf is None:
            return None
        try:
            noise_psd = normalized_noise_psd_from_image(degraded, params)
            value = wiener_gcv_cost(degraded.data, run_psf, float(params.get("K", 0.01)), noise_psd=noise_psd)
        except Exception:
            return float("-inf")
        return float(-value) if np.isfinite(value) else float("-inf")

    def _auto_score_description(self, alg_name: str, score: float) -> str:
        if alg_name in {"Wiener", "Torch batch Wiener"} and not reference_metrics_available(self.state):
            return f"lowest Wiener GCV {-score:.6g}" if np.isfinite(score) else "no valid Wiener GCV"
        return score_description(score, reference_metrics_available(self.state))

    def _score_params(self, params: Dict[str, Any]) -> float:
        reference: Optional[GrayImage] = self.state.get("image")
        degraded: Optional[GrayImage] = self.state.get("degraded")
        psf: Optional[PSF] = self.state.get("psf")
        alg = self.selected_algorithm()
        is_blind = isinstance(alg, (BlindRichardsonLucyDeconvolution, TorchBlindAdamTVMAPDeconvolution))
        if degraded is None:
            if reference is None or psf is None:
                return float("-inf")
            degradation_psf = _synchronize_calculation_psf(self.state, reference.data.shape) or calculation_psf_for_image(psf, reference.data.shape)
            degraded = degrade_image(reference, degradation_psf, noise_sigma=0.01)
            self.state["degraded"] = degraded
            self.state["degradation_psf"] = degradation_psf
        run_psf = self._calculation_psf(psf, degraded.data.shape, params, algorithm_name=str(getattr(alg, "name", "")))
        if run_psf is None and not is_blind:
            return float("-inf")
        gcv_score = self._plain_wiener_gcv_score(str(getattr(alg, "name", "")), degraded, run_psf, params)
        if gcv_score is not None:
            return float(gcv_score)
        try:
            safe_params = dict(params)
            if isinstance(alg, (TorchAdamTVMAPDeconvolution, TorchBlindAdamTVMAPDeconvolution)):
                # Auto tuning may call the algorithm many times; keep each trial short.
                safe_params["iterations"] = min(int(safe_params.get("iterations", 100)), 25)
            result = alg.run(degraded, run_psf, **safe_params)
        except Exception:
            return float("-inf")
        frames = result.history or [result.image]
        allow_ref = reference_metrics_available(self.state)
        scores = [metric_score(self._reconstruction_metrics(reference, degraded, frame, run_psf)) for frame in frames]
        return float(max(scores)) if scores else float("-inf")

    def _apply_params_to_widgets(self, params: Dict[str, Any]) -> None:
        if "use_torch_batch" in params: self.use_torch_batch_check.setChecked(bool(params["use_torch_batch"]))
        if "K" in params: self.k_spin.setValue(float(params["K"]))
        if "wiener_k_scan_enabled" in params: self.wiener_k_scan_check.setChecked(bool(params["wiener_k_scan_enabled"]))
        if "wiener_k_scan_min" in params: self.wiener_k_scan_min_spin.setValue(float(params["wiener_k_scan_min"]))
        if "wiener_k_scan_max" in params: self.wiener_k_scan_max_spin.setValue(float(params["wiener_k_scan_max"]))
        if "wiener_k_scan_points" in params: self.wiener_k_scan_points_spin.setValue(int(params["wiener_k_scan_points"]))
        if "iterations" in params: self.iter_spin.setValue(int(params["iterations"]))
        if "epsilon" in params: self.epsilon_spin.setValue(float(params["epsilon"]))
        if "step" in params: self.landweber_step_spin.setValue(float(params["step"]))
        if "kaczmarz_relaxation" in params: self.kaczmarz_relaxation_spin.setValue(float(params["kaczmarz_relaxation"]))
        if "kaczmarz_block_size" in params: self.kaczmarz_block_size_spin.setValue(int(params["kaczmarz_block_size"]))
        if "kaczmarz_blocks_per_iteration" in params: self.kaczmarz_blocks_per_iteration_spin.setValue(int(params["kaczmarz_blocks_per_iteration"]))
        if "kaczmarz_full_sweep" in params: self.kaczmarz_full_sweep_check.setChecked(bool(params["kaczmarz_full_sweep"]))
        if "kaczmarz_overlap" in params: self.kaczmarz_overlap_check.setChecked(bool(params["kaczmarz_overlap"]))
        if "kaczmarz_randomized" in params: self.kaczmarz_randomized_check.setChecked(bool(params["kaczmarz_randomized"]))
        if "kaczmarz_shift_grid" in params: self.kaczmarz_shift_grid_check.setChecked(bool(params["kaczmarz_shift_grid"]))
        if "kaczmarz_window" in params: self.kaczmarz_window_check.setChecked(bool(params["kaczmarz_window"]))
        if "kaczmarz_stabilized_sweep" in params: self.kaczmarz_stabilized_check.setChecked(bool(params["kaczmarz_stabilized_sweep"]))
        if "kaczmarz_update_damping" in params: self.kaczmarz_damping_spin.setValue(float(params["kaczmarz_update_damping"]))
        if "kaczmarz_max_update_fraction" in params: self.kaczmarz_max_update_spin.setValue(float(params["kaczmarz_max_update_fraction"]))
        if "rosen_L" in params: self.rosen_l_spin.setValue(float(params["rosen_L"]))
        if "rosen_M" in params: self.rosen_m_spin.setValue(float(params["rosen_M"]))
        if "rosen_relax_to_one" in params: self.rosen_relax_check.setChecked(bool(params["rosen_relax_to_one"]))
        if "rosen_relax_factor" in params: self.rosen_relax_factor_spin.setValue(float(params["rosen_relax_factor"]))
        if "psf_sigma" in params: self.psf_sigma_spin.setValue(float(params["psf_sigma"]))
        if "non_negative" in params: self.non_negative_check.setChecked(bool(params["non_negative"]))
        if "begin_with_wiener" in params: self.begin_with_wiener_check.setChecked(bool(params["begin_with_wiener"]))
        if "wiener_use_noise_psd" in params: self.wiener_use_noise_psd_check.setChecked(bool(params["wiener_use_noise_psd"]))
        if "use_tv_preconditioning" in params: self.tv_preconditioning_check.setChecked(bool(params["use_tv_preconditioning"]))
        if "tv_weight" in params: self.tv_weight_spin.setValue(float(params["tv_weight"]))
        if "tv_iterations" in params: self.tv_iterations_spin.setValue(int(params["tv_iterations"]))
        if "denoiser_type" in params:
            idx = self.denoiser_type_combo.findText(str(params["denoiser_type"]))
            if idx >= 0: self.denoiser_type_combo.setCurrentIndex(idx)
        if "neural_denoiser_mode" in params:
            idx = self.neural_denoiser_mode_combo.findText(str(params["neural_denoiser_mode"]))
            if idx >= 0: self.neural_denoiser_mode_combo.setCurrentIndex(idx)
        if "neural_denoiser_strength" in params: self.neural_denoiser_strength_spin.setValue(float(params["neural_denoiser_strength"]))
        if "neural_denoiser_weights" in params: self.neural_denoiser_weights_edit.setText(str(params.get("neural_denoiser_weights") or ""))
        if "torch_lr" in params: self.torch_lr_spin.setValue(float(params["torch_lr"]))
        if "blind_psf_lr" in params: self.blind_psf_lr_spin.setValue(float(params["blind_psf_lr"]))
        if "blind_psf_tv_weight" in params: self.blind_psf_tv_weight_spin.setValue(float(params["blind_psf_tv_weight"]))
        if "blind_psf_rotational_symmetry" in params: self.blind_psf_rot_sym_check.setChecked(bool(params["blind_psf_rotational_symmetry"]))
        if "blind_use_known_psf_init" in params: self.blind_use_known_psf_init_check.setChecked(bool(params["blind_use_known_psf_init"]))
        if "prefer_cuda" in params: self.prefer_cuda_check.setChecked(bool(params["prefer_cuda"]))
        if "torch_float64" in params: self.torch_float64_check.setChecked(bool(params["torch_float64"]))
        if "torch_record_every" in params: self.torch_record_every_spin.setValue(int(params["torch_record_every"]))
        if "auto_batch_size" in params: self.auto_batch_size_spin.setValue(int(params["auto_batch_size"]))
        if "auto_max_batch_size" in params: self.auto_max_batch_size_spin.setValue(int(params["auto_max_batch_size"]))
        if "auto_batch_policy" in params:
            idx = self.auto_batch_policy_combo.findText(str(params["auto_batch_policy"]))
            if idx >= 0: self.auto_batch_policy_combo.setCurrentIndex(idx)
        if "auto_strategy" in params:
            idx = self.auto_strategy_combo.findText(str(params["auto_strategy"]))
            if idx >= 0: self.auto_strategy_combo.setCurrentIndex(idx)
        if "auto_max_candidates" in params: self.auto_max_candidates_spin.setValue(int(params["auto_max_candidates"]))
        if "auto_tune_numeric" in params: self.auto_tune_numeric_check.setChecked(bool(params["auto_tune_numeric"]))
        if "auto_tune_boolean" in params: self.auto_tune_boolean_check.setChecked(bool(params["auto_tune_boolean"]))
        if "auto_tune_categorical" in params: self.auto_tune_categorical_check.setChecked(bool(params["auto_tune_categorical"]))
        if "auto_tune_denoiser" in params: self.auto_tune_denoiser_check.setChecked(bool(params["auto_tune_denoiser"]))
        if "auto_tune_denoiser_strength" in params: self.auto_tune_denoiser_strength_check.setChecked(bool(params["auto_tune_denoiser_strength"]))
        if "auto_tune_tv" in params: self.auto_tune_tv_check.setChecked(bool(params["auto_tune_tv"]))
        if "auto_tune_wiener_init" in params: self.auto_tune_wiener_init_check.setChecked(bool(params["auto_tune_wiener_init"]))

    def _score_params_batch(self, candidates: List[Dict[str, Any]]) -> List[float]:
        """Score candidates in one PyTorch batch when the selected algorithm supports it."""
        if not candidates:
            return []
        reference: Optional[GrayImage] = self.state.get("image")
        degraded: Optional[GrayImage] = self.state.get("degraded")
        psf: Optional[PSF] = self.state.get("psf")
        alg = self.selected_algorithm()
        is_blind = isinstance(alg, (BlindRichardsonLucyDeconvolution, TorchBlindAdamTVMAPDeconvolution))
        if not getattr(alg, "supports_batched_auto", False):
            return [self._score_params(p) for p in candidates]
        if degraded is None:
            if reference is None or psf is None:
                return [float("-inf")] * len(candidates)
            degradation_psf = _synchronize_calculation_psf(self.state, reference.data.shape) or calculation_psf_for_image(psf, reference.data.shape)
            degraded = degrade_image(reference, degradation_psf, noise_sigma=0.01)
            self.state["degraded"] = degraded
            self.state["degradation_psf"] = degradation_psf
        run_psf = self._calculation_psf(psf, degraded.data.shape, candidates[0], algorithm_name=str(getattr(alg, "name", "")))
        if run_psf is None and not is_blind:
            return [float("-inf")] * len(candidates)
        alg_name = str(getattr(alg, "name", ""))
        if alg_name in {"Wiener", "Torch batch Wiener"} and not reference_metrics_available(self.state):
            return [
                float(self._plain_wiener_gcv_score(alg_name, degraded, run_psf, candidate) or float("-inf"))
                for candidate in candidates
            ]
        try:
            if not reference_metrics_available(self.state):
                batched = alg.run_batch(degraded, run_psf, candidates, reference=None, keep_history=False)
                return [
                    metric_score(
                        self._reconstruction_metrics(
                            None,
                            degraded,
                            GrayImage(np.asarray(arr.get("image") if isinstance(arr, dict) else arr), name="candidate"),
                            PSF(np.asarray(arr.get("estimated_psf"), dtype=np.float64), name="candidate_estimated_psf")
                            if isinstance(arr, dict) and arr.get("estimated_psf") is not None else run_psf,
                        )
                    )
                    for arr in batched.infos
                ]
            return list(alg.score_batch(reference, degraded, run_psf, candidates))
        except Exception as exc:
            # Fall back to scalar scoring so Auto stays usable if a particular
            # Torch/CUDA installation cannot execute the batched path.
            return [self._score_params(p) for p in candidates]

    def _full_candidate_pool(self, base_params: Dict[str, Any], names: List[str]) -> List[Dict[str, Any]]:
        """Build the full local Cartesian product of Auto candidates.

        For Torch-batched algorithms this replaces coordinate search.  All
        candidate settings within +/-50% of the current numeric values are
        generated first, then scored in batches.  This exposes much more work
        to CUDA/FFT kernels and makes the Auto timing easier to interpret.
        """
        # These are execution/UI controls, not algorithmic hyperparameters.
        # Tuning them creates a huge Cartesian product without improving the reconstruction
        # and was the main reason Auto could start very long background jobs.
        skip_names = {
            "auto_batch_size", "auto_max_batch_size", "auto_batch_policy", "auto_strategy", "auto_max_candidates", "prefer_cuda", "torch_float64", "torch_record_every",
            "neural_denoiser_weights", "auto_tune_numeric",
            "auto_tune_boolean", "auto_tune_categorical", "auto_tune_denoiser",
            "auto_tune_denoiser_strength", "auto_tune_tv", "auto_tune_wiener_init",
            "wiener_k_scan_enabled", "wiener_k_scan_min", "wiener_k_scan_max", "wiener_k_scan_points",
            "__auto_selected_algorithm"
        }
        value_lists: List[List[Any]] = []
        active_names: List[str] = []
        for name in names:
            if name in skip_names:
                continue
            vals = self._auto_candidate_values(name, base_params.get(name), base_params)
            # Text/path values do not create meaningful alternatives. Keeping
            # a single value is fine, but including many such fields would only
            # duplicate candidates.
            if not vals:
                continue
            unique_vals: List[Any] = []
            for v in vals:
                if v not in unique_vals:
                    unique_vals.append(v)
            active_names.append(name)
            value_lists.append(unique_vals)

        if not value_lists:
            return [dict(base_params)]

        candidates: List[Dict[str, Any]] = []
        seen = set()
        max_candidates = int(base_params.get("auto_max_candidates", 256))
        max_candidates = max(16, min(max_candidates, 4096))
        for combo_index, combo in enumerate(itertools.product(*value_lists)):
            if getattr(self, "_auto_cancel_requested", False):
                break
            if len(candidates) >= max_candidates:
                break
            trial = dict(base_params)
            for name, value in zip(active_names, combo):
                trial[name] = value
            if int(trial.get("psf_size", 3)) > int(trial.get("max_psf_width", trial.get("psf_size", 3))):
                continue
            # Prefer a compact hashable signature so duplicate candidates from
            # rounded integer parameters are removed before scoring.
            sig = tuple((k, trial.get(k)) for k in active_names)
            if sig in seen:
                continue
            seen.add(sig)
            candidates.append(trial)
        return candidates or [dict(base_params)]

    def _score_params_for_algorithm(self, alg_name: str, params: Dict[str, Any]) -> float:
        params = self._strip_auto_internal_params(params)
        reference: Optional[GrayImage] = self.state.get("image")
        degraded: Optional[GrayImage] = self.state.get("degraded")
        psf: Optional[PSF] = self.state.get("psf")
        alg = self.registry.get(alg_name)
        is_blind = isinstance(alg, (BlindRichardsonLucyDeconvolution, TorchBlindAdamTVMAPDeconvolution))
        if degraded is None:
            if reference is None or psf is None:
                return float("-inf")
            degradation_psf = _synchronize_calculation_psf(self.state, reference.data.shape) or calculation_psf_for_image(psf, reference.data.shape)
            degraded = degrade_image(reference, degradation_psf, noise_sigma=0.01)
            self.state["degraded"] = degraded
            self.state["degradation_psf"] = degradation_psf
        run_psf = self._calculation_psf(psf, degraded.data.shape, params, algorithm_name=alg_name)
        if run_psf is None and not is_blind:
            return float("-inf")
        client = getattr(self, "_auto_numerical_client", None)
        if client is not None:
            try:
                return float(client.score_one(alg_name, params, run_psf))
            except AutoCancelledError:
                self._auto_cancel_requested = True
                raise
            except AutoProcessError:
                self._auto_process_failed = True
                self._auto_cancel_requested = True
                raise
        gcv_score = self._plain_wiener_gcv_score(alg_name, degraded, run_psf, params)
        if gcv_score is not None:
            return float(gcv_score)
        safe_params = dict(params)
        if isinstance(alg, TorchAdamTVMAPDeconvolution):
            safe_params["iterations"] = min(int(safe_params.get("iterations", 100)), 25)
        try:
            result = alg.run(degraded, run_psf, **safe_params)
        except Exception as exc:
            if _is_cuda_oom(exc) and bool(safe_params.get("prefer_cuda", True)):
                _safe_torch_worker_cleanup()
                retry_params = dict(safe_params)
                retry_params["prefer_cuda"] = False
                try:
                    result = alg.run(degraded, run_psf, **retry_params)
                except Exception:
                    return float("-inf")
            else:
                return float("-inf")
        frames = result.history or [result.image]
        allow_ref = reference_metrics_available(self.state)
        scores = [metric_score(self._reconstruction_metrics(reference, degraded, frame, run_psf)) for frame in frames]
        return float(max(scores)) if scores else float("-inf")

    def _score_params_batch_for_algorithm(self, alg_name: str, candidates: List[Dict[str, Any]]) -> List[float]:
        if not candidates:
            return []
        candidates = [self._strip_auto_internal_params(p) for p in candidates]
        reference: Optional[GrayImage] = self.state.get("image")
        degraded: Optional[GrayImage] = self.state.get("degraded")
        psf: Optional[PSF] = self.state.get("psf")
        alg = self.registry.get(alg_name)
        is_blind = isinstance(alg, (BlindRichardsonLucyDeconvolution, TorchBlindAdamTVMAPDeconvolution))
        if not getattr(alg, "supports_batched_auto", False):
            return [self._score_params_for_algorithm(alg_name, p) for p in candidates]
        if degraded is None:
            if reference is None or psf is None:
                return [float("-inf")] * len(candidates)
            degradation_psf = _synchronize_calculation_psf(self.state, reference.data.shape) or calculation_psf_for_image(psf, reference.data.shape)
            degraded = degrade_image(reference, degradation_psf, noise_sigma=0.01)
            self.state["degraded"] = degraded
            self.state["degradation_psf"] = degradation_psf
        run_psf = self._calculation_psf(psf, degraded.data.shape, candidates[0], algorithm_name=alg_name)
        if is_blind:
            blind_w = int(candidates[0].get("blind_psf_width", self.state.get("psf_support_width", 1)))
            blind_h = int(candidates[0].get("blind_psf_height", self.state.get("psf_support_height", 1)))
            for candidate in candidates:
                candidate["blind_psf_width"] = blind_w
                candidate["blind_psf_height"] = blind_h
                candidate["psf_size"] = max(blind_w, blind_h)
                candidate["max_psf_width"] = max(blind_w, blind_h)
        if run_psf is None and not is_blind:
            return [float("-inf")] * len(candidates)
        client = getattr(self, "_auto_numerical_client", None)
        if client is not None:
            try:
                return list(client.score_batch(alg_name, candidates, run_psf))
            except AutoCancelledError:
                self._auto_cancel_requested = True
                raise
            except AutoProcessError:
                self._auto_process_failed = True
                self._auto_cancel_requested = True
                raise
        if alg_name in {"Wiener", "Torch batch Wiener"} and not reference_metrics_available(self.state):
            return [
                float(self._plain_wiener_gcv_score(alg_name, degraded, run_psf, candidate) or float("-inf"))
                for candidate in candidates
            ]
        try:
            if not reference_metrics_available(self.state):
                batched = alg.run_batch(degraded, run_psf, candidates, reference=None, keep_history=False)
                result_scores = [
                    metric_score(
                        self._reconstruction_metrics(
                            None,
                            degraded,
                            GrayImage(np.asarray(arr.get("image") if isinstance(arr, dict) else arr), name="candidate"),
                            PSF(np.asarray(arr.get("estimated_psf"), dtype=np.float64), name="candidate_estimated_psf")
                            if isinstance(arr, dict) and arr.get("estimated_psf") is not None else run_psf,
                        )
                    )
                    for arr in batched.infos
                ]
            else:
                result_scores = list(alg.score_batch(reference, degraded, run_psf, candidates))
            if TORCH_AVAILABLE and torch.cuda.is_available() and bool(candidates[0].get("prefer_cuda", True)):
                key = self._auto_batch_cache_key(alg_name, degraded.data.shape, getattr(run_psf, "kernel", np.empty((1, 1))).shape if run_psf is not None else None, candidates[0])
                self._auto_batch_cache[key] = max(int(self._auto_batch_cache.get(key, 0) or 0), len(candidates))
            return result_scores
        except Exception as exc:
            if _is_cuda_oom(exc):
                _safe_torch_worker_cleanup()
                if len(candidates) > 1:
                    midpoint = max(1, len(candidates) // 2)
                    return (
                        self._score_params_batch_for_algorithm(alg_name, candidates[:midpoint])
                        + self._score_params_batch_for_algorithm(alg_name, candidates[midpoint:])
                    )
                cpu_candidate = dict(candidates[0])
                cpu_candidate["prefer_cuda"] = False
                return [self._score_params_for_algorithm(alg_name, cpu_candidate)]
            return [self._score_params_for_algorithm(alg_name, p) for p in candidates]

    def _auto_tune_scalar_coordinate_for_algorithm(self, alg_name: str, current: Dict[str, Any]) -> Tuple[Dict[str, Any], float, int, float]:
        t0 = time.perf_counter()
        best_params = dict(current)
        best_score = self._score_params_for_algorithm(alg_name, best_params)
        tested = 1
        best_params, best_score, k_tested = self._auto_optimize_wiener_k_for_algorithm(alg_name, best_params, best_score)
        tested += k_tested
        for name in self._active_parameter_names(alg_name, best_params):
            if name == "K":
                continue
            if getattr(self, "_auto_cancel_requested", False):
                break
            local_best_params = dict(best_params)
            local_best_score = best_score
            for candidate in self._auto_candidate_values(name, best_params.get(name), best_params):
                if getattr(self, "_auto_cancel_requested", False):
                    break
                trial = dict(best_params)
                trial[name] = candidate
                if name == "psf_size" and trial["psf_size"] > trial.get("max_psf_width", trial["psf_size"]):
                    continue
                score = self._score_params_for_algorithm(alg_name, trial)
                tested += 1
                if score > local_best_score:
                    local_best_score = score
                    local_best_params = trial
            best_params, best_score = local_best_params, local_best_score
        return best_params, best_score, tested, time.perf_counter() - t0

    @staticmethod
    def _auto_numeric_value(value: Any) -> bool:
        return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)

    def _coerce_auto_value(self, name: str, value: float) -> Any:
        """Coerce a proposed Auto value without touching Qt widgets.

        Auto runs in a worker thread, therefore querying QSpinBox/QDoubleSpinBox
        limits here would be undefined Qt behaviour.  The numerical bounds mirror
        the practical GUI limits but are stored as plain Python data.
        """
        bounds: Dict[str, Tuple[float, float]] = {
            "K": (1e-12, 1e4),
            "iterations": (1.0, 100000.0),
            "psf_size": (3.0, 8191.0),
            "max_psf_width": (3.0, 8191.0),
            "kaczmarz_block_size": (2.0, 4096.0),
            "kaczmarz_blocks_per_iteration": (1.0, 100000.0),
            "tv_iterations": (1.0, 10000.0),
            "torch_record_every": (1.0, 100000.0),
            "epsilon": (1e-15, 1.0),
            "torch_lr": (1e-12, 10.0),
            "blind_psf_lr": (1e-12, 10.0),
            "tv_weight": (0.0, 100.0),
            "blind_psf_tv_weight": (0.0, 100.0),
            "step": (1e-12, 100.0),
        }
        lo, hi = bounds.get(name, (-1e12, 1e12))
        value = float(np.clip(float(value), lo, hi))
        integer_names = {
            "iterations", "psf_size", "max_psf_width", "kaczmarz_block_size",
            "kaczmarz_blocks_per_iteration", "tv_iterations", "torch_record_every",
        }
        if name in integer_names:
            result = max(1, int(round(value)))
            if name in {"psf_size", "max_psf_width"} and result % 2 == 0:
                result += 1
            return result
        return float(value)

    def _quadratic_vertex_candidate(self, name: str, xs: List[float], ys: List[float]) -> Optional[Any]:
        finite = [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(y)]
        if len(finite) < 3:
            return None
        # Use three distinct points nearest to the current interval. Scaling x
        # improves conditioning when parameters such as epsilon are very small.
        finite = sorted({x: y for x, y in finite}.items())
        if len(finite) < 3:
            return None
        x_arr = np.asarray([p[0] for p in finite], dtype=np.float64)
        y_arr = np.asarray([p[1] for p in finite], dtype=np.float64)
        center = float(np.mean(x_arr))
        scale = float(np.max(np.abs(x_arr - center)))
        if scale <= 1e-18:
            return None
        z = (x_arr - center) / scale
        try:
            a, b, _ = np.polyfit(z, y_arr, 2)
        except Exception:
            return None
        # The score is maximized, therefore a useful local model must be concave.
        if not np.isfinite(a) or not np.isfinite(b) or a >= -1e-14:
            return None
        z_star = -b / (2.0 * a)
        x_star = center + scale * float(z_star)
        lo, hi = float(np.min(x_arr)), float(np.max(x_arr))
        if not np.isfinite(x_star) or x_star <= lo or x_star >= hi:
            return None
        candidate = self._coerce_auto_value(name, x_star)
        if any(abs(float(candidate) - x) <= 1e-14 * max(1.0, abs(x)) for x in x_arr):
            return None
        return candidate

    def _auto_tune_quadratic_coordinate_for_algorithm(self, alg_name: str, current: Dict[str, Any]) -> Tuple[Dict[str, Any], float, int, float]:
        """Fast coordinate search using a local quadratic model for numeric parameters.

        Each numeric coordinate is sampled at a small number of nearby values.
        A concave parabola is fitted to the score and its vertex is evaluated as
        an additional candidate. Boolean and categorical options are still
        handled by ordinary coordinate enumeration. Candidate sets are scored
        in one Torch batch whenever the selected implementation supports it.
        """
        t0 = time.perf_counter()
        best_params = dict(current)
        best_score = self._score_params_for_algorithm(alg_name, best_params)
        tested = 1
        best_params, best_score, k_tested = self._auto_optimize_wiener_k_for_algorithm(alg_name, best_params, best_score)
        tested += k_tested
        active_names = [name for name in self._active_parameter_names(alg_name, current) if name != "K"]

        # Two short passes allow the local model to adapt after other coordinates
        # have changed, while remaining much cheaper than a Cartesian grid.
        for _pass in range(2):
            changed = False
            for name in active_names:
                if getattr(self, "_auto_cancel_requested", False):
                    break
                raw_values = self._auto_candidate_values(name, best_params.get(name), best_params)
                if not raw_values:
                    continue
                unique_values: List[Any] = []
                for value in raw_values:
                    if value not in unique_values:
                        unique_values.append(value)
                if len(unique_values) <= 1:
                    continue

                # For numeric coordinates, start with lower/current/upper values;
                # categorical coordinates use all explicitly allowed choices.
                current_value = best_params.get(name)
                numeric = self._auto_numeric_value(current_value)
                if numeric:
                    ordered = sorted(unique_values, key=float)
                    current_float = float(current_value)
                    below = [v for v in ordered if float(v) < current_float]
                    above = [v for v in ordered if float(v) > current_float]
                    # Use the nearest lower and upper samples: a local parabola is
                    # more reliable than one fitted to the widest +/-50% endpoints.
                    sample_values = ([below[-1]] if below else []) + [current_value] + ([above[0]] if above else [])
                    sample_values = list(dict.fromkeys(sample_values))
                else:
                    sample_values = unique_values

                trials: List[Dict[str, Any]] = []
                scores: List[float] = []
                pending_trials: List[Dict[str, Any]] = []
                pending_positions: List[int] = []
                for value in sample_values:
                    trial = dict(best_params)
                    trial[name] = value
                    if name == "psf_size" and int(trial[name]) > int(trial.get("max_psf_width", trial[name])):
                        continue
                    trials.append(trial)
                    if value == current_value:
                        scores.append(float(best_score))
                    else:
                        pending_positions.append(len(scores))
                        scores.append(float("nan"))
                        pending_trials.append(trial)
                if not trials:
                    continue
                if pending_trials:
                    pending_scores = self._score_params_batch_for_algorithm(alg_name, pending_trials)
                    tested += len(pending_trials)
                    for pos, score in zip(pending_positions, pending_scores):
                        scores[pos] = float(score)

                if numeric and len(trials) >= 3:
                    xs = [float(t[name]) for t in trials]
                    vertex = self._quadratic_vertex_candidate(name, xs, scores)
                    if vertex is not None:
                        vertex_trial = dict(best_params)
                        vertex_trial[name] = vertex
                        if not (name == "psf_size" and int(vertex_trial[name]) > int(vertex_trial.get("max_psf_width", vertex_trial[name]))):
                            vertex_score = self._score_params_batch_for_algorithm(alg_name, [vertex_trial])[0]
                            tested += 1
                            trials.append(vertex_trial)
                            scores.append(vertex_score)

                finite = np.asarray(scores, dtype=np.float64)
                if finite.size and np.isfinite(finite).any():
                    local_idx = int(np.nanargmax(finite))
                    local_score = float(scores[local_idx])
                    if local_score > best_score:
                        best_score = local_score
                        best_params = trials[local_idx]
                        changed = True
            if getattr(self, "_auto_cancel_requested", False) or not changed:
                break
        return best_params, best_score, tested, time.perf_counter() - t0

    def _candidate_memory_bytes(self, alg_name: str, image_shape: Tuple[int, int], psf_shape: Optional[Tuple[int, int]], params: Dict[str, Any]) -> int:
        """Conservative per-candidate CUDA memory estimate for batched Auto.

        This is intentionally approximate. It includes image/PSF tensors, manual
        Adam moments where applicable, gradients, FFT/work buffers and a safety
        multiplier. The value is used only to choose an initial batch size; if a
        batch still fails, scoring falls back safely.
        """
        h, w = int(image_shape[0]), int(image_shape[1])
        kh, kw = (psf_shape or (1, 1))
        # All Torch paths default to float32; the optional float64 setting is
        # honored for both Adam and batched reference algorithms.
        bytes_per = 8 if bool(params.get("torch_float64", False)) else 4
        img = max(1, h * w) * bytes_per
        ker = max(1, int(kh) * int(kw)) * bytes_per
        if alg_name == "PyTorch Blind Adam TV-MAP":
            # x, grad_x, m_x, v_x, y, blurred, several buffers + h/grad/m/v
            units = 28 * img + 16 * ker
        elif alg_name == "PyTorch Adam TV-MAP":
            units = 24 * img + 4 * ker
        elif alg_name.startswith("Torch batch Richardson-Lucy"):
            units = 14 * img + 4 * ker
        elif alg_name.startswith("Torch batch Landweber"):
            units = 12 * img + 4 * ker
        elif alg_name.startswith("Torch batch Wiener"):
            units = 10 * img + 4 * ker
        else:
            units = 8 * img + 4 * ker
        return int(max(4 * 1024 * 1024, units))

    def _auto_batch_policy_fraction(self, policy: Optional[str] = None) -> float:
        text = str(policy or "Balanced (75%)")
        if "Conservative" in text:
            return 0.50
        if "Aggressive" in text:
            return 0.95
        return 0.75

    def _auto_batch_cache_key(self, alg_name: str, image_shape: Tuple[int, int], psf_shape: Optional[Tuple[int, int]], params: Dict[str, Any]) -> str:
        device = torch.cuda.get_device_name(0) if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        h, w = int(image_shape[0]), int(image_shape[1])
        if psf_shape is None:
            kh, kw = 1, 1
        else:
            kh, kw = int(psf_shape[0]), int(psf_shape[1])
        dtype = "float64" if bool(params.get("torch_float64", False)) else "float32"
        return f"{device}|{alg_name}|{h}x{w}|psf={kh}x{kw}|{dtype}"

    def _recommended_auto_batch_size(self, alg_name: str, current: Dict[str, Any], candidates_count: int) -> Tuple[int, str]:
        """Estimate a safe CUDA batch without stress-allocating most of VRAM.

        Earlier versions probed the GPU by allocating a single byte tensor using
        up to roughly 85% of free memory.  That could fragment or destabilize the
        CUDA allocator and, on some driver/IDE combinations, lead to a native
        crash during the next reconstruction.  The current implementation uses
        a conservative memory model and lets the real scorer split a batch
        recursively if an out-of-memory exception still occurs.
        """
        requested = max(1, int(current.get("auto_batch_size", 32)))
        max_batch = max(1, int(current.get("auto_max_batch_size", 2048)))
        policy = str(current.get("auto_batch_policy", "Balanced (75%)"))
        if candidates_count <= 1:
            return 1, "single candidate"
        if getattr(self, "_auto_numerical_client", None) is not None:
            # CUDA belongs to the isolated Auto process.  Do not synchronize or
            # query the GUI process's CUDA context here; the child scorer will
            # split an oversized batch recursively after a recoverable OOM.
            return min(requested, candidates_count, max_batch), "isolated-process/manual batch"
        if not TORCH_AVAILABLE or not bool(current.get("prefer_cuda", True)) or not torch.cuda.is_available():
            return min(requested, candidates_count, max_batch), "CPU/manual batch"

        degraded = self.state.get("degraded") or self.state.get("image")
        image_shape = getattr(getattr(degraded, "data", None), "shape", (256, 256))
        psf = self.state.get("calculation_psf")
        if not isinstance(psf, PSF):
            psf = _synchronize_calculation_psf(self.state, image_shape)
        psf_shape = getattr(getattr(psf, "kernel", None), "shape", None)
        per_candidate = max(1, self._candidate_memory_bytes(alg_name, image_shape, psf_shape, current))
        cache_key = self._auto_batch_cache_key(alg_name, image_shape, psf_shape, current)
        policy_fraction = self._auto_batch_policy_fraction(policy)

        try:
            torch.cuda.synchronize()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            free_bytes, total_bytes = 0, 0

        cached_max = int(getattr(self, "_auto_batch_cache", {}).get(cache_key, 0) or 0)
        if free_bytes > 0:
            # The policy fraction is applied to a 60% base budget.  Even the
            # aggressive profile therefore keeps a substantial reserve for the
            # CUDA context, cuFFT workspaces, matplotlib and allocator fragmentation.
            budget_fraction = 0.60 * policy_fraction
            memory_estimate = max(1, int((budget_fraction * float(free_bytes)) // float(per_candidate)))
        else:
            memory_estimate = requested

        estimated_max = max(requested, memory_estimate)
        if cached_max > 0:
            # A previous successful real batch is useful evidence, but do not
            # exceed the current memory estimate after image/PSF changes.
            estimated_max = max(requested, min(cached_max, estimated_max))

        chosen = min(max_batch, candidates_count, max(1, estimated_max))
        chosen = max(1, chosen)
        note = (
            f"CUDA safe estimate {chosen} candidates/batch; policy {policy}; "
            f"free VRAM {free_bytes / 1024**3:.2f} GiB; "
            f"estimated {per_candidate / 1024**2:.1f} MiB/candidate"
            if free_bytes > 0
            else f"CUDA memory query unavailable; using batch {chosen}"
        )
        return chosen, note

    def _torch_auto_equivalent(self, alg_name: str) -> str:
        """Return the Torch-batched algorithm to use for Auto when available."""
        candidate = self._TORCH_BATCH_PAIRS.get(str(alg_name), str(alg_name))
        try:
            self.registry.get(candidate)
            return candidate
        except Exception:
            return str(alg_name)

    def _plain_equivalent_for_torch_auto(self, alg_name: str) -> Optional[str]:
        """Return the NumPy/reference algorithm paired with a Torch-batched target."""
        plain = self._TORCH_BATCH_REVERSE.get(str(alg_name))
        if plain is None:
            return None
        try:
            self.registry.get(plain)
            return plain
        except Exception:
            return None

    def _strip_auto_internal_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove worker-only keys before running or storing algorithm parameters."""
        return {k: v for k, v in dict(params).items() if not str(k).startswith("__auto_")}

    def _auto_validation_algorithm(self, tuned_alg_name: str, params: Dict[str, Any]) -> Optional[str]:
        """Return the algorithm whose Run button result should validate Auto scores.

        Auto may score a Torch-batched implementation for speed even when the
        user selected the ordinary NumPy algorithm.  The score displayed after
        Auto should then be re-measured on the implementation that will actually
        be run by the button, otherwise the reported PSNR can be slightly too
        optimistic or simply different.
        """
        selected = params.get("__auto_selected_algorithm")
        if isinstance(selected, str) and selected:
            if selected != tuned_alg_name:
                try:
                    self.registry.get(selected)
                    return selected
                except Exception:
                    pass
        return None

    def _related_algorithm_names_for_params(self, alg_name: str) -> List[str]:
        """Algorithms that should share Auto-selected hyperparameters."""
        pairs = self._TORCH_BATCH_PAIRS
        for plain, torch_name in pairs.items():
            if alg_name in (plain, torch_name):
                return [plain, torch_name]
        return [alg_name]

    def _merge_params_for_target_algorithm(self, target_alg: str, source_params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge source params with target defaults/remembered values for Auto."""
        merged = dict(self._algorithm_param_values.get(target_alg, {}))
        try:
            merged.update(getattr(self.registry.get(target_alg), "default_params", {}))
        except Exception:
            pass
        merged.update(source_params)
        return merged

    def _auto_tune_algorithm_sync(self, alg_name: str, current: Dict[str, Any]) -> Tuple[Dict[str, Any], float, str]:
        """Synchronous Auto implementation used by the background worker.

        Auto is guarded against regressions: before accepting tuned parameters,
        the current parameter set is scored on the same implementation used for
        validation/Run deconvolution.  Tuned parameters are accepted only when
        they do not make that score worse.
        """
        if getattr(self, "_auto_cancel_requested", False):
            return dict(current), float("-inf"), f"Auto cancelled before tuning {alg_name}."

        # Freeze which parameters are semantically active at the beginning of
        # this Auto run. Candidate changes may not enable a previously disabled
        # Wiener initializer, denoiser or optional TV stage and then tune values
        # that were hidden when Auto started.
        current = dict(current)
        initial_active = self._active_parameter_names(alg_name, current)
        current["__auto_allowed_parameter_names"] = auto_tunable_parameter_names(
            alg_name, initial_active, current
        )

        validation_baseline_alg = self._auto_validation_algorithm(alg_name, current) or alg_name
        baseline_params = self._strip_auto_internal_params(dict(current))
        baseline_score = self._score_params_for_algorithm(validation_baseline_alg, baseline_params)
        accept_tol = 1e-9

        use_batch = getattr(self.registry.get(alg_name), "supports_batched_auto", False)
        auto_strategy = str(current.get("auto_strategy", "Quadratic coordinate (fast)"))
        if auto_strategy.startswith("Quadratic"):
            best_params, best_score, tested, elapsed = self._auto_tune_quadratic_coordinate_for_algorithm(alg_name, current)
            if getattr(self, "_auto_cancel_requested", False):
                return best_params, best_score, f"Auto cancelled for {alg_name} after {tested} tested candidates."

            validation_alg = self._auto_validation_algorithm(alg_name, current)
            validated_note = ""
            if validation_alg is not None and np.isfinite(best_score):
                proxy_score = best_score
                validated_score = self._score_params_for_algorithm(validation_alg, best_params)
                if np.isfinite(validated_score):
                    best_score = float(validated_score)
                    validated_note = (
                        f"; proxy {alg_name}={self._auto_score_description(alg_name, proxy_score)}; "
                        f"validated on {validation_alg}"
                    )
            per = 1000.0 * elapsed / max(1, tested)
            status = (
                f"Auto (quadratic coordinate): {validation_alg or alg_name}; "
                f"{self._auto_score_description(alg_name, best_score)}; "
                f"tested {tested}; {elapsed:.2f} s; {per:.1f} ms/candidate{validated_note}"
            ) if np.isfinite(best_score) else f"Auto (quadratic coordinate): {alg_name}; no valid score"
            best_params = self._strip_auto_internal_params(best_params)
            if np.isfinite(baseline_score) and (not np.isfinite(best_score) or best_score < baseline_score - accept_tol):
                status = (
                    f"Auto kept previous parameters for {alg_name}; "
                    f"quadratic-search result {self._auto_score_description(alg_name, best_score)} "
                    f"was worse than baseline {self._auto_score_description(alg_name, baseline_score)}; "
                    f"tested {tested}"
                )
                return baseline_params, baseline_score, status
            return best_params, best_score, status

        if not use_batch:
            best_params, best_score, tested, elapsed = self._auto_tune_scalar_coordinate_for_algorithm(alg_name, current)
            if getattr(self, "_auto_cancel_requested", False):
                return best_params, best_score, f"Auto cancelled for {alg_name} after {tested} tested candidates."
            if np.isfinite(best_score):
                per = 1000.0 * elapsed / max(1, tested)
                status = f"Auto (background scalar): {alg_name}; {self._auto_score_description(alg_name, best_score)}; tested {tested}; {elapsed:.2f} s; {per:.1f} ms/candidate"
            else:
                status = f"Auto (background scalar): {alg_name}; no valid score"
            if np.isfinite(baseline_score) and (not np.isfinite(best_score) or best_score < baseline_score - accept_tol):
                status = (
                    f"Auto kept previous parameters for {alg_name}; "
                    f"best candidate {self._auto_score_description(alg_name, best_score)} "
                    f"was worse than baseline {self._auto_score_description(alg_name, baseline_score)}; "
                    f"tested {tested}; {elapsed:.2f} s"
                )
                return baseline_params, baseline_score, status
            return best_params, best_score, status

        t0 = time.perf_counter()
        proxy_initial_score = self._score_params_for_algorithm(alg_name, current)
        current, proxy_initial_score, k_tested = self._auto_optimize_wiener_k_for_algorithm(alg_name, current, proxy_initial_score)
        names = [name for name in self._active_parameter_names(alg_name, current) if name != "K"]
        candidates = self._full_candidate_pool(current, names)
        batch_size, batch_note = self._recommended_auto_batch_size(alg_name, current, len(candidates))
        current["auto_batch_size"] = batch_size
        scores: List[float] = []
        cancelled = False
        for i in range(0, len(candidates), batch_size):
            if getattr(self, "_auto_cancel_requested", False):
                cancelled = True
                break
            chunk = candidates[i:i + batch_size]
            scores.extend(self._score_params_batch_for_algorithm(alg_name, chunk))
        elapsed = time.perf_counter() - t0
        if cancelled:
            return dict(current), float("-inf"), f"Auto cancelled for {alg_name} after {len(scores) + k_tested} scored candidates."
        proxy_score = float("-inf")
        proxy_idx = -1
        if scores and np.isfinite(np.asarray(scores, dtype=np.float64)).any():
            arr = np.asarray(scores, dtype=np.float64)
            proxy_idx = int(np.nanargmax(arr))
            proxy_score = float(scores[proxy_idx])
            best_idx = proxy_idx
            best_score = proxy_score
            best_params = candidates[best_idx]
        else:
            best_idx = -1
            best_score = float("-inf")
            best_params = dict(current)

        validation_alg = self._auto_validation_algorithm(alg_name, current)
        validated_note = ""
        if validation_alg is not None and scores and np.isfinite(np.asarray(scores, dtype=np.float64)).any():
            # Validate the best Torch-proxy candidates on the exact implementation
            # that the Run button will use.  This removes the confusing case in
            # which Auto reports a slightly higher Torch-batch PSNR, but the
            # ordinary Richardson-Lucy run gives a lower value after the same
            # hyperparameters are copied.  We validate only a small top-K set so
            # the fast batched search is still useful.
            arr = np.asarray(scores, dtype=np.float64)
            finite_idx = np.flatnonzero(np.isfinite(arr))
            if finite_idx.size:
                top_k = int(min(8, finite_idx.size))
                order = finite_idx[np.argsort(arr[finite_idx])[-top_k:]][::-1]
                validation_scores: List[float] = []
                for idx in order:
                    if getattr(self, "_auto_cancel_requested", False):
                        break
                    validation_scores.append(self._score_params_for_algorithm(validation_alg, candidates[int(idx)]))
                if validation_scores and np.isfinite(np.asarray(validation_scores, dtype=np.float64)).any():
                    local = int(np.nanargmax(np.asarray(validation_scores, dtype=np.float64)))
                    best_idx = int(order[local])
                    best_params = candidates[best_idx]
                    best_score = float(validation_scores[local])
                    validated_note = (
                        f"; proxy {alg_name} best={self._auto_score_description(alg_name, proxy_score)}; "
                        f"validated on {validation_alg} using top {len(validation_scores)}"
                    )

        best_params = self._strip_auto_internal_params(best_params)
        device = torch_backend_device(bool(best_params.get("prefer_cuda", True))) if TORCH_AVAILABLE else "unavailable"
        if np.isfinite(best_score):
            per = 1000.0 * elapsed / max(1, len(candidates) + k_tested)
            score_alg = validation_alg or alg_name
            status = (
                f"Auto (background full batched, {device}): {score_alg}; {self._auto_score_description(alg_name, best_score)}; "
                f"candidate {best_idx + 1}/{len(candidates)}; logarithmic K trials {k_tested}; batch {batch_size}; {batch_note}; {elapsed:.2f} s; {per:.1f} ms/candidate"
                f"{validated_note}"
            )
        else:
            status = f"Auto (background full batched): {alg_name}; no valid score; tested {len(candidates) + k_tested}"

        if np.isfinite(baseline_score) and (not np.isfinite(best_score) or best_score < baseline_score - accept_tol):
            status = (
                f"Auto kept previous parameters for {alg_name}; "
                f"best candidate {self._auto_score_description(alg_name, best_score)} "
                f"was worse than baseline {self._auto_score_description(alg_name, baseline_score)}; "
                f"tested {len(candidates) + k_tested} candidates; baseline measured on {validation_baseline_alg}"
            )
            return baseline_params, baseline_score, status
        return best_params, best_score, status

    def _set_auto_controls_running(self, running: bool) -> None:
        """Keep all controls that can alter Auto state synchronized.

        The Rosen preset changes the same parameter store that Auto reads and
        writes, so it must not be applied while an Auto worker is active or is
        still completing cancellation.
        """
        running = bool(running)
        self.auto_button.setEnabled(not running)
        self.auto_all_button.setEnabled(not running)
        self.cancel_auto_button.setEnabled(running)
        try:
            self.rosen_match_rl_button.setEnabled(not running)
        except Exception:
            pass

    def _active_auto_threads_running(self) -> bool:
        """Return True only when a real Auto QThread is still alive.

        Older versions also trusted ``_auto_running``.  If the Qt ``finished``
        callback was delayed or skipped during cancellation/reloading, that
        flag could remain True indefinitely and leave Auto/Auto All disabled.
        The QThread objects are the source of truth; a stale flag is repaired
        automatically when no live thread exists.
        """
        cleaned: List[Tuple[QThread, AutoTuneWorker]] = []
        running = False
        candidates = list(getattr(self, "_auto_threads", []))
        current_thread = getattr(self, "_auto_thread", None)
        current_worker = getattr(self, "_auto_worker", None)
        if current_thread is not None and all(t is not current_thread for t, _ in candidates):
            candidates.append((current_thread, current_worker))
        for thread, worker in candidates:
            try:
                if thread is not None and thread.isRunning():
                    cleaned.append((thread, worker))
                    running = True
            except RuntimeError:
                # Wrapped C++ object already deleted.
                continue
        self._auto_threads = cleaned
        if not running:
            # Recover from a stale busy state, including the sequence
            # Cancel Auto -> apply Rosen preset before the thread-finished slot.
            self._auto_running = False
            self._auto_thread = None
            self._auto_worker = None
            self._auto_cancel_requested = False
            self._set_auto_controls_running(False)
        return running

    def _prepare_auto_numerical_payload(self) -> Optional[Dict[str, Any]]:
        """Freeze the image data used by one isolated Auto process.

        Auto controls are locked while the worker is active, but a private
        process should still receive an immutable snapshot.  If only a
        reference image is available, create the same synthetic degraded input
        that the former in-thread scoring path created lazily.
        """
        reference: Optional[GrayImage] = self.state.get("image")
        degraded: Optional[GrayImage] = self.state.get("degraded")
        source_psf: Optional[PSF] = self.state.get("psf")
        if degraded is None:
            if reference is None or source_psf is None:
                return None
            degradation_psf = _synchronize_calculation_psf(self.state, reference.data.shape)
            if degradation_psf is None:
                degradation_psf = calculation_psf_for_image(source_psf, reference.data.shape)
            if degradation_psf is None:
                return None
            degraded = degrade_image(reference, degradation_psf, noise_sigma=0.01)
            self.state["degraded"] = degraded
            self.state["degradation_psf"] = degradation_psf
        calculation_psf = _synchronize_calculation_psf(self.state, degraded.data.shape)
        return {
            "reference": reference,
            "degraded": degraded,
            "psf": calculation_psf,
            "allow_reference": bool(reference_metrics_available(self.state)),
        }

    def _start_auto_worker(self, jobs: List[Tuple[str, Dict[str, Any]]]) -> None:
        reference: Optional[GrayImage] = self.state.get("image")
        degraded: Optional[GrayImage] = self.state.get("degraded")
        if reference is None and degraded is None:
            QMessageBox.warning(self, "Missing data", "Load a measured/degraded image or generate a reference image before running Auto.")
            return
        if self._active_auto_threads_running():
            QMessageBox.information(self, "Auto is running", "Auto optimization is already running in the background. Please wait until it finishes before starting another Auto job.")
            return
        active_owner = _current_numerical_owner()
        if active_owner is not None:
            QMessageBox.information(
                self,
                "Numerical task is running",
                f"Cannot start Auto while {active_owner} is still using the numerical backend.",
            )
            return
        numerical_payload = self._prepare_auto_numerical_payload()
        if numerical_payload is None:
            QMessageBox.warning(self, "Missing data", "Auto requires a measured/degraded image and, for non-blind methods, a calculation PSF.")
            return
        self._set_auto_controls_running(True)
        self._auto_cancel_requested = False
        self._auto_process_failed = False
        self._auto_running = True
        self.auto_status.setText("Auto: starting background worker ...")
        thread = QThread()  # no QObject parent; lifetime is held explicitly below
        worker = AutoTuneWorker(self, jobs, numerical_payload, cancel_grace_seconds=5.0)
        worker.moveToThread(thread)
        self._auto_thread = thread
        self._auto_worker = worker
        self._auto_threads.append((thread, worker))
        thread.started.connect(worker.run)
        worker.progress.connect(self.auto_status.setText)
        worker.algorithm_finished.connect(self._on_auto_algorithm_finished)
        worker.failed.connect(self.auto_status.setText)
        # Deferred deletion is requested while the worker thread event loop is
        # still alive.  Controls are re-enabled only from QThread.finished.
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(lambda t=thread, w=worker: self._on_auto_worker_finished(t, w))
        thread.start()

    def _on_auto_algorithm_finished(self, alg_name: str, best_params: Dict[str, Any], best_score: float, status: str) -> None:
        best_params = self._strip_auto_internal_params(best_params)
        # If Auto tuned a Torch-batched equivalent, copy the recovered
        # hyperparameters to the ordinary NumPy algorithm as well.  This keeps
        # both versions synchronized without forcing the user to tune twice.
        for related in self._related_algorithm_names_for_params(alg_name):
            previous = dict(self._algorithm_param_values.get(related, {}))
            previous.update(best_params)
            self._algorithm_param_values[related] = previous
        if self.combo.currentText() in self._related_algorithm_names_for_params(alg_name):
            self._apply_params_to_widgets(self._algorithm_param_values[self.combo.currentText()])
            self._update_visible_parameter_rows()
        self.auto_status.setText(status)

    def _on_auto_worker_returned(self, thread: Optional[QThread] = None, worker: Optional[AutoTuneWorker] = None) -> None:
        """Handle return from ``AutoTuneWorker.run`` without racing QThread shutdown.

        The worker has completed, but the QThread may need one more event-loop
        cycle to process ``quit()``.  Controls remain locked during that short
        interval.  A timer repairs the UI if Spyder delays the normal
        ``QThread.finished`` callback.
        """
        self._auto_running = False
        try:
            if thread is not None:
                thread.quit()
        except RuntimeError:
            pass
        QTimer.singleShot(50, lambda t=thread: self._recover_auto_controls_after_return(t, 0))

    def _recover_auto_controls_after_return(self, thread: Optional[QThread], attempt: int = 0) -> None:
        """Re-enable Auto controls once the returned worker thread really stops."""
        running = False
        try:
            running = bool(thread is not None and thread.isRunning())
        except RuntimeError:
            running = False
        if running and attempt < 200:
            QTimer.singleShot(50, lambda t=thread, a=attempt + 1: self._recover_auto_controls_after_return(t, a))
            return
        if not running:
            self._set_auto_controls_running(False)
            try:
                if bool(getattr(self, "_auto_cancel_requested", False)):
                    self.auto_status.setText("Auto: cancelled; worker stopped")
                elif "finished" not in self.auto_status.text().lower():
                    self.auto_status.setText(self.auto_status.text() + " | finished")
            except Exception:
                pass

    def _on_auto_worker_finished(self, thread: Optional[QThread] = None, worker: Optional[AutoTuneWorker] = None) -> None:
        self._set_auto_controls_running(False)
        was_cancelled = bool(getattr(self, "_auto_cancel_requested", False))
        self._auto_running = False
        try:
            txt = self.auto_status.text()
            if was_cancelled:
                self.auto_status.setText("Auto: cancellation requested; worker stopped")
            elif "finished" not in txt:
                self.auto_status.setText(txt + " | finished")
        except Exception:
            pass
        if thread is None:
            thread = self._auto_thread
        if worker is None:
            worker = self._auto_worker
        self._auto_threads = [(t, w) for (t, w) in self._auto_threads if t is not thread]
        if self._auto_thread is thread:
            self._auto_thread = None
            self._auto_worker = None
        self._auto_cancel_requested = False
        try:
            if thread is not None:
                thread.deleteLater()
        except RuntimeError:
            pass

    def request_cancel_auto(self) -> None:
        """Request cancellation of the running Auto/Auto All job without destroying the QThread."""
        if not self._active_auto_threads_running():
            self.auto_status.setText("Auto: no running job to cancel")
            self.cancel_auto_button.setEnabled(False)
            return
        self._auto_cancel_requested = True
        self.cancel_auto_button.setEnabled(False)
        self.auto_status.setText(
            "Auto: cancellation requested; stopping cooperatively. "
            "If the current numerical iteration does not return within 5.0 s, it will be force-stopped ..."
        )
        for thread, worker in list(getattr(self, "_auto_threads", [])):
            try:
                if worker is not None:
                    worker.cancel()
            except Exception:
                pass

    def cancel_auto_workers(self, timeout_ms: int = 8000) -> bool:
        """Request cancellation and keep QThread references until they stop.

        Returning ``False`` means at least one worker is still inside a numerical
        iteration/batch.  The caller must not destroy the window or clear the
        references in that state.
        """
        self._auto_cancel_requested = True
        try:
            self.cancel_auto_button.setEnabled(False)
        except Exception:
            pass
        pairs = list(self._auto_threads)
        for thread, worker in pairs:
            try:
                if worker is not None:
                    worker.cancel()
            except Exception:
                pass
            try:
                if thread is not None and thread.isRunning():
                    thread.quit()
            except RuntimeError:
                pass

        remaining = max(0, int(timeout_ms))
        alive: List[Tuple[QThread, AutoTuneWorker]] = []
        for thread, worker in pairs:
            try:
                if thread is not None and thread.isRunning():
                    t0 = time.perf_counter()
                    thread.wait(remaining)
                    elapsed = int((time.perf_counter() - t0) * 1000.0)
                    remaining = max(0, remaining - elapsed)
                if thread is not None and thread.isRunning():
                    alive.append((thread, worker))
            except RuntimeError:
                continue

        if alive:
            self._auto_threads = alive
            self._auto_thread, self._auto_worker = alive[0]
            self._auto_running = True
            self.auto_status.setText("Auto: cancellation requested; waiting for the 5 s grace period / isolated-process stop ...")
            return False

        self._auto_threads.clear()
        self._auto_thread = None
        self._auto_worker = None
        self._auto_running = False
        self._auto_cancel_requested = False
        try:
            self._set_auto_controls_running(False)
        except Exception:
            pass
        return True

    def auto_tune(self) -> None:
        """Start background Auto for the currently selected algorithm."""
        self._remember_current_algorithm_params()
        selected_name = self.combo.currentText()
        target_name = self._torch_auto_equivalent(selected_name)
        current = self._merge_params_for_target_algorithm(target_name, dict(self.params()))
        current["__auto_selected_algorithm"] = target_name if bool(current.get("use_torch_batch", True)) else selected_name
        self._start_auto_worker([(target_name, current)])

    def auto_tune_all(self) -> None:
        """Sequentially tune all algorithms in a background worker."""
        self._remember_current_algorithm_params()
        jobs: List[Tuple[str, Dict[str, Any]]] = []
        current_name = self.combo.currentText()
        seen_targets = set()
        for alg_name in self._display_algorithm_names():
            target_name = self._torch_auto_equivalent(alg_name)
            if target_name in seen_targets:
                continue
            seen_targets.add(target_name)
            params = dict(self._algorithm_param_values.get(alg_name, self.params()))
            # Preserve current visible values for the currently selected method.
            if alg_name == current_name:
                params = dict(self.params())
            params = self._merge_params_for_target_algorithm(target_name, params)
            params["__auto_selected_algorithm"] = target_name if bool(params.get("use_torch_batch", True)) else alg_name
            jobs.append((target_name, params))
        self._start_auto_worker(jobs)


class DeconvolutionRunWorker(QObject):
    """Execute one deconvolution run outside the Qt GUI thread.

    Iterative algorithms receive a cooperative stop event and an iteration
    callback.  A stop request is checked only after a completed iteration, so
    FFT/CUDA operations are never interrupted in the middle of an update.
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object, bool, int, int)
    failed = pyqtSignal(str)

    def __init__(self, algorithm: DeconvolutionAlgorithm, image: GrayImage, psf: Optional[PSF], params: Dict[str, Any]) -> None:
        super().__init__()
        self.algorithm = algorithm
        self.image = image
        self.psf = psf
        self.params = dict(params)
        self.stop_event = threading.Event()
        self.current_iteration = 0
        self.total_iterations = self.planned_iterations(algorithm, self.params)

    @staticmethod
    def wiener_scan_values(params: Dict[str, Any]) -> List[float]:
        lo = max(1e-12, float(params.get("wiener_k_scan_min", 1e-10)))
        hi = max(1e-12, float(params.get("wiener_k_scan_max", 1e-1)))
        if hi < lo:
            lo, hi = hi, lo
        count = max(2, min(200, int(params.get("wiener_k_scan_points", 31))))
        if abs(np.log10(hi) - np.log10(lo)) < 1e-15:
            return [float(lo)]
        return [float(v) for v in np.logspace(np.log10(lo), np.log10(hi), count)]

    @classmethod
    def planned_iterations(cls, algorithm: DeconvolutionAlgorithm, params: Dict[str, Any]) -> int:
        name = str(getattr(algorithm, "name", ""))
        if name in {"Wiener", "Torch batch Wiener"}:
            if bool(params.get("wiener_k_scan_enabled", False)):
                return max(1, len(cls.wiener_scan_values(params)))
            return 1
        return max(1, int(params.get("iterations", 1)))

    def request_stop(self) -> None:
        self.stop_event.set()

    def _on_iteration(self, current: int, total: int) -> None:
        self.current_iteration = int(current)
        self.total_iterations = max(1, int(total))
        self.progress.emit(self.current_iteration, self.total_iterations)

    def run(self) -> None:
        owner = f"run:{id(self)}"
        if not _try_begin_numerical_work(owner):
            active = _current_numerical_owner() or "another numerical task"
            self.failed.emit(f"Deconvolution cannot start while {active} is still running.")
            return
        try:
            with _NUMERICAL_WORK_LOCK:
                params = dict(self.params)
                params["_iteration_callback"] = self._on_iteration
                params["_stop_event"] = self.stop_event
                self.progress.emit(0, self.total_iterations)
                name = str(getattr(self.algorithm, "name", ""))
                scan_enabled = bool(params.get("wiener_k_scan_enabled", False)) and name in {"Wiener", "Torch batch Wiener"}
                if scan_enabled:
                    if self.psf is None:
                        raise ValueError("The Wiener K scan requires a PSF.")
                    history: List[GrayImage] = []
                    gcv_values: List[float] = []
                    k_values = self.wiener_scan_values(params)
                    noise_psd = normalized_noise_psd_from_image(self.image, params)
                    for index, kval in enumerate(k_values, start=1):
                        trial = dict(params)
                        trial["K"] = float(kval)
                        trial["wiener_k_scan_enabled"] = False
                        trial.pop("_iteration_callback", None)
                        trial.pop("_stop_event", None)
                        one = self.algorithm.run(self.image, self.psf, **trial)
                        frame = one.image
                        metadata = dict(getattr(frame, "metadata", {}) or {})
                        gcv = wiener_gcv_cost(self.image.data, self.psf, float(kval), noise_psd=noise_psd)
                        metadata.update({
                            "wiener_K": float(kval),
                            "wiener_gcv": float(gcv),
                            "wiener_k_scan_index": int(index),
                            "wiener_k_scan_total": int(len(k_values)),
                        })
                        frame.metadata = metadata
                        history.append(frame)
                        gcv_values.append(float(gcv))
                        self.current_iteration = index
                        self.total_iterations = len(k_values)
                        self.progress.emit(index, self.total_iterations)
                        if self.stop_event.is_set():
                            break
                    if not history:
                        raise RuntimeError("The Wiener K scan produced no result.")
                    finite = np.asarray(gcv_values, dtype=np.float64)
                    best_index = int(np.nanargmin(finite)) if np.isfinite(finite).any() else len(history) - 1
                    best_frame = history[best_index]
                    result = DeconvolutionResult(
                        best_frame,
                        history=history,
                        info=(
                            f"Wiener logarithmic K scan; points={len(history)}/{len(k_values)}; "
                            f"range={k_values[0]:.6g}..{k_values[-1]:.6g}; "
                            f"minimum GCV at K={best_frame.metadata.get('wiener_K', float('nan')):.6g}"
                        ),
                    )
                else:
                    result = self.algorithm.run(self.image, self.psf, **params)
                if self.current_iteration <= 0:
                    self.current_iteration = self.total_iterations
                    self.progress.emit(self.current_iteration, self.total_iterations)
                stopped = bool(self.stop_event.is_set() and self.current_iteration < self.total_iterations)
                _safe_torch_worker_cleanup()
                self.finished.emit(result, stopped, self.current_iteration, self.total_iterations)
        except Exception as exc:
            _safe_torch_worker_cleanup()
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            _end_numerical_work(owner)



class TestTab(QWidget):
    psfRedefined = pyqtSignal()

    def __init__(self, app_state: Dict[str, Any], alg_tab: AlgorithmTab) -> None:
        super().__init__()
        self.state = app_state
        self.alg_tab = alg_tab
        self.history: List[GrayImage] = []
        self.run_thread: Optional[QThread] = None
        self.run_worker: Optional[DeconvolutionRunWorker] = None
        self._run_context: Dict[str, Any] = {}
        self._run_current_iteration = 0
        self._run_total_iterations = 0
        self._display_hist_cache: Dict[int, Dict[str, Any]] = {}
        self._common_display_hist_cache: Optional[Dict[str, Any]] = None
        self._frame_metrics_cache: Dict[int, Dict[str, float]] = {}
        self._display_refresh_timer = QTimer(self)
        self._display_refresh_timer.setSingleShot(True)
        self._display_refresh_timer.setInterval(60)
        self._display_refresh_timer.timeout.connect(self._refresh_result_display_only)
        layout = QVBoxLayout(self)

        self.btn_run = QPushButton("Run deconvolution")
        self.btn_stop = QPushButton("Stop after current iteration")
        self.btn_stop.setEnabled(False)
        self.btn_use_displayed_k = QPushButton("Use displayed K")
        self.btn_use_displayed_k.setEnabled(False)
        self.btn_use_displayed_k.setToolTip("Copy the K value of the currently displayed Wiener scan frame to the algorithm settings.")
        btn_save = QPushButton("Save current result")
        btn_save_psf = QPushButton("Save current PSF")
        btn_redefine_psf = QPushButton("Redefine PSF")
        btn_redefine_psf.setToolTip("Replace the current known PSF with the PSF corresponding to the displayed result/iteration.")
        btn_save_mat = QPushButton("Save test images to MAT")
        top = QHBoxLayout()
        top.addWidget(self.btn_run)
        top.addWidget(self.btn_stop)
        top.addWidget(self.btn_use_displayed_k)
        top.addWidget(btn_save)
        top.addWidget(btn_save_psf)
        top.addWidget(btn_redefine_psf)
        top.addWidget(btn_save_mat)
        top.addStretch(1)
        layout.addLayout(top)

        progress_row = QHBoxLayout()
        self.run_progress_label = QLabel("Current iteration: - / -")
        self.run_progress_bar = QProgressBar()
        self.run_progress_bar.setRange(0, 1)
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setTextVisible(True)
        progress_row.addWidget(self.run_progress_label)
        progress_row.addWidget(self.run_progress_bar, 1)
        layout.addLayout(progress_row)

        display_group = QGroupBox("Result display levels")
        display_grid = QGridLayout(display_group)
        self.display_independent_check = QCheckBox("Map slider positions to each iteration independently")
        self.display_independent_check.setChecked(False)
        self.display_independent_check.setToolTip(
            "When enabled, slider positions are mapped between the minimum and maximum of the current iteration. "
            "When disabled, one common intensity domain is used for all stored iterations."
        )
        self.display_black_level_slider = QSlider(Qt.Horizontal)
        self.display_black_level_slider.setRange(0, 4095)
        self.display_black_level_slider.setSingleStep(1)
        self.display_black_level_slider.setPageStep(64)
        self.display_black_level_slider.setValue(0)
        self.display_black_level_slider.setTracking(True)
        self.display_black_level_slider.setToolTip(
            "Direct display-black intensity. Percentile rank is calculated from a cached 4096-bin cumulative histogram."
        )
        self.display_black_level_label = QLabel("Black: -")
        self.display_white_level_slider = QSlider(Qt.Horizontal)
        self.display_white_level_slider.setRange(0, 4095)
        self.display_white_level_slider.setSingleStep(1)
        self.display_white_level_slider.setPageStep(64)
        self.display_white_level_slider.setValue(4095)
        self.display_white_level_slider.setTracking(True)
        self.display_white_level_slider.setToolTip(
            "Direct display-white intensity. Percentile rank is calculated from a cached 4096-bin cumulative histogram."
        )
        self.display_white_level_label = QLabel("White: -")
        # Compatibility aliases for settings/scripts written for v89-v91.
        self.display_low_percentile_slider = self.display_black_level_slider
        self.display_percentile_slider = self.display_white_level_slider
        self.display_low_percentile_label = self.display_black_level_label
        self.display_percentile_label = self.display_white_level_label

        self.btn_select_best_iteration = QPushButton("Select best iteration")
        self.btn_select_best_iteration.setEnabled(False)
        self.btn_select_best_iteration.setToolTip(
            "Return to the best stored iteration using the same criterion that is applied automatically after a run."
        )
        self.btn_auto_levels = QPushButton("Auto levels")
        self.btn_auto_levels.setEnabled(False)
        self.btn_auto_levels.setToolTip(
            "Instantly set black and white from the cached cumulative histogram (0.5th and 99.5th percentiles)."
        )
        self.btn_auto_display_range = self.btn_auto_levels
        self.btn_optimize_display_range = QPushButton("Optimize display criterion")
        self.btn_optimize_display_range.setEnabled(False)
        self.btn_optimize_display_range.setToolTip(
            "Evaluate a small set of intensity-level candidates on a downsampled image. This is slower than Auto levels but much faster than the former dense percentile search."
        )
        self.display_scale_label = QLabel("Display range: -")
        display_grid.addWidget(self.display_independent_check, 0, 0, 1, 3)
        display_grid.addWidget(QLabel("Black level"), 1, 0)
        display_grid.addWidget(self.display_black_level_slider, 1, 1)
        display_grid.addWidget(self.display_black_level_label, 1, 2)
        display_grid.addWidget(QLabel("White level"), 2, 0)
        display_grid.addWidget(self.display_white_level_slider, 2, 1)
        display_grid.addWidget(self.display_white_level_label, 2, 2)
        display_buttons = QHBoxLayout()
        display_buttons.addWidget(self.btn_select_best_iteration)
        display_buttons.addWidget(self.btn_auto_levels)
        display_buttons.addWidget(self.btn_optimize_display_range)
        display_buttons.addStretch(1)
        display_grid.addLayout(display_buttons, 3, 0, 1, 3)
        display_grid.addWidget(self.display_scale_label, 4, 0, 1, 3)
        display_grid.setColumnStretch(1, 1)
        layout.addWidget(display_group)

        self.selected_method_label = QLabel("Selected algorithm to run: -")
        self.selected_method_label.setWordWrap(True)
        layout.addWidget(self.selected_method_label)

        self.last_run_method_label = QLabel("Last run: -")
        self.last_run_method_label.setWordWrap(True)
        layout.addWidget(self.last_run_method_label)

        views = QHBoxLayout()
        self.reference_canvas = ImageCanvas("Reference image")
        self.input_canvas = ImageCanvas("Degraded input")
        self.result_canvas = ImageCanvas("Result")
        self.estimated_psf_canvas = ImageCanvas("Estimated PSF")
        views.addWidget(self.reference_canvas)
        views.addWidget(self.input_canvas)
        views.addWidget(self.result_canvas)
        views.addWidget(self.estimated_psf_canvas)
        layout.addLayout(views)

        browser = QHBoxLayout()
        self.iter_slider = QSlider()
        self.iter_slider.setOrientation(1)  # Horizontal
        self.iter_slider.setEnabled(False)
        self.iter_label = QLabel("Iteration: -")
        self.metrics_label = QLabel("Original region — PSNR: -    SSIM: -    TV: -")
        browser.addWidget(QLabel("Browse iterations"))
        browser.addWidget(self.iter_slider)
        browser.addWidget(self.iter_label)
        browser.addWidget(self.metrics_label)
        layout.addLayout(browser)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(QLabel("Test log"))
        layout.addWidget(self.log)

        self.btn_run.clicked.connect(self.run_algorithm)
        self.btn_stop.clicked.connect(self.request_stop)
        self.btn_use_displayed_k.clicked.connect(self.use_displayed_wiener_k)
        self.btn_select_best_iteration.clicked.connect(self.select_best_iteration)
        self.btn_auto_levels.clicked.connect(self.auto_set_display_levels)
        self.btn_optimize_display_range.clicked.connect(self.optimize_display_levels)
        btn_save.clicked.connect(self.save_result)
        btn_save_psf.clicked.connect(self.save_current_psf)
        btn_redefine_psf.clicked.connect(self.redefine_psf)
        btn_save_mat.clicked.connect(self.save_mat)
        self.iter_slider.valueChanged.connect(self.show_iteration)
        self.display_independent_check.toggled.connect(self._on_display_normalization_changed)
        self.display_black_level_slider.valueChanged.connect(self._on_display_level_slider_changed)
        self.display_white_level_slider.valueChanged.connect(self._on_display_level_slider_changed)
        self.display_black_level_slider.sliderReleased.connect(self._on_display_level_slider_released)
        self.display_white_level_slider.sliderReleased.connect(self._on_display_level_slider_released)
        self._connect_algorithm_info_updates()
        self.update_selected_method_info()

    def _connect_algorithm_info_updates(self) -> None:
        """Refresh the Test tab description whenever the selected algorithm or parameters change."""
        for widget in self.alg_tab.findChildren((QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit)):
            try:
                if isinstance(widget, QComboBox):
                    widget.currentIndexChanged.connect(self.update_selected_method_info)
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.valueChanged.connect(self.update_selected_method_info)
                elif isinstance(widget, QCheckBox):
                    widget.toggled.connect(self.update_selected_method_info)
                elif isinstance(widget, QLineEdit):
                    widget.textChanged.connect(self.update_selected_method_info)
            except Exception:
                pass

    def update_selected_method_info(self, *args: Any) -> None:
        try:
            params = self.alg_tab.params()
            summary = self.alg_tab.important_params_summary(params)
        except Exception as exc:
            summary = f"unavailable ({exc})"
        self.selected_method_label.setText(f"Selected algorithm to run: {summary}")

    @staticmethod
    def _slider_fraction(slider: QSlider) -> float:
        span = max(1, slider.maximum() - slider.minimum())
        return float(slider.value() - slider.minimum()) / float(span)

    @staticmethod
    def _display_level_minimum_gap_steps(slider: QSlider) -> int:
        """Keep a small but visible nonzero interval between black and white."""
        span = max(1, slider.maximum() - slider.minimum())
        return max(1, int(round(span / 1024.0)))

    def _frame_display_histogram(self, frame: GrayImage) -> Dict[str, Any]:
        key = id(frame)
        cached = self._display_hist_cache.get(key)
        if cached is not None:
            return cached
        roi_source: Optional[GrayImage] = self.state.get("degraded")
        roi = crop_to_original_region(np.asarray(frame.data, dtype=np.float32), roi_source)
        cached = build_intensity_histogram(roi, bins=4096, value_range=(0.0, 1.0))
        self._display_hist_cache[key] = cached
        return cached

    def _common_display_histogram(self) -> Dict[str, Any]:
        if self._common_display_hist_cache is None:
            self._common_display_hist_cache = combine_intensity_histograms(
                [self._frame_display_histogram(frame) for frame in self.history]
            )
        return self._common_display_hist_cache

    def _active_display_histogram(self, current: GrayImage) -> Dict[str, Any]:
        if self.display_independent_check.isChecked():
            return self._frame_display_histogram(current)
        return self._common_display_histogram()

    @staticmethod
    def _histogram_domain(stats: Dict[str, Any]) -> Tuple[float, float]:
        """Use the complete configured display range rather than data min/max."""
        value_range = stats.get("range", (0.0, 1.0))
        try:
            low, high = float(value_range[0]), float(value_range[1])
        except Exception:
            low, high = 0.0, 1.0
        if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-12:
            low, high = 0.0, 1.0
        return low, high

    def _level_from_slider(self, slider: QSlider, current: GrayImage) -> float:
        stats = self._active_display_histogram(current)
        domain_low, domain_high = self._histogram_domain(stats)
        return float(domain_low + self._slider_fraction(slider) * (domain_high - domain_low))

    def _slider_value_for_level(self, slider: QSlider, current: GrayImage, level: float) -> int:
        stats = self._active_display_histogram(current)
        domain_low, domain_high = self._histogram_domain(stats)
        fraction = (float(level) - domain_low) / max(domain_high - domain_low, 1e-15)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return int(round(slider.minimum() + fraction * (slider.maximum() - slider.minimum())))

    def _current_display_levels(self, current: GrayImage) -> Tuple[float, float]:
        black = self._level_from_slider(self.display_black_level_slider, current)
        white = self._level_from_slider(self.display_white_level_slider, current)
        gap_steps = self._display_level_minimum_gap_steps(self.display_black_level_slider)
        if white <= black:
            stats = self._active_display_histogram(current)
            domain_low, domain_high = self._histogram_domain(stats)
            slider_span = max(1, self.display_black_level_slider.maximum() - self.display_black_level_slider.minimum())
            minimum_gap = max((domain_high - domain_low) * gap_steps / slider_span, 1e-8)
            white = min(domain_high, black + minimum_gap)
        return float(black), float(white)

    def _set_display_levels(self, current: GrayImage, black: float, white: float) -> None:
        black_value = self._slider_value_for_level(self.display_black_level_slider, current, black)
        white_value = self._slider_value_for_level(self.display_white_level_slider, current, white)
        gap = self._display_level_minimum_gap_steps(self.display_black_level_slider)
        if white_value < black_value + gap:
            white_value = min(self.display_white_level_slider.maximum(), black_value + gap)
            if white_value < black_value + gap:
                black_value = max(self.display_black_level_slider.minimum(), white_value - gap)
        self.display_black_level_slider.blockSignals(True)
        self.display_white_level_slider.blockSignals(True)
        self.display_black_level_slider.setValue(black_value)
        self.display_white_level_slider.setValue(white_value)
        self.display_black_level_slider.blockSignals(False)
        self.display_white_level_slider.blockSignals(False)
        self._update_display_level_labels(current)

    def _update_display_level_labels(self, current: Optional[GrayImage] = None) -> None:
        if current is None:
            if not self.history:
                self.display_black_level_label.setText("Black: -")
                self.display_white_level_label.setText("White: -")
                return
            idx = max(0, min(self.iter_slider.value() - 1, len(self.history) - 1))
            current = self.history[idx]
        black, white = self._current_display_levels(current)
        stats = self._active_display_histogram(current)
        black_p = histogram_percentile(stats, black)
        white_p = histogram_percentile(stats, white)
        self.display_black_level_label.setText(f"Black: {black:.6g}  (p{black_p:.2f})")
        self.display_white_level_label.setText(f"White: {white:.6g}  (p{white_p:.2f})")

    def _normalized_result_for_display(
        self, current: GrayImage, preview: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Clip only the displayed result; reconstruction data stay unchanged."""
        low, high = self._current_display_levels(current)
        data = np.asarray(current.data, dtype=np.float32)
        if preview:
            max_side = max(data.shape)
            step = max(1, int(np.ceil(max_side / 512.0)))
            if step > 1:
                data = data[::step, ::step]
        shown = (data - np.float32(low)) / np.float32(max(high - low, 1e-8))
        shown = np.clip(shown, 0.0, 1.0)
        return shown, float(low), float(high)

    def _on_display_level_slider_changed(self, *args: Any) -> None:
        if not self.history:
            return
        sender = self.sender()
        black_value = self.display_black_level_slider.value()
        white_value = self.display_white_level_slider.value()
        gap = self._display_level_minimum_gap_steps(self.display_black_level_slider)
        if white_value < black_value + gap:
            if sender is self.display_black_level_slider:
                self.display_white_level_slider.blockSignals(True)
                self.display_white_level_slider.setValue(
                    min(self.display_white_level_slider.maximum(), black_value + gap)
                )
                self.display_white_level_slider.blockSignals(False)
            else:
                self.display_black_level_slider.blockSignals(True)
                self.display_black_level_slider.setValue(
                    max(self.display_black_level_slider.minimum(), white_value - gap)
                )
                self.display_black_level_slider.blockSignals(False)
        self._update_display_level_labels()
        self._display_refresh_timer.start()

    def _on_display_level_slider_released(self) -> None:
        self._display_refresh_timer.stop()
        self._refresh_result_display_only(preview=False)

    def _refresh_result_display_only(self, preview: Optional[bool] = None) -> None:
        """Refresh only the result artist and level labels, not metrics or PSF."""
        if not self.history:
            return
        idx = max(0, min(self.iter_slider.value() - 1, len(self.history) - 1))
        current = self.history[idx]
        if preview is None:
            preview = bool(
                self.display_black_level_slider.isSliderDown()
                or self.display_white_level_slider.isSliderDown()
            )
        shown, black, white = self._normalized_result_for_display(current, preview=bool(preview))
        stats = self._active_display_histogram(current)
        black_p = histogram_percentile(stats, black)
        white_p = histogram_percentile(stats, white)
        mode = "independent" if self.display_independent_check.isChecked() else "common"
        metadata = dict(getattr(current, "metadata", {}) or {})
        kval = metadata.get("wiener_K")
        frame_name = f"K={float(kval):.6g}" if kval is not None else f"iteration {idx + 1}"
        preview_note = ", preview" if preview else ""
        self.result_canvas.show_image(
            shown,
            f"Result - {frame_name} ({mode}{preview_note}, {black:.4g}–{white:.4g}; p{black_p:.2f}–p{white_p:.2f})",
        )
        self.display_scale_label.setText(
            f"Display range: black={black:.6g} (p{black_p:.2f}), white={white:.6g} (p{white_p:.2f})"
        )

    def _on_display_normalization_changed(self, *args: Any) -> None:
        self._update_display_level_labels()
        if self.history:
            self._refresh_result_display_only(preview=False)

    def settings(self) -> Dict[str, Any]:
        return {
            "display_normalize_each_iteration": self.display_independent_check.isChecked(),
            "display_black_position": self._slider_fraction(self.display_black_level_slider),
            "display_white_position": self._slider_fraction(self.display_white_level_slider),
        }

    def apply_settings(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        if "display_normalize_each_iteration" in data:
            self.display_independent_check.setChecked(bool(data.get("display_normalize_each_iteration")))
        if "display_black_position" in data:
            fraction = float(np.clip(_safe_float(data.get("display_black_position"), 0.0), 0.0, 1.0))
        else:
            # Approximate migration from v89-v91 percentile-position settings.
            fraction = float(np.clip(_safe_float(data.get("display_min_percentile"), 0.0) / 100.0, 0.0, 1.0))
        if "display_white_position" in data:
            white_fraction = float(np.clip(_safe_float(data.get("display_white_position"), 0.97), 0.0, 1.0))
        else:
            white_fraction = float(np.clip(_safe_float(data.get("display_max_percentile"), 97.0) / 100.0, 0.0, 1.0))
        black_value = int(round(self.display_black_level_slider.minimum() + fraction * (self.display_black_level_slider.maximum() - self.display_black_level_slider.minimum())))
        white_value = int(round(self.display_white_level_slider.minimum() + white_fraction * (self.display_white_level_slider.maximum() - self.display_white_level_slider.minimum())))
        gap = self._display_level_minimum_gap_steps(self.display_black_level_slider)
        if white_value < black_value + gap:
            white_value = min(self.display_white_level_slider.maximum(), black_value + gap)
            if white_value < black_value + gap:
                black_value = max(self.display_black_level_slider.minimum(), white_value - gap)
        self.display_black_level_slider.blockSignals(True)
        self.display_white_level_slider.blockSignals(True)
        self.display_black_level_slider.setValue(black_value)
        self.display_white_level_slider.setValue(white_value)
        self.display_black_level_slider.blockSignals(False)
        self.display_white_level_slider.blockSignals(False)
        self._on_display_normalization_changed()

    def refresh_input_views(self) -> None:
        """Refresh Test-tab inputs without implying that measured data is a reference."""
        reference: Optional[GrayImage] = self.state.get("image")
        degraded: Optional[GrayImage] = self.state.get("degraded")
        has_reference = reference is not None and reference_metrics_available(self.state)
        self.reference_canvas.setVisible(has_reference)
        self.reference_canvas.show_image(reference.data if has_reference else None, "Reference image")
        self.input_canvas.show_image(degraded.data if degraded is not None else None, "Measured/degraded input")
        if not has_reference:
            self.metrics_label.setText("Original region — PSNR: n/a    SSIM: n/a    TV: -")

    def _quality_psf_for_frame(self, frame: Optional[GrayImage]) -> Optional[PSF]:
        if frame is not None and isinstance(getattr(frame, "metadata", None), dict):
            estimated = frame.metadata.get("estimated_psf")
            if estimated is not None:
                try:
                    return PSF(np.asarray(estimated, dtype=np.float64), name="estimated_psf_for_metrics")
                except Exception:
                    pass
        last_run = self.state.get("last_run_psf")
        if isinstance(last_run, PSF):
            return last_run
        current = self.state.get("calculation_psf")
        if isinstance(current, PSF):
            return current
        degraded = _calculation_image_from_state(self.state)
        shape = degraded.data.shape if isinstance(degraded, GrayImage) else None
        return _synchronize_calculation_psf(self.state, shape)

    def _metrics_for_frame(self, frame: Optional[GrayImage]) -> Dict[str, float]:
        if frame is None:
            return {}
        key = id(frame)
        cached = self._frame_metrics_cache.get(key)
        if cached is not None:
            return dict(cached)
        reference: Optional[GrayImage] = self.state.get("image")
        degraded: Optional[GrayImage] = self.state.get("degraded")
        metrics = compute_metrics(
            reference,
            frame,
            allow_reference_metrics=reference_metrics_available(self.state),
            roi_source=degraded,
            measured=degraded,
            psf=self._quality_psf_for_frame(frame),
        )
        metadata = dict(getattr(frame, "metadata", {}) or {})
        if metadata.get("wiener_gcv") is not None:
            metrics["WIENER_GCV"] = float(metadata.get("wiener_gcv"))
        if metadata.get("wiener_K") is not None:
            metrics["WIENER_K"] = float(metadata.get("wiener_K"))
        self._frame_metrics_cache[key] = dict(metrics)
        return metrics

    def _precompute_history_metrics_batch(self) -> None:
        """Populate the per-frame cache in one Torch/CUDA postprocessing pass."""
        if not self.history:
            return
        reference: Optional[GrayImage] = self.state.get("image")
        degraded: Optional[GrayImage] = self.state.get("degraded")
        psfs = [self._quality_psf_for_frame(frame) for frame in self.history]
        diagnostics: Dict[str, Any] = {}
        params = dict(self._run_context.get("params", {}) or {})
        prefer_cuda = bool(params.get("prefer_cuda", True))
        started = time.perf_counter()
        metrics_items = compute_metrics_batch(
            reference,
            self.history,
            allow_reference_metrics=reference_metrics_available(self.state),
            roi_source=degraded,
            measured=degraded,
            psfs=psfs,
            prefer_cuda=prefer_cuda,
            diagnostics=diagnostics,
        )
        for frame, metrics in zip(self.history, metrics_items):
            metadata = dict(getattr(frame, "metadata", {}) or {})
            if metadata.get("wiener_gcv") is not None:
                metrics["WIENER_GCV"] = float(metadata.get("wiener_gcv"))
            if metadata.get("wiener_K") is not None:
                metrics["WIENER_K"] = float(metadata.get("wiener_K"))
            self._frame_metrics_cache[id(frame)] = dict(metrics)
        elapsed = time.perf_counter() - started
        frames = int(diagnostics.get("frames", len(self.history)))
        batch_size = int(diagnostics.get("batch_size", 1))
        batches = int(diagnostics.get("batches", max(1, len(self.history))))
        mode = "one batch" if batches == 1 and batch_size >= frames else f"{batches} memory-safe batches"
        fallback = diagnostics.get("cuda_fallback") or diagnostics.get("reason")
        fallback_text = f"; fallback={fallback}" if fallback else ""
        self.log.append(
            f"Criteria postprocessing: {frames} stored frames in {mode}; "
            f"device={diagnostics.get('device', 'unknown')}; dtype={diagnostics.get('dtype', 'n/a')}; "
            f"PSF groups={diagnostics.get('psf_groups', 'n/a')}; time={elapsed:.3f} s{fallback_text}."
        )

    @staticmethod
    def _no_reference_metrics_text(metrics: Dict[str, float]) -> str:
        ntv = metrics.get("NTV", float("nan"))
        residual = metrics.get("RELATIVE_REBLUR_RESIDUAL", float("nan"))
        cost = metrics.get("NO_REFERENCE_COST", float("nan"))
        whiteness = metrics.get("RESIDUAL_WHITENESS", float("nan"))
        gcv = metrics.get("WIENER_GCV", float("nan"))
        kval = metrics.get("WIENER_K", float("nan"))
        ntv_text = f"NTV: {ntv:.6f}" if np.isfinite(ntv) else "NTV: n/a"
        residual_text = f"reblur residual: {residual:.6f}" if np.isfinite(residual) else "reblur residual: n/a"
        whiteness_text = f"residual whiteness: {whiteness:.6f}" if np.isfinite(whiteness) else "residual whiteness: n/a"
        cost_text = f"cost: {cost:.6f}" if np.isfinite(cost) else "cost: n/a"
        gcv_text = f"K={kval:.6g}    Wiener GCV: {gcv:.6g}" if np.isfinite(gcv) and np.isfinite(kval) else ""
        parts = [part for part in (gcv_text, ntv_text, residual_text, whiteness_text, cost_text) if part]
        return "    ".join(parts)

    def _run_is_active(self) -> bool:
        return bool(self.run_thread is not None and self.run_thread.isRunning())

    def run_algorithm(self) -> None:
        if self._run_is_active():
            QMessageBox.information(self, "Deconvolution running", "A deconvolution run is already in progress.")
            return
        active_owner = _current_numerical_owner()
        if active_owner is not None:
            QMessageBox.information(
                self,
                "Numerical task is running",
                f"Cannot start deconvolution while {active_owner} is still using the numerical backend.",
            )
            return

        reference: Optional[GrayImage] = self.state.get("image")
        psf: Optional[PSF] = self.state.get("psf")
        degraded: Optional[GrayImage] = self.state.get("degraded")
        alg = self.alg_tab.selected_algorithm()
        is_blind = isinstance(alg, (BlindRichardsonLucyDeconvolution, TorchBlindAdamTVMAPDeconvolution))

        if degraded is None and reference is None:
            QMessageBox.warning(self, "Missing data", "Load a measured/degraded image or generate a reference image first.")
            return
        if degraded is None:
            if psf is None:
                QMessageBox.warning(self, "Missing data", "Generate a degraded input first, or provide a PSF so it can be generated.")
                return
            degradation_psf = _synchronize_calculation_psf(self.state, reference.data.shape) or calculation_psf_for_image(psf, reference.data.shape)
            degraded = degrade_image(reference, degradation_psf, noise_sigma=0.01)
            self.state["degraded"] = degraded
            self.state["degradation_psf"] = degradation_psf
        if psf is None and not is_blind:
            QMessageBox.warning(self, "Missing data", "This algorithm requires a PSF.")
            return

        params = self.alg_tab.params()
        selected_summary = self.alg_tab.important_params_summary(params)
        self.selected_method_label.setText(f"Selected algorithm to run: {selected_summary}")
        run_psf = self.alg_tab._calculation_psf(psf, degraded.data.shape, params, algorithm_name=str(getattr(alg, "name", "")))
        self.state["last_run_calculation_shape"] = tuple(int(v) for v in degraded.data.shape)
        if run_psf is not None:
            self.state["last_run_psf"] = PSF(
                run_psf.kernel.copy(),
                name=f"{run_psf.name}_last_run",
                raw_kernel=run_psf.kernel.copy(),
            )
        else:
            self.state.pop("last_run_psf", None)

        self._run_context = {
            "reference": reference,
            "degraded": degraded,
            "run_psf": run_psf,
            "is_blind": is_blind,
            "algorithm": alg,
            "params": dict(params),
            "summary": selected_summary,
        }

        total = DeconvolutionRunWorker.planned_iterations(alg, params)
        self._run_current_iteration = 0
        self._run_total_iterations = total
        self.run_progress_bar.setRange(0, total)
        self.run_progress_bar.setValue(0)
        self.run_progress_label.setText(f"Current iteration: 0 / {total}")
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(total > 1)
        self.btn_select_best_iteration.setEnabled(False)
        self.btn_auto_display_range.setEnabled(False)
        self.btn_optimize_display_range.setEnabled(False)
        self.log.append(f"Started: {alg.name}; planned iterations={total}")

        thread = QThread(self)
        worker = DeconvolutionRunWorker(alg, degraded, run_psf, params)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_run_progress)
        worker.finished.connect(self._on_run_finished)
        worker.failed.connect(self._on_run_failed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._cleanup_run_thread)
        self.run_thread = thread
        self.run_worker = worker
        thread.start()

    def request_stop(self) -> None:
        if not self._run_is_active() or self.run_worker is None:
            return
        self.run_worker.request_stop()
        self.btn_stop.setEnabled(False)
        self.run_progress_label.setText(
            f"Stop requested — finishing iteration {max(1, self._run_current_iteration + 1)} / {self._run_total_iterations}"
        )
        self.log.append("Stop requested. The run will end after the current iteration.")

    def _on_run_progress(self, current: int, total: int) -> None:
        self._run_current_iteration = max(0, int(current))
        self._run_total_iterations = max(1, int(total))
        self.run_progress_bar.setRange(0, self._run_total_iterations)
        self.run_progress_bar.setValue(min(self._run_current_iteration, self._run_total_iterations))
        if self.run_worker is not None and self.run_worker.stop_event.is_set() and current < total:
            self.run_progress_label.setText(f"Stop requested — completed {current} / {total}")
        else:
            self.run_progress_label.setText(f"Current iteration: {current} / {total}")

    def _on_run_finished(self, result: DeconvolutionResult, stopped: bool, completed: int, total: int) -> None:
        context = dict(self._run_context)
        reference: Optional[GrayImage] = context.get("reference")
        degraded: Optional[GrayImage] = context.get("degraded")
        run_psf: Optional[PSF] = context.get("run_psf")
        is_blind = bool(context.get("is_blind", False))
        alg = context.get("algorithm")
        summary = str(context.get("summary", "-"))

        self.last_run_method_label.setText(f"Last run: {summary}")
        if result.image.metadata and result.image.metadata.get("initial_psf") is not None:
            initial_kernel = np.asarray(result.image.metadata.get("initial_psf"), dtype=np.float64)
            initial_obj = PSF(initial_kernel, name="blind_initial_psf")
            self.log.append(describe_psf_kernel(initial_obj, "Blind initial PSF"))
            if run_psf is not None:
                self.log.append(compare_psf_kernels(run_psf, initial_obj))
        self.history = result.history or [result.image]
        self.btn_select_best_iteration.setEnabled(bool(self.history))
        self.btn_auto_display_range.setEnabled(bool(self.history))
        self.btn_optimize_display_range.setEnabled(bool(self.history))
        self._display_hist_cache.clear()
        self._common_display_hist_cache = None
        self._frame_metrics_cache.clear()
        self.state["result"] = result.image
        if result.image.metadata and result.image.metadata.get("estimated_psf") is not None:
            self.state["estimated_psf"] = result.image.metadata.get("estimated_psf")
        else:
            self.state.pop("estimated_psf", None)

        has_reference = reference is not None and reference_metrics_available(self.state)
        self.reference_canvas.setVisible(has_reference)
        self.reference_canvas.show_image(reference.data if has_reference else None, "Reference image")
        self.input_canvas.show_image(degraded.data if degraded is not None else None, "Measured/degraded input")

        self.run_progress_label.setText(
            f"Postprocessing criteria in batch: {len(self.history)} stored frame(s) ..."
        )
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            self._precompute_history_metrics_batch()
        finally:
            QApplication.restoreOverrideCursor()

        best_idx = self.best_iteration_index(reference)
        self.iter_slider.blockSignals(True)
        self.iter_slider.setEnabled(bool(self.history))
        self.iter_slider.setRange(1, max(1, len(self.history)))
        self.iter_slider.setValue(best_idx + 1)
        self.iter_slider.blockSignals(False)
        self.show_iteration(best_idx + 1)
        # Complete the automatic postprocessing by applying robust display
        # levels to the selected best frame. This uses the cached 4096-bin CDF
        # and does not recompute reconstruction criteria.
        self.auto_set_display_levels()

        allow_ref = reference_metrics_available(self.state)
        best_metrics = self._metrics_for_frame(self.history[best_idx]) if self.history else {}
        if best_metrics:
            if allow_ref and np.isfinite(best_metrics.get("PSNR", float("nan"))):
                best_note = (
                    f"; best stored frame={best_idx + 1}; "
                    f"best PSNR={best_metrics['PSNR']:.3f} dB; "
                    f"best SSIM={best_metrics.get('SSIM', float('nan')):.4f}; "
                    f"best TV={best_metrics.get('TV', float('nan')):.6f}"
                )
            else:
                best_note = (
                    f"; best stored frame={best_idx + 1}; "
                    f"PSNR/SSIM not computed for measured input; "
                    f"best no-reference cost={best_metrics.get('NO_REFERENCE_COST', float('nan')):.6f}; "
                    f"Wiener K={best_metrics.get('WIENER_K', float('nan')):.6g}; "
                    f"Wiener GCV={best_metrics.get('WIENER_GCV', float('nan')):.6g}; "
                    f"NTV={best_metrics.get('NTV', float('nan')):.6f}; "
                    f"reblur residual={best_metrics.get('RELATIVE_REBLUR_RESIDUAL', float('nan')):.6f}; "
                    f"residual whiteness={best_metrics.get('RESIDUAL_WHITENESS', float('nan')):.6f}"
                )
        else:
            best_note = ""
        degradation_psf = self.state.get("degradation_psf")
        psf_note = ""
        if not is_blind:
            psf_note = (
                "\n" + describe_psf_kernel(degradation_psf, "Degradation PSF") +
                "\n" + describe_psf_kernel(run_psf, "Reconstruction PSF") +
                "\n" + compare_psf_kernels(degradation_psf, run_psf)
            )
        status = f"stopped after iteration {completed}/{total}" if stopped else f"completed {completed}/{total}"
        alg_name = str(getattr(alg, "name", "algorithm"))
        self.log.append(f"Run: {alg_name}; {status}; {result.info}; history frames={len(self.history)}{best_note}{psf_note}")
        self.run_progress_bar.setValue(min(max(0, completed), max(1, total)))
        self.run_progress_label.setText(f"Finished: {completed} / {total}" + (" (stopped)" if stopped else ""))
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_select_best_iteration.setEnabled(bool(self.history))
        self.btn_auto_display_range.setEnabled(bool(self.history))
        self.btn_optimize_display_range.setEnabled(bool(self.history))

    def _on_run_failed(self, message: str) -> None:
        self.log.append(f"Deconvolution error: {message}")
        QMessageBox.critical(self, "Deconvolution error", message)
        self.run_progress_label.setText("Run failed")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_select_best_iteration.setEnabled(bool(self.history))
        self.btn_auto_display_range.setEnabled(bool(self.history))
        self.btn_optimize_display_range.setEnabled(bool(self.history))

    def _cleanup_run_thread(self) -> None:
        thread = self.run_thread
        self.run_worker = None
        self.run_thread = None
        self._run_context = {}
        if thread is not None:
            thread.deleteLater()

    def cancel_run_and_wait(self, timeout_ms: int = 5000) -> bool:
        """Request cooperative cancellation and report whether the thread stopped."""
        if self.run_worker is not None:
            self.run_worker.request_stop()
        thread = self.run_thread
        if thread is not None:
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(max(0, int(timeout_ms)))
                return not thread.isRunning()
            except RuntimeError:
                return True
        return True

    def best_iteration_index(self, reference: Optional[GrayImage]) -> int:
        if not self.history:
            return 0
        allow_ref = reference_metrics_available(self.state)
        if not allow_ref:
            gcv_values = np.asarray([
                float((getattr(frame, "metadata", {}) or {}).get("wiener_gcv", np.nan))
                for frame in self.history
            ], dtype=np.float64)
            if np.isfinite(gcv_values).any():
                return int(np.nanargmin(gcv_values))
        scores = []
        for frame in self.history:
            metrics = self._metrics_for_frame(frame)
            scores.append(metric_score(metrics))
        return int(np.nanargmax(scores)) if scores else max(0, len(self.history) - 1)

    def select_best_iteration(self) -> None:
        """Return the iteration browser to the best stored frame."""
        if not self.history:
            QMessageBox.information(self, "No iteration history", "Run a deconvolution algorithm first.")
            return
        best_idx = self.best_iteration_index(self.state.get("image"))
        self.iter_slider.setValue(best_idx + 1)
        # valueChanged is not emitted if the best frame was already selected.
        self.show_iteration(best_idx + 1)
        frame = self.history[best_idx]
        metrics = self._metrics_for_frame(frame)
        metadata = dict(getattr(frame, "metadata", {}) or {})
        gcv = float(metadata.get("wiener_gcv", float("nan")))
        if not reference_metrics_available(self.state) and np.isfinite(gcv):
            criterion = f"minimum Wiener GCV {gcv:.6g}"
        else:
            score = metric_score(metrics)
            criterion = score_description(score, reference_metrics_available(self.state))
        self.log.append(f"Selected best stored frame {best_idx + 1}/{len(self.history)}: {criterion}.")

    @staticmethod
    def _downsample_for_level_optimization(data: np.ndarray, max_side: int = 192) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        step = max(1, int(np.ceil(max(arr.shape) / float(max(16, max_side)))))
        return np.ascontiguousarray(arr[::step, ::step], dtype=np.float32)

    def auto_set_display_levels(self) -> None:
        """Instantly choose robust black/white intensities from cached CDF data."""
        if not self.history:
            QMessageBox.information(self, "No iteration history", "Run a deconvolution algorithm first.")
            return
        idx = max(0, min(self.iter_slider.value() - 1, len(self.history) - 1))
        current = self.history[idx]
        stats = self._active_display_histogram(current)
        black = histogram_quantile(stats, 0.5)
        white = histogram_quantile(stats, 99.5)
        if white <= black + 1e-8:
            black = histogram_quantile(stats, 0.0)
            white = histogram_quantile(stats, 100.0)
        self._set_display_levels(current, black, white)
        self._refresh_result_display_only(preview=False)
        self.log.append(
            f"Auto levels for stored frame {idx + 1}: black={black:.8g} (p0.5), "
            f"white={white:.8g} (p99.5)."
        )

    def optimize_display_levels(self) -> None:
        """Optimize a small intensity-level candidate set on a downsampled ROI."""
        if not self.history:
            QMessageBox.information(self, "No iteration history", "Run a deconvolution algorithm first.")
            return
        if self._run_is_active():
            QMessageBox.information(
                self, "Deconvolution running", "Wait until the current deconvolution run has finished."
            )
            return

        idx = max(0, min(self.iter_slider.value() - 1, len(self.history) - 1))
        current = self.history[idx]
        stats = self._active_display_histogram(current)
        current_black, current_white = self._current_display_levels(current)
        degraded: Optional[GrayImage] = self.state.get("degraded")
        reference: Optional[GrayImage] = self.state.get("image")

        current_roi = crop_to_original_region(np.asarray(current.data, dtype=np.float32), degraded)
        current_small = self._downsample_for_level_optimization(current_roi, max_side=192)
        reference_small: Optional[np.ndarray] = None
        if reference is not None and reference_metrics_available(self.state):
            ref_roi = crop_to_original_region(np.asarray(reference.data, dtype=np.float32), reference)
            reference_small = self._downsample_for_level_optimization(ref_roi, max_side=192)
            if reference_small.shape != current_small.shape:
                reference_small = GrayImage.resize_array(
                    reference_small, width=current_small.shape[1], height=current_small.shape[0]
                ).astype(np.float32)

        measured_small_image: Optional[GrayImage] = None
        if reference_small is None and degraded is not None:
            measured_roi = crop_to_original_region(np.asarray(degraded.data, dtype=np.float32), degraded)
            measured_small = self._downsample_for_level_optimization(measured_roi, max_side=192)
            if measured_small.shape != current_small.shape:
                measured_small = GrayImage.resize_array(
                    measured_small, width=current_small.shape[1], height=current_small.shape[0]
                ).astype(np.float32)
            measured_small_image = GrayImage(
                measured_small,
                name="measured_display_optimization",
                metadata={"_preserve_intensity": True},
            )

        psf_for_quality = self._quality_psf_for_frame(current)
        self.btn_auto_levels.setEnabled(False)
        self.btn_optimize_display_range.setEnabled(False)
        self.btn_select_best_iteration.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        evaluations = 0

        def objective(black: float, white: float) -> float:
            nonlocal evaluations
            evaluations += 1
            shown = (current_small - np.float32(black)) / np.float32(max(white - black, 1e-8))
            shown = np.clip(shown, 0.0, 1.0)
            if reference_small is not None:
                mse = float(np.mean((reference_small - shown) ** 2, dtype=np.float64))
                return 300.0 if mse <= 1e-30 else float(10.0 * np.log10(1.0 / mse))
            candidate = GrayImage(
                shown,
                name=f"{current.name}_display_level_candidate",
                metadata={"_preserve_intensity": True},
            )
            metrics = compute_metrics(
                None,
                candidate,
                allow_reference_metrics=False,
                roi_source=measured_small_image,
                measured=measured_small_image,
                psf=psf_for_quality,
            )
            if evaluations % 8 == 0:
                QApplication.processEvents()
            return metric_score(metrics)

        try:
            black, white, score, evaluated = optimize_intensity_levels(
                objective,
                quantile_function=lambda p: histogram_quantile(stats, p),
                current_low=current_black,
                current_high=current_white,
            )
            self._set_display_levels(current, black, white)
            self._refresh_result_display_only(preview=False)
            criterion = (
                f"PSNR={score:.3f} dB"
                if reference_small is not None
                else f"no-reference cost={-score:.6f}"
            )
            black_p = histogram_percentile(stats, black)
            white_p = histogram_percentile(stats, white)
            self.log.append(
                f"Optimized display levels for stored frame {idx + 1}: "
                f"black={black:.8g} (p{black_p:.2f}), white={white:.8g} (p{white_p:.2f}); "
                f"{criterion}; downsampled to {current_small.shape[1]}x{current_small.shape[0]}; "
                f"evaluations={evaluated}."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Display-level optimization error", f"{type(exc).__name__}: {exc}")
            self.log.append(f"Display-level optimization failed: {type(exc).__name__}: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
            enabled = bool(self.history)
            self.btn_auto_levels.setEnabled(enabled)
            self.btn_optimize_display_range.setEnabled(enabled)
            self.btn_select_best_iteration.setEnabled(enabled)

    # Backward-compatible method name from v89-v91.  It now invokes the fast CDF method.
    def auto_set_percentile_range(self) -> None:
        self.auto_set_display_levels()

    # Backward-compatible name used in older versions.
    def best_iteration_index_by_psnr(self, reference: Optional[GrayImage]) -> int:
        return self.best_iteration_index(reference)

    def show_iteration(self, value: int) -> None:
        if not self.history:
            return
        self._display_refresh_timer.stop()
        idx = max(0, min(value - 1, len(self.history) - 1))
        current = self.history[idx]
        self.state["result"] = current
        self._update_display_level_labels(current)
        self._refresh_result_display_only(preview=False)

        metadata = dict(getattr(current, "metadata", {}) or {})
        kval = metadata.get("wiener_K")
        self.btn_use_displayed_k.setEnabled(kval is not None)
        estimated_psf = current.metadata.get("estimated_psf") if current.metadata else None
        if estimated_psf is None:
            estimated_psf = self.state.get("estimated_psf")
        self.estimated_psf_canvas.show_image(estimated_psf, "Estimated PSF", normalize_display=True)
        if kval is not None:
            gcv = metadata.get("wiener_gcv", float("nan"))
            gcv_text = f", GCV={float(gcv):.6g}" if np.isfinite(float(gcv)) else ""
            self.iter_label.setText(f"K scan: {idx + 1}/{len(self.history)}, K={float(kval):.6g}{gcv_text}")
        else:
            self.iter_label.setText(f"Iteration: {idx + 1}/{len(self.history)}")
        allow_ref = reference_metrics_available(self.state)
        metrics = self._metrics_for_frame(current)
        tv_text = f"TV: {metrics.get('TV', float('nan')):.6f}" if np.isfinite(metrics.get('TV', float('nan'))) else "TV: n/a"
        if allow_ref and np.isfinite(metrics.get("PSNR", float("nan"))):
            self.metrics_label.setText(f"Original region — PSNR: {metrics['PSNR']:.3f} dB    SSIM: {metrics.get('SSIM', float('nan')):.4f}    {tv_text}")
        else:
            self.metrics_label.setText(f"Original region — PSNR: n/a    SSIM: n/a    {self._no_reference_metrics_text(metrics)}")

    def use_displayed_wiener_k(self) -> None:
        if not self.history:
            return
        idx = max(0, min(self.iter_slider.value() - 1, len(self.history) - 1))
        metadata = dict(getattr(self.history[idx], "metadata", {}) or {})
        kval = metadata.get("wiener_K")
        if kval is None:
            QMessageBox.information(self, "No scanned K", "The displayed result is not part of a Wiener K scan.")
            return
        value = float(kval)
        self.alg_tab.k_spin.setValue(value)
        self.alg_tab._remember_current_algorithm_params()
        self.alg_tab.auto_status.setText(f"Wiener K copied from Test frame: K={value:.6g}")
        self.update_selected_method_info()
        self.log.append(f"Copied displayed Wiener scan value to algorithm settings: K={value:.12g}")

    def reset_runtime_view(self) -> None:
        self.history = []
        self._run_context = {}
        self._run_current_iteration = 0
        self._run_total_iterations = 0
        self._display_refresh_timer.stop()
        self._display_hist_cache.clear()
        self._common_display_hist_cache = None
        self._frame_metrics_cache.clear()
        self.iter_slider.setEnabled(False)
        self.iter_slider.setRange(1, 1)
        self.iter_slider.setValue(1)
        self.iter_label.setText("Iteration: -")
        self.metrics_label.setText("Original region — PSNR: -    SSIM: -    TV: -")
        self.display_scale_label.setText("Display range: -")
        self.display_black_level_label.setText("Black: -")
        self.display_white_level_label.setText("White: -")
        self.reference_canvas.show_image(None, "Reference image")
        self.input_canvas.show_image(None, "Measured/degraded input")
        self.result_canvas.show_image(None, "Result")
        self.estimated_psf_canvas.show_image(None, "Estimated PSF")
        self.last_run_method_label.setText("Last run: -")
        self.run_progress_label.setText("Current iteration: - / -")
        self.run_progress_bar.setRange(0, 1)
        self.run_progress_bar.setValue(0)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_use_displayed_k.setEnabled(False)
        self.btn_select_best_iteration.setEnabled(False)
        self.btn_auto_display_range.setEnabled(False)
        self.btn_optimize_display_range.setEnabled(False)
        self.log.clear()

    def save_result(self) -> None:
        """Save the current reconstruction as 8-bit, 16-bit, or MATLAB data."""
        result: Optional[GrayImage] = self.state.get("result")
        if result is None:
            QMessageBox.warning(self, "No result", "Run an algorithm first.")
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save current result",
            "result.png",
            "8-bit PNG (*.png);;16-bit PNG (*.png);;8-bit TIFF (*.tif *.tiff);;16-bit TIFF (*.tif *.tiff);;MATLAB MAT (*.mat)",
        )
        if not path:
            return
        try:
            arr = np.clip(np.asarray(result.data, dtype=np.float64), 0.0, 1.0)
            if "MATLAB" in selected_filter or path.lower().endswith(".mat"):
                if not path.lower().endswith(".mat"):
                    path += ".mat"
                savemat(
                    path,
                    {
                        "result": arr,
                        "method": np.array([self.alg_tab.combo.currentText()], dtype=object),
                        "parameters_json": np.array([json.dumps(self.alg_tab.params(), indent=2)], dtype=object),
                    },
                    do_compression=True,
                )
                mode = "MAT float64"
            else:
                is_tiff = "TIFF" in selected_filter or path.lower().endswith((".tif", ".tiff"))
                if not path.lower().endswith((".png", ".tif", ".tiff")):
                    path += ".tif" if is_tiff else ".png"
                if "16-bit" in selected_filter:
                    Image.fromarray(np.rint(arr * 65535.0).astype(np.uint16)).save(path)
                    mode = "16-bit"
                else:
                    Image.fromarray(np.rint(arr * 255.0).astype(np.uint8), mode="L").save(path)
                    mode = "8-bit"
        except Exception as exc:
            QMessageBox.warning(self, "Result save error", f"Could not save the result:\n{exc}")
            return
        self.log.append(f"Saved current result: {path}; format={mode}; shape={arr.shape[1]}x{arr.shape[0]}")

    def _current_psf_for_saving(self) -> Tuple[Optional[np.ndarray], str]:
        """Return the PSF corresponding to the currently displayed result.

        Blind methods are preferred because their PSF changes during the run.
        For non-blind methods, the exact calculation PSF retained at the last
        invocation is used. The returned array is not resized or display-cropped.
        """
        idx = 0
        if self.history:
            idx = max(0, min(self.iter_slider.value() - 1, len(self.history) - 1))
            frame = self.history[idx]
            if frame.metadata:
                estimated = frame.metadata.get("estimated_psf")
                if estimated is not None:
                    return np.asarray(estimated, dtype=np.float64), f"estimated PSF, iteration {idx + 1}"

        # Some blind algorithms store a PSF stack only in the final result.
        result: Optional[GrayImage] = self.state.get("result")
        if result is not None and result.metadata:
            psf_history = result.metadata.get("estimated_psf_history")
            if psf_history is not None:
                stack = np.asarray(psf_history, dtype=np.float64)
                if stack.ndim == 3 and stack.shape[0] > 0:
                    history_idx = min(idx, stack.shape[0] - 1)
                    return stack[history_idx], f"estimated PSF history, frame {history_idx + 1}"
            estimated = result.metadata.get("estimated_psf")
            if estimated is not None:
                return np.asarray(estimated, dtype=np.float64), "final estimated PSF"

        estimated = self.state.get("estimated_psf")
        if estimated is not None:
            return np.asarray(estimated, dtype=np.float64), "final estimated PSF"

        last_run_psf: Optional[PSF] = self.state.get("last_run_psf")
        if last_run_psf is not None:
            return last_run_psf.kernel.copy(), "PSF used by the last deconvolution run"

        calculation_psf: Optional[PSF] = self.state.get("calculation_psf")
        if calculation_psf is None:
            degraded = _calculation_image_from_state(self.state)
            shape = degraded.data.shape if isinstance(degraded, GrayImage) else None
            calculation_psf = _synchronize_calculation_psf(self.state, shape)
        if calculation_psf is not None:
            return calculation_psf.kernel.copy(), "current thresholded, cropped and normalized calculation PSF"

        return None, ""

    def _calculation_shape_for_psf_save(self) -> Tuple[int, int]:
        """Return the actual HxW resolution used for the latest calculation."""
        saved = self.state.get("last_run_calculation_shape")
        if isinstance(saved, (tuple, list)) and len(saved) == 2:
            h, w = int(saved[0]), int(saved[1])
            if h > 0 and w > 0:
                return h, w

        for key in ("degraded", "result", "image"):
            item = self.state.get(key)
            if isinstance(item, GrayImage) and item.data.ndim == 2 and item.data.size:
                return int(item.data.shape[0]), int(item.data.shape[1])

        fallback = self.state.get("calculation_image_shape", (256, 256))
        try:
            h, w = int(fallback[0]), int(fallback[1])
        except Exception:
            h, w = 256, 256
        return max(1, h), max(1, w)

    @staticmethod
    def _embed_psf_on_calculation_canvas(psf_array: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Place a PSF on a zero canvas at the full calculation resolution.

        The PSF support center is aligned with the geometric center of the
        calculation canvas. No interpolation is performed: the numerical PSF
        samples remain unchanged, and only zero-valued pixels are added. If a
        supplied PSF is larger than the canvas, the centered overlapping part is
        retained.
        """
        arr = np.asarray(psf_array, dtype=np.float64)
        arr = np.maximum(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        target_h, target_w = max(1, int(shape[0])), max(1, int(shape[1]))
        canvas = np.zeros((target_h, target_w), dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return canvas

        source_cy, source_cx = PSF.support_center(arr)
        target_cy, target_cx = target_h // 2, target_w // 2
        destination_top = target_cy - int(source_cy)
        destination_left = target_cx - int(source_cx)

        source_y0 = max(0, -destination_top)
        source_x0 = max(0, -destination_left)
        source_y1 = min(arr.shape[0], target_h - destination_top)
        source_x1 = min(arr.shape[1], target_w - destination_left)
        if source_y1 <= source_y0 or source_x1 <= source_x0:
            return canvas

        destination_y0 = destination_top + source_y0
        destination_x0 = destination_left + source_x0
        destination_y1 = destination_y0 + (source_y1 - source_y0)
        destination_x1 = destination_x0 + (source_x1 - source_x0)
        canvas[destination_y0:destination_y1, destination_x0:destination_x1] = arr[
            source_y0:source_y1, source_x0:source_x1
        ]
        return canvas

    def redefine_psf(self) -> None:
        """Replace the current known PSF with the PSF of the displayed result."""
        psf_array, source = self._current_psf_for_saving()
        if psf_array is None:
            QMessageBox.warning(self, "No PSF", "No current, used, or estimated PSF is available.")
            return
        kernel = np.maximum(np.nan_to_num(np.asarray(psf_array, dtype=np.float64)), 0.0)
        if kernel.ndim != 2 or kernel.size == 0 or float(kernel.max()) <= 0.0:
            QMessageBox.warning(self, "Invalid PSF", "The current PSF is not a valid positive two-dimensional array.")
            return
        calculation_shape = self._calculation_shape_for_psf_save()
        full = self._embed_psf_on_calculation_canvas(kernel, calculation_shape)
        peak = float(full.max())
        if peak <= 0.0:
            QMessageBox.warning(self, "Invalid PSF", "The PSF could not be placed on the calculation canvas.")
            return
        full_peak_normalized = np.clip(full / peak, 0.0, 1.0)
        new_psf = PSF(
            full_peak_normalized,
            name="redefined_from_current_result",
            raw_kernel=full_peak_normalized.copy(),
            metadata={"source_description": source, "calculation_shape": calculation_shape},
        )
        compact = PSF(kernel, name="redefined_compact_calculation_psf", raw_kernel=kernel.copy())
        self.state["psf"] = new_psf
        # Exact-PSF mode must use the newly redefined PSF rather than an older
        # synthetic degradation kernel retained in application state.
        self.state["degradation_psf"] = compact
        self.state["last_run_psf"] = compact
        self.psfRedefined.emit()
        self.log.append(
            f"Redefined current PSF from {source}; full canvas={calculation_shape[1]}x{calculation_shape[0]}; "
            f"compact kernel={kernel.shape[1]}x{kernel.shape[0]}."
        )

    def save_current_psf(self) -> None:
        """Save the current PSF on a full calculation-resolution canvas."""
        psf_array, source = self._current_psf_for_saving()
        if psf_array is None:
            QMessageBox.warning(
                self,
                "No PSF",
                "No current, used, or estimated PSF is available to save.",
            )
            return

        kernel = np.asarray(psf_array, dtype=np.float64)
        if kernel.ndim != 2 or kernel.size == 0:
            QMessageBox.warning(self, "Invalid PSF", "The current PSF is not a valid two-dimensional image.")
            return
        kernel = np.maximum(np.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        kernel_peak = float(kernel.max())
        if kernel_peak <= 0.0:
            QMessageBox.warning(self, "Invalid PSF", "The current PSF has no positive intensity values.")
            return

        calculation_shape = self._calculation_shape_for_psf_save()
        full_resolution_psf = self._embed_psf_on_calculation_canvas(kernel, calculation_shape)
        peak = float(full_resolution_psf.max())
        if peak <= 0.0:
            QMessageBox.warning(self, "Invalid PSF", "The PSF could not be placed on the calculation canvas.")
            return
        normalized = np.clip(full_resolution_psf / peak, 0.0, 1.0)

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save current PSF",
            "current_psf_full_resolution.png",
            "8-bit PNG (*.png);;16-bit PNG (*.png);;8-bit TIFF (*.tif *.tiff);;16-bit TIFF (*.tif *.tiff);;MATLAB MAT (*.mat)",
        )
        if not path:
            return
        try:
            if "MATLAB" in selected_filter or path.lower().endswith(".mat"):
                if not path.lower().endswith(".mat"):
                    path += ".mat"
                savemat(
                    path,
                    {
                        "psf": normalized,
                        "psf_kernel": kernel,
                        "psf_kernel_peak_normalized": kernel / kernel_peak,
                        "calculation_shape_yx": np.asarray(calculation_shape, dtype=np.int32),
                        "source_description": np.array([source], dtype=object),
                    },
                    do_compression=True,
                )
                mode = "MAT float64"
            else:
                is_tiff = "TIFF" in selected_filter or path.lower().endswith((".tif", ".tiff"))
                if not path.lower().endswith((".png", ".tif", ".tiff")):
                    path += ".tif" if is_tiff else ".png"
                if "16-bit" in selected_filter:
                    Image.fromarray(np.rint(normalized * 65535.0).astype(np.uint16)).save(path)
                    mode = "16-bit"
                else:
                    Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), mode="L").save(path)
                    mode = "8-bit"
        except Exception as exc:
            QMessageBox.warning(self, "PSF save error", f"Could not save the PSF:\n{exc}")
            return

        self.log.append(
            f"Saved current PSF: {path}; format={mode}; source={source}; "
            f"kernel shape={kernel.shape[1]}x{kernel.shape[0]}; "
            f"saved calculation canvas={calculation_shape[1]}x{calculation_shape[0]}; "
            f"original kernel sum={kernel.sum():.8g}; saved peak normalization=1"
        )

    def save_mat(self) -> None:
        """Save the current Test tab data to a MATLAB .mat file."""
        if self.state.get("result") is None and not self.history:
            QMessageBox.warning(self, "No result", "Run an algorithm first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save test images to MAT", "deconvolution_test.mat", "MATLAB MAT (*.mat)")
        if not path:
            return
        if not path.lower().endswith(".mat"):
            path += ".mat"
        reference: Optional[GrayImage] = self.state.get("image")
        degraded: Optional[GrayImage] = self.state.get("degraded")
        result: Optional[GrayImage] = self.state.get("result")
        degradation_psf: Optional[PSF] = self.state.get("degradation_psf")
        current_psf: Optional[PSF] = self.state.get("psf")
        estimated_psf = None
        estimated_psf_history = np.empty((0, 0, 0))
        if result is not None and result.metadata:
            estimated_psf = result.metadata.get("estimated_psf")
            estimated_psf_history = result.metadata.get("estimated_psf_history", estimated_psf_history)
        if estimated_psf is None:
            estimated_psf = self.state.get("estimated_psf")
        history_stack = np.stack([frame.data for frame in self.history], axis=0) if self.history else np.empty((0, 0, 0))
        wiener_k_history = np.asarray([
            float((getattr(frame, "metadata", {}) or {}).get("wiener_K", np.nan))
            for frame in self.history
        ], dtype=np.float64)
        wiener_gcv_history = np.asarray([
            float((getattr(frame, "metadata", {}) or {}).get("wiener_gcv", np.nan))
            for frame in self.history
        ], dtype=np.float64)
        params = self.alg_tab.params()
        metrics = self._metrics_for_frame(result) if result is not None else {}
        mat_data = {
            "reference": reference.data if reference is not None else np.empty((0, 0)),
            "degraded": degraded.data if degraded is not None else np.empty((0, 0)),
            "result": result.data if result is not None else np.empty((0, 0)),
            "history": history_stack,
            "wiener_k_history": wiener_k_history,
            "wiener_gcv_history": wiener_gcv_history,
            "degradation_psf": degradation_psf.kernel if degradation_psf is not None else np.empty((0, 0)),
            "current_psf": current_psf.kernel if current_psf is not None else np.empty((0, 0)),
            "estimated_psf": np.asarray(estimated_psf) if estimated_psf is not None else np.empty((0, 0)),
            "estimated_psf_history": np.asarray(estimated_psf_history),
            "iteration_index": np.array([[self.iter_slider.value()]], dtype=np.int32),
            "method": np.array([self.alg_tab.combo.currentText()], dtype=object),
            "parameter_summary": np.array([self.alg_tab.important_params_summary(params)], dtype=object),
            "parameters_json": np.array([json.dumps(params, indent=2)], dtype=object),
            "psnr": np.array([[metrics.get("PSNR", np.nan)]], dtype=np.float64),
            "ssim": np.array([[metrics.get("SSIM", np.nan)]], dtype=np.float64),
            "tv_norm": np.array([[metrics.get("TV", np.nan)]], dtype=np.float64),
            "normalized_tv": np.array([[metrics.get("NTV", np.nan)]], dtype=np.float64),
            "relative_reblur_residual": np.array([[metrics.get("RELATIVE_REBLUR_RESIDUAL", np.nan)]], dtype=np.float64),
            "residual_whiteness": np.array([[metrics.get("RESIDUAL_WHITENESS", np.nan)]], dtype=np.float64),
            "wiener_k": np.array([[metrics.get("WIENER_K", np.nan)]], dtype=np.float64),
            "wiener_gcv": np.array([[metrics.get("WIENER_GCV", np.nan)]], dtype=np.float64),
            "no_reference_cost": np.array([[metrics.get("NO_REFERENCE_COST", np.nan)]], dtype=np.float64),
            "metric_content_roi": np.array([list((degraded.metadata.get("content_roi") if degraded is not None else None) or (0, 0, 0, 0))], dtype=np.int32),
            "metrics_use_original_region": np.array([[1]], dtype=np.uint8),
            "visible_zero_padding_enabled": np.array([[1 if self.state.get("zero_padding_enabled", False) else 0]], dtype=np.uint8),
            "reference_metrics_available": np.array([[1 if reference_metrics_available(self.state) else 0]], dtype=np.uint8),
        }
        try:
            savemat(path, mat_data, do_compression=True)
        except Exception as exc:
            QMessageBox.warning(self, "MAT save error", f"Could not save MAT file:\n{exc}")
            return
        self.log.append(f"Saved MAT: {path}")


def _saved_gui_language() -> str:
    """Read only the language field before constructing the Qt widget tree."""
    for path in (_active_settings_file(), SETTINGS_FILE, LEGACY_SETTINGS_FILE, PACKAGE_LEGACY_SETTINGS_FILE):
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                return str(data.get("language", "en"))
        except Exception:
            continue
    return "en"


class DeconvolutionMainWindow(QMainWindow):
    def __init__(self) -> None:
        set_language(_saved_gui_language())
        super().__init__()
        self.setStatusBar(QStatusBar(self))
        self.setWindowTitle(translate("Grayscale Image Deconvolution Test Environment"))
        self.resize(1300, 800)
        self.state: Dict[str, Any] = {}
        self.registry = AlgorithmRegistry()
        self.current_settings_file: Path = _active_settings_file()

        tabs = QTabWidget()
        self.tabs = tabs
        self.load_tab = LoadGenerateTab(self.state)
        self.degraded_tab = DegradedInputTab(self.state)
        self.alg_tab = AlgorithmTab(self.state, self.registry)
        self.load_tab.calculationPsfSupportChanged.connect(self.alg_tab.update_known_psf_support_info)
        self.degraded_tab.calculationPsfSupportChanged.connect(self.alg_tab.update_known_psf_support_info)
        self.degraded_tab.wienerKOptimized.connect(self.alg_tab.set_wiener_k_from_tab2)
        self.degraded_tab.calculationDataChanged.connect(self.load_tab.refresh_calculation_views)
        self.degraded_tab.calculationDataChanged.connect(self.alg_tab._update_psf_policy_label)
        self.alg_tab.apply_resolution_linked_psf_defaults(
            int(self.state.get("resolution_linked_psf_support", self.load_tab.recommended_psf_support()))
        )
        self.alg_tab.update_known_psf_support_info(self.load_tab.current_psf_support_width(self.state.get("psf")))
        self.test_tab = TestTab(self.state, self.alg_tab)
        self.load_tab.resetRequested.connect(self.reset_application)
        self.load_tab.exitRequested.connect(self.close)
        self.test_tab.psfRedefined.connect(self._on_psf_redefined)
        tabs.addTab(self.load_tab, "1. Image and PSF")
        tabs.addTab(self.degraded_tab, "2. Degraded input")
        tabs.addTab(self.alg_tab, "3. Algorithm")
        tabs.addTab(self.test_tab, "4. Test")
        tabs.currentChanged.connect(
            lambda _: (self.load_tab.refresh_calculation_views(), self.degraded_tab.refresh(), self.alg_tab._update_psf_policy_label(), self.test_tab.refresh_input_views())
        )
        self.setCentralWidget(tabs)
        self._create_settings_menu()
        self._create_language_menu()
        # Capture untouched widget defaults before loading any profile.
        self.default_settings = self.collect_settings()
        self.load_settings()
        self.save_settings()

    def _on_psf_redefined(self) -> None:
        psf = self.state.get("psf")
        if isinstance(psf, PSF):
            self.load_tab._initialize_tab2_psf_selection(psf)
            image = _calculation_image_from_state(self.state) or self.state.get("image")
            shape = image.data.shape if isinstance(image, GrayImage) else psf.kernel.shape
            _synchronize_calculation_psf(self.state, shape)
        self.load_tab.refresh_calculation_views()
        self.degraded_tab.refresh()
        self.alg_tab._update_psf_policy_label()
        self.statusBar().showMessage("Current PSF was redefined from the Test-tab result and applied as the calculation PSF.", 6000)

    def reset_application(self) -> None:
        answer = QMessageBox.question(
            self,
            "Reset application",
            "Reset all parameters to their defaults and clear loaded images, PSFs, histories and results?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        auto_stopped = True
        run_stopped = True
        try:
            auto_stopped = self.alg_tab.cancel_auto_workers(8000)
        except Exception:
            auto_stopped = False
        try:
            run_stopped = self.test_tab.cancel_run_and_wait(30000)
        except Exception:
            run_stopped = False
        if not (auto_stopped and run_stopped):
            QMessageBox.warning(
                self,
                "Numerical task still running",
                "Reset was postponed because a numerical worker has not yet reached a safe stopping point.",
            )
            return
        self.state.clear()
        reset_settings = json.loads(json.dumps(self.default_settings))
        reset_settings["language"] = get_language()
        self._apply_settings_data(reset_settings)
        self.load_tab.image_canvas.setVisible(True)
        self.load_tab.image_canvas.show_image(None, "Reference image")
        self.load_tab.psf_canvas.show_image(None, "PSF")
        self.load_tab.degraded_canvas.show_image(None, "Degraded input")
        self.load_tab.refresh_calculation_views()
        self.degraded_tab.reset_lower_thresholds()
        self.degraded_tab.refresh()
        self.test_tab.reset_runtime_view()
        self.test_tab.update_selected_method_info()
        self.save_settings()
        self.statusBar().showMessage("Application reset to default settings (256 × 256).", 6000)

    def collect_settings(self) -> Dict[str, Any]:
        return {
            "settings_schema_version": SETTINGS_SCHEMA_VERSION,
            "settings_file": str(self.current_settings_file),
            "language": get_language(),
            "load_generate": self.load_tab.settings(),
            "degraded_input": self.degraded_tab.settings(),
            "algorithm": self.alg_tab.settings(),
            "test": self.test_tab.settings(),
        }

    def _create_settings_menu(self) -> None:
        menu = self.menuBar().addMenu(translate("Settings"))
        self.settings_menu = menu
        act_open = QAction("Open settings profile...", self)
        act_new = QAction("New settings profile...", self)
        act_save_as = QAction("Save settings profile as...", self)
        act_save = QAction("Save settings", self)
        act_open.triggered.connect(self.choose_settings_file)
        act_new.triggered.connect(self.create_new_settings_file)
        act_save_as.triggered.connect(self.save_settings_as)
        act_save.triggered.connect(self.save_settings)
        menu.addAction(act_open)
        menu.addAction(act_new)
        menu.addAction(act_save_as)
        menu.addSeparator()
        menu.addAction(act_save)

    def _create_language_menu(self) -> None:
        self.language_menu = self.menuBar().addMenu(translate("Language"))
        self.language_action_group = QActionGroup(self)
        self.language_action_group.setExclusive(True)
        self.language_actions = {}
        for code, source_name in (("en", "English"), ("pl", "Polish")):
            action = QAction(source_name, self)
            action.setCheckable(True)
            action.setData(code)
            action.triggered.connect(lambda checked=False, lang=code: self.set_gui_language(lang, persist=True))
            self.language_action_group.addAction(action)
            self.language_menu.addAction(action)
            self.language_actions[code] = action
        self._update_language_checks()

    def _update_language_checks(self) -> None:
        active = get_language()
        for code, action in getattr(self, "language_actions", {}).items():
            action.setChecked(code == active)

    def set_gui_language(self, language: str, *, persist: bool = False) -> None:
        """Switch all visible application text without changing numerical state."""
        set_language(language)
        retranslate_all()
        self.setWindowTitle(translate("Grayscale Image Deconvolution Test Environment"))
        if hasattr(self, "settings_menu"):
            self.settings_menu.setTitle(translate("Settings"))
        if hasattr(self, "language_menu"):
            self.language_menu.setTitle(translate("Language"))
        self._update_language_checks()
        # Refresh dynamic labels and plot titles that depend on the current state.
        try:
            self.load_tab.refresh_calculation_views()
            self.degraded_tab.refresh()
            self.alg_tab._update_psf_policy_label()
            self.test_tab.update_selected_method_info()
            self.test_tab.refresh_input_views()
            if self.test_tab.history:
                self.test_tab.show_iteration(self.test_tab.iter_slider.value())
        except Exception:
            pass
        self.statusBar().showMessage(
            f"GUI language: {language_display_name(get_language())}", 4000
        )
        if persist and hasattr(self, "default_settings"):
            self.save_settings()

    def _migrate_settings_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate older profiles and mark them with the current schema.

        v81 changed numerical defaults. v86 adds optional Wiener K scan fields;
        absent scan fields deliberately keep their new GUI defaults.
        """
        if not isinstance(data, dict):
            return {}
        migrated = json.loads(json.dumps(data))
        try:
            version = int(migrated.get("settings_schema_version", 0))
        except Exception:
            version = 0
        if version < 81:
            load_data = migrated.setdefault("load_generate", {})
            try:
                if abs(float(load_data.get("psf_support_fraction_percent", 15.0)) - 15.0) < 1e-9:
                    load_data["psf_support_fraction_percent"] = 45.0
            except Exception:
                pass
            alg_data = migrated.setdefault("algorithm", {})
            # The old GUI stored these former defaults explicitly at both the
            # active-profile level and in every per-algorithm profile.
            if bool(alg_data.get("torch_float64", True)):
                alg_data["torch_float64"] = False
            if bool(alg_data.get("blind_psf_rotational_symmetry", True)):
                alg_data["blind_psf_rotational_symmetry"] = False
            by_alg = alg_data.get("algorithm_params_by_algorithm")
            if isinstance(by_alg, dict):
                for name, params in by_alg.items():
                    if not isinstance(params, dict):
                        continue
                    if "torch_float64" in params and bool(params.get("torch_float64")):
                        params["torch_float64"] = False
                    if name in {"Blind Richardson-Lucy", "PyTorch Blind Adam TV-MAP"}:
                        if bool(params.get("blind_psf_rotational_symmetry", True)):
                            params["blind_psf_rotational_symmetry"] = False
        migrated["settings_schema_version"] = SETTINGS_SCHEMA_VERSION
        return migrated

    def _apply_settings_data(self, data: Dict[str, Any]) -> None:
        self.set_gui_language(str(data.get("language", get_language())), persist=False)
        self.load_tab.apply_settings(data.get("load_generate", {}))
        self.degraded_tab.apply_settings(data.get("degraded_input", {}))
        self.alg_tab.apply_settings(data.get("algorithm", {}))
        self.test_tab.apply_settings(data.get("test", {}))

    def load_settings(self, source: Optional[Path] = None) -> None:
        if source is None:
            source = _active_settings_file()
            if not source.exists():
                for legacy in (LEGACY_SETTINGS_FILE, PACKAGE_LEGACY_SETTINGS_FILE):
                    if legacy.exists():
                        source = legacy
                        break
        if not source.exists():
            self.current_settings_file = SETTINGS_FILE
            self.save_settings()
            return
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
            data = self._migrate_settings_data(data)
            self._apply_settings_data(data)
            resolved_source = Path(source).resolve()
            legacy_sources = {LEGACY_SETTINGS_FILE.resolve(), PACKAGE_LEGACY_SETTINGS_FILE.resolve()}
            self.current_settings_file = SETTINGS_FILE.resolve() if resolved_source in legacy_sources else resolved_source
            _remember_active_settings_file(self.current_settings_file)
            if self.current_settings_file != resolved_source:
                self.save_settings()
            self.statusBar().showMessage(f"Settings profile: {self.current_settings_file}", 8000)
        except Exception as exc:
            QMessageBox.warning(self, "Settings load error", f"Could not load settings profile:\n{exc}")

    def choose_settings_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open settings profile", str(self.current_settings_file.parent), "JSON settings (*.json);;All files (*)"
        )
        if path:
            self.load_settings(Path(path))

    def create_new_settings_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Create new settings profile", str(APP_DIR / "deconvolution_settings_new.json"), "JSON settings (*.json)"
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".json")
        try:
            new_settings = json.loads(json.dumps(self.default_settings))
            new_settings["language"] = get_language()
            self._apply_settings_data(new_settings)
            self.current_settings_file = target.resolve()
            self.save_settings()
            self.statusBar().showMessage(f"Created default settings profile: {self.current_settings_file}", 8000)
        except Exception as exc:
            QMessageBox.warning(self, "Settings creation error", f"Could not create settings profile:\n{exc}")

    def save_settings_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save settings profile as", str(self.current_settings_file), "JSON settings (*.json)"
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".json")
        self.current_settings_file = target.resolve()
        self.save_settings()

    def save_settings(self) -> None:
        try:
            self.current_settings_file.parent.mkdir(parents=True, exist_ok=True)
            self.current_settings_file.write_text(json.dumps(self.collect_settings(), indent=2), encoding="utf-8")
            _remember_active_settings_file(self.current_settings_file)
            self.statusBar().showMessage(f"Settings saved: {self.current_settings_file}", 5000)
        except Exception as exc:
            QMessageBox.warning(self, "Settings save error", f"Could not save settings profile:\n{exc}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        # Never destroy Qt/PyTorch worker objects while native numerical code is
        # still running.  Doing so can abort the entire Spyder kernel.
        try:
            auto_stopped = self.alg_tab.cancel_auto_workers(8000)
        except Exception:
            auto_stopped = False
        try:
            run_stopped = self.test_tab.cancel_run_and_wait(30000)
        except Exception:
            run_stopped = False
        if not (auto_stopped and run_stopped):
            QMessageBox.warning(
                self,
                "Numerical task still running",
                "The window cannot be closed yet because a worker is finishing the current iteration/batch. "
                "Please try again after the progress indicator stops.",
            )
            event.ignore()
            return
        _safe_torch_worker_cleanup()
        self.save_settings()
        event.accept()



def main() -> None:
    import sys
    # Reusing an existing QApplication is safer in interactive IDE kernels.
    app = QApplication.instance()
    owns_application = app is None
    if app is None:
        app = QApplication(sys.argv)
    win = DeconvolutionMainWindow()
    win.show()
    exit_code = app.exec_()
    if owns_application:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
