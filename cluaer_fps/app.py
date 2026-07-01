from __future__ import annotations

import copy
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .analyzer import FrameAnalyzer
from .capture import MonitorInfo, ScreenCapture
from .config import AppConfig
from .model import DeepLearningDetector
from .predictor import PredictionEngine
from .situation_model import SituationModel
from .storage import PredictionStorage
from .types import ManualContext, PredictionReport


class CluaerFPSApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("cluaerAI FPS Predictor")
        self.root.geometry("1280x760")
        self.root.minsize(1080, 680)

        self.config = AppConfig.load()
        self.config.save()
        self.situation_model = SituationModel.load(self.config.situation_model_path)
        self.capture = ScreenCapture()
        self.detector = DeepLearningDetector(
            model_path=self.config.model_path,
            labels_path=self.config.labels_path,
            backend=self.config.model_backend,
            confidence=self.config.detection_confidence,
        )
        self.analyzer = FrameAnalyzer(self.detector)
        self.predictor = PredictionEngine(self.situation_model)
        self.storage = PredictionStorage(self.config.data_path)

        self.running = False
        self.worker: threading.Thread | None = None
        self.output_queue: queue.Queue[tuple[object, PredictionReport]] = queue.Queue(maxsize=2)
        self.error_queue: queue.Queue[str] = queue.Queue(maxsize=2)
        self.context_lock = threading.Lock()
        self.manual_context = ManualContext()
        self.last_report: PredictionReport | None = None
        self.last_image: ImageTk.PhotoImage | None = None
        self.last_auto_log = 0.0
        self.capture_monitor_index = self.config.monitor_index
        self.capture_frame_interval = 1.0 / max(1, self.config.capture_fps)

        self.monitors = self._safe_monitors()
        self.vars: dict[str, tk.StringVar] = {}

        self._build_ui()
        self._refresh_manual_context()
        self._poll_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _safe_monitors(self) -> list[MonitorInfo]:
        try:
            return self.capture.list_monitors()
        except Exception:
            return []

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=(10, 6))
        style.configure("TLabel", padding=(2, 2))

        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill=tk.BOTH, expand=True)
        root_frame.columnconfigure(0, weight=3)
        root_frame.columnconfigure(1, weight=2)
        root_frame.rowconfigure(1, weight=1)

        controls = ttk.Frame(root_frame)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        for index in range(10):
            controls.columnconfigure(index, weight=0)
        controls.columnconfigure(9, weight=1)

        self.monitor_var = tk.StringVar(value=self._default_monitor_label())
        monitor_combo = ttk.Combobox(
            controls,
            textvariable=self.monitor_var,
            values=[monitor.label for monitor in self.monitors],
            state="readonly",
            width=24,
        )
        monitor_combo.grid(row=0, column=0, sticky="w", padx=(0, 6))

        self.game_var = tk.StringVar(value="auto")
        game_combo = ttk.Combobox(
            controls,
            textvariable=self.game_var,
            values=["auto", "valorant", "pubg", "overwatch", "apex", "generic"],
            state="readonly",
            width=14,
        )
        game_combo.grid(row=0, column=1, sticky="w", padx=(0, 6))

        ttk.Label(controls, text="FPS").grid(row=0, column=2, sticky="e")
        self.fps_var = tk.StringVar(value=str(self.config.capture_fps))
        ttk.Spinbox(controls, from_=1, to=30, textvariable=self.fps_var, width=5).grid(
            row=0, column=3, sticky="w", padx=(2, 6)
        )

        self.start_button = ttk.Button(controls, text="Start Capture", command=self.start_capture)
        self.start_button.grid(row=0, column=4, padx=(0, 6))
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=5, padx=(0, 6))
        ttk.Button(controls, text="Save Prediction", command=self.save_prediction).grid(
            row=0, column=6, padx=(0, 6)
        )

        self.status_var = tk.StringVar(
            value=f"Detector: {self.detector.status} | Predictor: {self.situation_model.status}"
        )
        ttk.Label(controls, textvariable=self.status_var).grid(row=0, column=9, sticky="e")

        preview_frame = ttk.Frame(root_frame)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.preview_label = ttk.Label(preview_frame, anchor=tk.CENTER)
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(root_frame)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        context_frame = ttk.LabelFrame(right, text="Context")
        context_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for col in range(4):
            context_frame.columnconfigure(col, weight=1)

        fields = [
            ("map_name", "Map"),
            ("mode", "Mode"),
            ("score", "Score/Phase"),
            ("remaining_time", "Time"),
            ("health", "Health"),
            ("armor", "Armor"),
            ("weapon", "Weapon"),
            ("ammo", "Ammo"),
            ("utility", "Utility"),
            ("alive_allies", "Allies"),
            ("alive_enemies", "Enemies"),
            ("position", "Position"),
            ("minimap_info", "Minimap"),
            ("kill_log", "Kill Log"),
        ]

        for index, (key, label) in enumerate(fields):
            row = index // 2
            col = (index % 2) * 2
            ttk.Label(context_frame, text=label).grid(row=row, column=col, sticky="w")
            var = tk.StringVar()
            self.vars[key] = var
            entry = ttk.Entry(context_frame, textvariable=var)
            entry.grid(row=row, column=col + 1, sticky="ew", padx=(4, 8), pady=2)

        self.report_text = tk.Text(right, wrap=tk.WORD, height=18, font=("Consolas", 10))
        self.report_text.grid(row=1, column=0, sticky="nsew")
        self.report_text.insert(tk.END, "Start Capture를 누르면 실시간 분석이 시작됩니다.")
        self.report_text.configure(state=tk.DISABLED)

        feedback_frame = ttk.Frame(right)
        feedback_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        feedback_frame.columnconfigure(0, weight=1)
        self.feedback_var = tk.StringVar()
        ttk.Entry(feedback_frame, textvariable=self.feedback_var).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(feedback_frame, text="Save Feedback", command=self.save_feedback).grid(
            row=0, column=1, sticky="e"
        )

    def _default_monitor_label(self) -> str:
        if not self.monitors:
            return "No monitor"
        preferred = next(
            (monitor for monitor in self.monitors if monitor.index == self.config.monitor_index),
            self.monitors[0],
        )
        return preferred.label

    def _selected_monitor_index(self) -> int:
        label = self.monitor_var.get()
        for monitor in self.monitors:
            if monitor.label == label:
                return monitor.index
        return 1

    def _refresh_manual_context(self) -> None:
        context = ManualContext(
            game=self.game_var.get(),
            map_name=self.vars.get("map_name", tk.StringVar()).get(),
            mode=self.vars.get("mode", tk.StringVar()).get(),
            score=self.vars.get("score", tk.StringVar()).get(),
            remaining_time=self.vars.get("remaining_time", tk.StringVar()).get(),
            health=self.vars.get("health", tk.StringVar()).get(),
            armor=self.vars.get("armor", tk.StringVar()).get(),
            weapon=self.vars.get("weapon", tk.StringVar()).get(),
            ammo=self.vars.get("ammo", tk.StringVar()).get(),
            utility=self.vars.get("utility", tk.StringVar()).get(),
            alive_allies=self.vars.get("alive_allies", tk.StringVar()).get(),
            alive_enemies=self.vars.get("alive_enemies", tk.StringVar()).get(),
            position=self.vars.get("position", tk.StringVar()).get(),
            minimap_info=self.vars.get("minimap_info", tk.StringVar()).get(),
            kill_log=self.vars.get("kill_log", tk.StringVar()).get(),
        )
        with self.context_lock:
            self.manual_context = context
        self.root.after(500, self._refresh_manual_context)

    def start_capture(self) -> None:
        if self.running:
            return
        if not self.monitors:
            messagebox.showerror("Capture Error", "No monitor was detected.")
            return

        self.running = True
        self.capture_monitor_index = self._selected_monitor_index()
        capture_fps = self._safe_int(self.fps_var.get(), self.config.capture_fps)
        self.capture_frame_interval = 1.0 / max(1, capture_fps)
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.worker = threading.Thread(target=self._capture_loop, daemon=True)
        self.worker.start()

    def stop_capture(self) -> None:
        self.running = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)

    def _capture_loop(self) -> None:
        last_analysis = 0.0
        analysis_interval = 1.0 / max(1, self.config.analysis_fps)

        try:
            with self.capture.session() as capture_session:
                while self.running:
                    start = time.perf_counter()
                    frame = capture_session.grab(self.capture_monitor_index)
                    now = time.time()
                    if now - last_analysis >= analysis_interval:
                        last_analysis = now
                        with self.context_lock:
                            context = copy.deepcopy(self.manual_context)
                        signals = self.analyzer.analyze(frame)
                        report = self.predictor.predict(context, signals)
                        self._put_output(frame, report)
                        self._maybe_auto_log(report, now)

                    elapsed = time.perf_counter() - start
                    time.sleep(max(0.0, self.capture_frame_interval - elapsed))
        except Exception as exc:
            self._put_error(str(exc))
            self.running = False

    def _put_output(self, frame: object, report: PredictionReport) -> None:
        try:
            self.output_queue.put_nowait((frame, report))
        except queue.Full:
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            self.output_queue.put_nowait((frame, report))

    def _put_error(self, message: str) -> None:
        try:
            self.error_queue.put_nowait(message)
        except queue.Full:
            try:
                self.error_queue.get_nowait()
            except queue.Empty:
                pass
            self.error_queue.put_nowait(message)

    def _maybe_auto_log(self, report: PredictionReport, now: float) -> None:
        interval = self.config.auto_log_interval_sec
        if interval <= 0:
            return
        if now - self.last_auto_log < interval:
            return
        self.last_auto_log = now
        self.storage.append(report.learning_data())

    def _poll_queue(self) -> None:
        self._poll_errors()
        try:
            while True:
                frame, report = self.output_queue.get_nowait()
                self.last_report = report
                self._update_preview(frame)
                self._update_report(report)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _poll_errors(self) -> None:
        try:
            message = self.error_queue.get_nowait()
        except queue.Empty:
            return
        self._show_worker_error(message)

    def _update_preview(self, frame: object) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = self.config.preview_width
        scale = width / max(1, image.width)
        height = int(image.height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        self.last_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.last_image)

    def _update_report(self, report: PredictionReport) -> None:
        text = self.predictor.format_report(report)
        self.report_text.configure(state=tk.NORMAL)
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert(tk.END, text)
        self.report_text.configure(state=tk.DISABLED)

    def save_prediction(self) -> None:
        if not self.last_report:
            messagebox.showinfo("No Prediction", "No prediction has been created yet.")
            return
        self.storage.append(self.last_report.learning_data())
        messagebox.showinfo("Saved", f"Prediction saved to {self.config.data_path}")

    def save_feedback(self) -> None:
        if not self.last_report:
            messagebox.showinfo("No Prediction", "No prediction has been created yet.")
            return
        actual_result = self.feedback_var.get().strip()
        if not actual_result:
            messagebox.showinfo("Missing Result", "Enter the actual result first.")
            return
        self.storage.append_feedback(self.last_report.learning_data(), actual_result)
        self.feedback_var.set("")
        messagebox.showinfo("Saved", f"Feedback saved to {self.config.data_path}")

    def _show_worker_error(self, message: str) -> None:
        self.stop_capture()
        messagebox.showerror("Capture Error", message)

    def _safe_int(self, value: str, default: int) -> int:
        try:
            return int(value)
        except ValueError:
            return default

    def _on_close(self) -> None:
        self.running = False
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = CluaerFPSApp(root)
    root.mainloop()
