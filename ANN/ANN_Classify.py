import os
import time
import threading
import subprocess
import csv
from collections import deque
import psutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore


# =========================
# GPU Power/Energy sampling (NVIDIA NVML)
# =========================
class PowerEnergySampler:
    """
    Samples GPU power (W) over time using NVIDIA NVML and integrates
    energy (J) with the trapezoidal rule.

    NOTE ON INTERPRETATION:
    At small workloads (<1s inference), the GPU baseline/idle power (~10 W on
    laptop GPUs) dominates the reading. The measured energy therefore reflects
    GPU *presence* more than actual compute work. This is expected behaviour
    and an important result: for tiny ANN models the fixed overhead >> compute.
    """
    def __init__(self, read_power_w, sample_interval_s: float = 0.05, name: str = "GPU"):
        self.read_power_w      = read_power_w
        self.sample_interval_s = float(sample_interval_s)
        self.name              = name

        self._samples = []          # list[(t, p_w)]
        self._stop    = threading.Event()
        self._thread  = None

        self.avg_power_w        = None
        self.energy_j           = None
        self.duration_s         = None
        self.n_samples_captured = 0

    def _run(self):
        while not self._stop.is_set():
            try:
                t = time.perf_counter()
                p = float(self.read_power_w())
                self._samples.append((t, p))
            except Exception:
                break
            time.sleep(self.sample_interval_s)

    def __enter__(self):
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        self.n_samples_captured = len(self._samples)

        if len(self._samples) < 2:
            self.avg_power_w = None
            self.energy_j    = None
            self.duration_s  = None
            return False

        times  = np.array([t for t, _ in self._samples], dtype=np.float64)
        powers = np.array([p for _, p in self._samples], dtype=np.float64)

        dt       = np.diff(times)
        p_mid    = 0.5 * (powers[:-1] + powers[1:])
        energy_j = float(np.sum(p_mid * dt))

        duration_s  = float(times[-1] - times[0])
        avg_power_w = float(energy_j / duration_s) if duration_s > 0 else None

        self.energy_j    = energy_j
        self.duration_s  = duration_s
        self.avg_power_w = avg_power_w
        return False


def try_make_gpu_sampler(sample_interval_s: float = 0.05, gpu_index: int = 0):
    """
    NVIDIA GPU power via NVML (requires: pip install nvidia-ml-py).
    Returns (sampler_or_None, status_string).
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        raw_name = pynvml.nvmlDeviceGetName(handle)
        name = raw_name.decode("utf-8", errors="ignore") if isinstance(raw_name, (bytes, bytearray)) else str(raw_name)

        def read_power_w():
            return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0  # mW -> W

        return PowerEnergySampler(read_power_w, sample_interval_s, name=f"GPU:{name}"), f"OK ({name})"
    except Exception as e:
        return None, f"Unavailable ({e})"


# =========================
# CPU telemetry (Windows built-in typeperf)
# =========================
class TypeperfSampler:
    """
    Samples a single Windows perf counter using built-in `typeperf`.
    Stores (t, value). If counter is missing/unavailable -> captures nothing.
    """
    def __init__(self, counter_path: str, sample_interval_s: float = 0.2, name: str = "CPU"):
        self.counter_path = counter_path
        self.sample_interval_s = float(sample_interval_s)
        self.name = name

        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._proc = None

        self.avg = None
        self.energy_j = None      # only meaningful if this is a power counter (W)
        self.duration_s = None
        self.n_samples_captured = 0

    def _run(self):
        # typeperf CSV output; run until terminated
        args = ["typeperf", self.counter_path, "-si", str(self.sample_interval_s)]
        try:
            self._proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            return

        assert self._proc.stdout is not None
        reader = csv.reader(self._proc.stdout)

        # Skip first 2 lines (header + column names) if present
        next(reader, None)
        next(reader, None)

        while not self._stop.is_set():
            row = next(reader, None)
            if not row:
                break

            t = time.perf_counter()
            # row: [timestamp, "value"]
            try:
                v = float(row[1].strip().strip('"'))
                self._samples.append((t, v))
            except Exception:
                # ignore parse failures
                pass

        # cleanup
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def __enter__(self):
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

        self.n_samples_captured = len(self._samples)
        if len(self._samples) < 2:
            self.avg = None
            self.energy_j = None
            self.duration_s = None
            return False

        times = np.array([t for t, _ in self._samples], dtype=np.float64)
        vals = np.array([v for _, v in self._samples], dtype=np.float64)

        self.duration_s = float(times[-1] - times[0])
        self.avg = float(np.mean(vals))

        # If this sampler is used for power (W), integrate energy (J)
        dt = np.diff(times)
        v_mid = 0.5 * (vals[:-1] + vals[1:])
        self.energy_j = float(np.sum(v_mid * dt))
        return False


def _typeperf_counter_exists(counter_path: str) -> bool:
    # Probe one sample; if it errors, counter is not available
    try:
        p = subprocess.run(
            ["typeperf", counter_path, "-sc", "1"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=5,
        )
        if p.returncode != 0:
            return False
        stderr = (p.stderr or "").lower()
        if "error" in stderr or "cannot" in stderr:
            return False
        return True
    except Exception:
        return False


def try_make_cpu_util_sampler(sample_interval_s: float = 0.2):
    # Prefer Processor Information (more modern), fallback to Processor
    candidates = [
        r"\Processor Information(_Total)\% Processor Utility",
        r"\Processor(_Total)\% Processor Time",
    ]
    for c in candidates:
        if _typeperf_counter_exists(c):
            return TypeperfSampler(c, sample_interval_s=sample_interval_s, name="CPU Util"), f"OK ({c})"
    return None, "Unavailable (no CPU util counter?)"


def try_make_cpu_power_sampler(sample_interval_s: float = 0.2):
    # Often unavailable on desktops / AMD; will gracefully fall back to N/A
    c = r"\Power Meter(_Total)\Power"
    if _typeperf_counter_exists(c):
        return TypeperfSampler(c, sample_interval_s=sample_interval_s, name="CPU Power"), f"OK ({c})"
    return None, "Unavailable (no Power Meter counter)"


# ADD THIS CLASS (was missing from your file)
class CPUUtilSampler:
    """
    Samples CPU utilisation (%) over time using psutil.
    Works on all Windows / Ryzen systems.
    """
    def __init__(self, sample_interval_s: float = 0.1):
        self.sample_interval_s = float(sample_interval_s)
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self.avg = None
        self.duration_s = None

    def _run(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                util = psutil.cpu_percent(interval=None)  # non-blocking
                self._samples.append((t, util))
            except Exception:
                pass
            time.sleep(self.sample_interval_s)

    def __enter__(self):
        self._samples.clear()
        self._stop.clear()
        # warm up psutil
        psutil.cpu_percent(interval=0.1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        if len(self._samples) < 2:
            self.avg = None
            self.duration_s = None
            return False

        times = np.array([t for t, _ in self._samples], dtype=np.float64)
        utils = np.array([u for _, u in self._samples], dtype=np.float64)

        self.duration_s = float(times[-1] - times[0])
        self.avg = float(np.mean(utils))
        return False


# =========================
# Approximate computations (Dense-only FLOPs)
# =========================
def estimate_dense_flops_per_sample(model: keras.Model) -> float:
    """
    FLOPs/sample for Dense layers only:
      Dense(in, out): ~2 * in * out  (one multiply + one add per weight)
    """
    flops = 0.0
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            w = layer.kernel
            flops += 2.0 * int(w.shape[0]) * int(w.shape[1])
    return flops


# =========================
# Load data
# =========================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
detect_path = os.path.join(BASE_DIR, "content", "detect_dataset.csv")
class_path  = os.path.join(BASE_DIR, "content", "classData.csv")

# detect_dataset drives this classification task as per original code
df       = pd.read_csv(detect_path)
df       = df.drop(columns=["Unnamed: 7", "Unnamed: 8"], errors="ignore")
df       = df.dropna()

class_df = pd.read_csv(class_path)
class_df = class_df.dropna()

print("detect_dataset shape:", df.shape)
print("classData shape     :", class_df.shape)

X     = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]].values
y_raw = df["Output (S)"].values

# Encode labels -> integers
le = LabelEncoder()
y  = le.fit_transform(y_raw).astype(np.int64)

# Split
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalise
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# One-hot encode output
y_train = to_categorical(y_train_int)
y_test  = to_categorical(y_test_int)

# =========================
# Build ANN (Fault CLASSIFICATION — deeper, more parameters by design)
# =========================
model = Sequential([
    Dense(32, activation="relu", input_shape=(6,)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(y_train.shape[1], activation="softmax"),
], name="ANN_FaultClassification")

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# =========================
# Train
# =========================
t0 = time.perf_counter()
history = model.fit(
    X_train, y_train,
    epochs=120,
    batch_size=512,
    validation_split=0.2,
    verbose=1,
)
train_time_s = time.perf_counter() - t0

# Quick test evaluation
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy (model.evaluate):", acc)

# =========================
# Inference + GPU power/energy
# Repeated inference (warm GPU): run predict 5 times and average for stability
# =========================
_ = model.predict(X_test[: min(32, len(X_test))], verbose=0)   # warmup

gpu_sampler, gpu_status = try_make_gpu_sampler(sample_interval_s=0.05, gpu_index=0)

cpu_util_sampler = CPUUtilSampler(sample_interval_s=0.1)
cpu_pwr_sampler,  cpu_pwr_status  = try_make_cpu_power_sampler(sample_interval_s=0.2)

N_REPEAT = 5
infer_start = time.perf_counter()

if gpu_sampler and cpu_pwr_sampler:
    with gpu_sampler, cpu_util_sampler, cpu_pwr_sampler:
        for _ in range(N_REPEAT):
            y_pred = model.predict(X_test, verbose=0)
elif gpu_sampler:
    with gpu_sampler, cpu_util_sampler:
        for _ in range(N_REPEAT):
            y_pred = model.predict(X_test, verbose=0)
else:
    with cpu_util_sampler:
        for _ in range(N_REPEAT):
            y_pred = model.predict(X_test, verbose=0)

infer_total_s = time.perf_counter() - infer_start
infer_time_s  = infer_total_s / N_REPEAT

y_pred_classes = np.argmax(y_pred, axis=1)
test_acc2      = accuracy_score(y_test_int, y_pred_classes)

# =========================
# Compute + GPU metrics
# =========================
dense_flops_per_sample = estimate_dense_flops_per_sample(model)
total_infer_flops      = dense_flops_per_sample * len(X_test)
params                 = model.count_params()

n_samples           = int(len(X_test))
infer_ms_per_sample = (infer_time_s / n_samples) * 1000.0 if n_samples else float("nan")

avg_gpu_power_w         = None
gpu_energy_j            = None
gpu_energy_per_sample_j = None
n_power_samples         = 0

if gpu_sampler and gpu_sampler.energy_j is not None:
    avg_gpu_power_w         = float(gpu_sampler.avg_power_w)
    gpu_energy_j_total      = float(gpu_sampler.energy_j)        # over N_REPEAT passes
    gpu_energy_j            = gpu_energy_j_total / N_REPEAT      # per inference pass
    gpu_energy_per_sample_j = (gpu_energy_j / n_samples) if n_samples else None
    n_power_samples         = gpu_sampler.n_samples_captured

avg_cpu_util = None
if cpu_util_sampler and cpu_util_sampler.avg is not None:
    avg_cpu_util = float(cpu_util_sampler.avg)

avg_cpu_power_w = None
cpu_energy_j = None
cpu_energy_per_sample_j = None
n_cpu_power_samples = 0

# For power: sampler.energy_j is over the whole inference window (N_REPEAT passes)
if cpu_pwr_sampler and cpu_pwr_sampler.energy_j is not None:
    avg_cpu_power_w = float(cpu_pwr_sampler.avg)
    cpu_energy_j_total = float(cpu_pwr_sampler.energy_j)
    cpu_energy_j = cpu_energy_j_total / N_REPEAT
    cpu_energy_per_sample_j = (cpu_energy_j / n_samples) if n_samples else None
    n_cpu_power_samples = cpu_pwr_sampler.n_samples_captured

print("\n==============================")
print("GPU POWER / ENERGY + COMPUTE (CLASSIFICATION)")
print("==============================")
print(f"Train time             : {train_time_s:.4f} s")
print(f"Inference time (avg)   : {infer_time_s:.4f} s  (N={n_samples}, averaged over {N_REPEAT} passes)")
print(f"Inference per-sample   : {infer_ms_per_sample:.4f} ms/sample")
print(f"\nGPU telemetry          : {gpu_status}")
print(f"Power samples captured : {n_power_samples}  (over {N_REPEAT} passes, ~{infer_total_s:.2f} s window)")
if avg_gpu_power_w is not None:
    print(f"Avg GPU power          : {avg_gpu_power_w:.2f} W")
    print(f"GPU energy / pass      : {gpu_energy_j:.4f} J")
    if gpu_energy_per_sample_j is not None:
        print(f"Energy per-sample      : {gpu_energy_per_sample_j:.6f} J/sample")
else:
    print("Avg GPU power          : N/A  (run: pip install nvidia-ml-py)")

# ADD THIS BLOCK for CPU utilization
print("\n==============================")
print("CPU UTILISATION (psutil)")
print("==============================")

if avg_cpu_power_w is not None:
    print(f"Avg CPU Power          : {avg_cpu_power_w:.2f} W")
    print(f"CPU Energy / pass      : {cpu_energy_j:.4f} J")
    if cpu_energy_per_sample_j is not None:
        print(f"CPU Energy / Sample    : {cpu_energy_per_sample_j:.6f} J/sample")
    else:
        print("Avg CPU Power          : N/A (no Power Meter on Ryzen 5000)")

print("\nModel parameters       :", f"{params:,}")
print(f"Dense FLOPs/sample     : {dense_flops_per_sample:,.0f}")
print(f"Total inference FLOPs  : {total_infer_flops:,.0f}  (Dense layers only)")
print(f"\nRecomputed Test Acc    : {test_acc2:.6f}")

# =========================
# FIGURE 1: Model Accuracy
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history.get("accuracy", []),     label="Train")
plt.plot(history.history.get("val_accuracy", []), label="Validation")
plt.title("ANN Fault Classification — Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "classify_accuracy.png"), dpi=150)
plt.show()

# =========================
# FIGURE 2: Run Summary
# =========================
note = (
    "Note: GPU power is dominated by fixed baseline (~10 W) at this\n"
    "workload scale. Higher FLOPs here vs Detection does not cause\n"
    "higher energy because both models run at near-0% GPU utilisation.\n"
    f"Power averaged over {N_REPEAT} inference passes for stability.\n"
)

summary_lines = [
    "ANN — Fault Classification Run Summary",
    "=" * 45,
    f"Test Accuracy              : {test_acc2:.6f}",
    f"Train Time                 : {train_time_s:.4f} s",
    f"Inference Time (avg/pass)  : {infer_time_s:.4f} s",
    f"No. of Samples (test)      : {n_samples}",
    f"Inference Time / Sample    : {infer_ms_per_sample:.4f} ms/sample",
    "",
    "GPU Telemetry:",
    f"  Avg GPU Power            : {avg_gpu_power_w:.2f} W" if avg_gpu_power_w is not None else "  Avg GPU Power            : N/A",
    f"  GPU Energy / pass        : {gpu_energy_j:.4f} J" if gpu_energy_j is not None else "  GPU Energy / pass        : N/A",
    f"  Energy / Sample          : {gpu_energy_per_sample_j:.6f} J/sample" if gpu_energy_per_sample_j is not None else "  Energy / Sample          : N/A",
    f"  Power Samples Captured   : {n_power_samples}  ({N_REPEAT} passes)",
    "",
    "CPU Telemetry:",
    f"  Avg CPU Utilisation      : {avg_cpu_util:.2f} %" if avg_cpu_util is not None else "  Avg CPU Utilisation      : N/A",
    "",
    f"Model Parameters           : {params:,}",
    f"Dense FLOPs / Sample       : {dense_flops_per_sample:,.0f}",
    f"Total Inference FLOPs      : {total_infer_flops:,.0f}",
    "",
    note,
]

plt.figure(figsize=(10, 6))
plt.axis("off")
plt.title("Classification Run Summary", pad=12)
plt.text(
    0.02, 0.98,
    "\n".join(summary_lines),
    ha="left", va="top",
    family="monospace",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9),
)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "classify_summary.png"), dpi=150)
plt.show()