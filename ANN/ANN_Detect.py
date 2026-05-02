import os
import time
import threading
import psutil  # <-- ADD THIS IMPORT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore


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
        self.read_power_w   = read_power_w
        self.sample_interval_s = float(sample_interval_s)
        self.name           = name

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

        dt      = np.diff(times)
        p_mid   = 0.5 * (powers[:-1] + powers[1:])
        energy_j = float(np.sum(p_mid * dt))

        duration_s = float(times[-1] - times[0])
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
# Load dataset
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "content", "detect_dataset.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found: {csv_path}")

df = pd.read_csv(csv_path)
df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"], errors="ignore")
df = df.dropna()

print("Dataset shape:", df.shape)
print(df.head())

X = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]].values
y = df["Output (S)"].values  # numeric class IDs (0/1)

# =========================
# Train/test split + normalisation
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# =========================
# Build ANN  (Fault DETECTION — binary-like, fewer parameters by design)
# =========================
num_classes = len(np.unique(y))

model = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(6,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
], name="ANN_FaultDetection")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
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
# Inference (fault detection) + GPU power/energy
# Repeated inference (warm GPU): run predict 5 times and average for stability
# =========================
_ = model.predict(X_test[: min(32, len(X_test))], verbose=0)   # warmup

gpu_sampler, gpu_status = try_make_gpu_sampler(sample_interval_s=0.05, gpu_index=0)

cpu_util_sampler = CPUUtilSampler(sample_interval_s=0.1)  # <-- ADD THIS

N_REPEAT = 5    # repeat inference to get a longer, more stable power window
infer_start = time.perf_counter()
if gpu_sampler:
    with gpu_sampler, cpu_util_sampler:  # <-- ADD cpu_util_sampler HERE
        for _ in range(N_REPEAT):
            y_pred_raw = model.predict(X_test, verbose=0)
else:
    with cpu_util_sampler:  # <-- ADD THIS ELSE BRANCH
        for _ in range(N_REPEAT):
            y_pred_raw = model.predict(X_test, verbose=0)
infer_total_s = time.perf_counter() - infer_start
infer_time_s  = infer_total_s / N_REPEAT   # per-pass average

y_pred_classes = np.argmax(y_pred_raw, axis=1)
test_acc = accuracy_score(y_test, y_pred_classes)
cm       = confusion_matrix(y_test, y_pred_classes)

print("\nAccuracy:", test_acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred_classes))

# =========================
# Compute + GPU metrics
# =========================
dense_flops_per_sample = estimate_dense_flops_per_sample(model)
total_infer_flops      = dense_flops_per_sample * len(X_test)
params                 = model.count_params()

n_samples           = int(len(X_test))
infer_ms_per_sample = (infer_time_s / n_samples) * 1000.0 if n_samples else float("nan")

avg_gpu_power_w        = None
gpu_energy_j           = None
gpu_energy_per_sample_j = None
n_power_samples        = 0

if gpu_sampler and gpu_sampler.energy_j is not None:
    avg_gpu_power_w         = float(gpu_sampler.avg_power_w)
    gpu_energy_j_total      = float(gpu_sampler.energy_j)        # over N_REPEAT passes
    gpu_energy_j            = gpu_energy_j_total / N_REPEAT      # per inference pass
    gpu_energy_per_sample_j = (gpu_energy_j / n_samples) if n_samples else None
    n_power_samples         = gpu_sampler.n_samples_captured

# Extract CPU utilization metric
avg_cpu_util = None
if cpu_util_sampler and cpu_util_sampler.avg is not None:
    avg_cpu_util = float(cpu_util_sampler.avg)

print("\n==============================")
print("GPU POWER / ENERGY + COMPUTE (FAULT DETECTION)")
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

print("\n==============================")
print("CPU UTILISATION (psutil)")
print("==============================")
if avg_cpu_util is not None:
    print(f"Avg CPU Utilisation    : {avg_cpu_util:.2f} %")
else:
    print("Avg CPU Utilisation    : N/A")

print("\nModel parameters       :", f"{params:,}")
print(f"Dense FLOPs/sample     : {dense_flops_per_sample:,.0f}")
print(f"Total inference FLOPs  : {total_infer_flops:,.0f}  (Dense layers only)")

# =========================
# FIGURE 1: Model Accuracy
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history.get("accuracy", []),     label="Train")
plt.plot(history.history.get("val_accuracy", []), label="Validation")
plt.title("ANN Fault Detection — Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "detect_accuracy.png"), dpi=150)
plt.show()

# =========================
# FIGURE 2: Run Summary
# =========================
note = (
    "Note: GPU power is dominated by fixed baseline (~10 W) at this\n"
    "workload scale. Differences between models are in sampling noise,\n"
    f"not compute. Power averaged over {N_REPEAT} inference passes for stability."
)

summary_lines = [
    "ANN — Fault Detection Run Summary",
    "=" * 45,
    f"Test Accuracy              : {test_acc:.6f}",
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
plt.title("Fault Detection Run Summary", pad=12)
plt.text(
    0.02, 0.98,
    "\n".join(summary_lines),
    ha="left", va="top",
    family="monospace",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9),
)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "detect_summary.png"), dpi=150)
plt.show()