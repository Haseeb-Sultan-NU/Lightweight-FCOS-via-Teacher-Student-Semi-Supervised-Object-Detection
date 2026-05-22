# Edge-Optimized FCOS+ SSOD Distillation

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-Object_Detection-blue?style=for-the-badge)

A production-grade **Semi-Supervised Object Detection** system built on a lightweight FCOS+ ResNet-50 student trained via teacher–student distillation. Achieves **49.54% mAP@0.50** on COCO 2017 with real-time inference at **30+ FPS on NVIDIA T4**, trained across **230,000+ images** without a single CUDA OOM crash.

---

## 📊 Performance Benchmarks

### Detection Accuracy — COCO 2017 Validation Set
> Evaluated after ~280,000 parameter updates using strict TorchMetrics benchmarks.

| Metric | Score | Description |
|:---|:---|:---|
| **mAP @ 0.50** | `49.54%` | Standard PASCAL VOC overlap metric |
| **mAP @ 0.75** | `34.43%` | Strict bounding box tightness metric |
| **mAP @ 0.50:0.95** | `32.32%` | Overall COCO benchmark (strictest) |

**Size-based breakdown:** Small `13.05%` · Medium `30.92%` · Large `42.64%`

### Hardware Inference Speeds — 800×800 HD Frames

| Hardware | Latency | FPS |
|:---|:---|:---|
| **NVIDIA T4 (Cloud GPU)** | 20–30 ms | 30+ FPS ✅ Real-Time |
| **Apple Silicon (M-Series)** | 70–100 ms | 10–15 FPS |
| **Standard CPU (Intel/AMD)** | 300–500 ms | 2–5 FPS (Viable Offline) |

> Latency covers the full forward-pass regression, bypassing the overhead of two-stage models like Faster R-CNN.

---

## 🧠 Training Data Scale

| Stream | Size | Purpose |
|:---|:---|:---|
| **Labeled Anchor Stream** | 117,266 images | Human-annotated COCO ground truth — prevents hallucination |
| **Unlabeled Oracle Stream** | 112,987 images | Teacher-generated pseudo-labels — transfers spatial reasoning |
| **Blind Validation Set** | 5,000 images | Held-out exclusively for mAP evaluation |
| **Total Pipeline Scale** | **230,000+ environments/epoch** | Orchestrated without hardware failure |

---

## 🚀 Key Engineering Features

**Dual-Stream Distillation Engine**
A custom `DataLoader` that simultaneously processes human-verified ground-truth and synthetic Oracle pseudo-labels in the same batch, preventing the student model from hallucinating on unlabeled data alone.

**Hardware-Resilient Training Loop**
Engineered for stability on constrained cloud hardware (T4 GPUs):
- *Active NaN-Blocker* — detects `inf`/`NaN` loss spikes and safely purges the computational graph before weights are corrupted
- *Unified Backward Pass* — optimizes gradient accumulation and dynamically flushes dead VRAM caches to prevent fragmentation and OOM crashes

**Universal Hardware Benchmarking**
Built-in asynchronous CUDA timing and CPU performance counters measure exact FPS and millisecond latency across NVIDIA GPUs, Apple Silicon, and standard CPUs.

**Automated Cloud Recovery**
Epoch and batch-level checkpointing ensures the full training pipeline resumes seamlessly after cloud disconnections or preemptions.

---

## 🏗️ System Architecture

```
COCO 2017 Dataset
      │
      ├── Labeled Stream (117K)──────────────┐
      │                                      │
      └── Unlabeled Stream (113K)            │
                │                            │
                ▼                            │
        Teacher Model (DETR)                 │
        pseudo-label generation              │
                │                            │
                ▼                            ▼
         Dual-Stream DataLoader
         (ground truth + pseudo-labels in same batch)
                │
                ▼
        Student Model (FCOS+ ResNet-50)
        + Active NaN-Blocker
        + Unified Backward Pass
        + VRAM Cache Management
                │
                ▼
        TorchMetrics mAP Evaluation
        (Blind val set — 5,000 images)
                │
                ▼
        OUTPUT: Edge-Optimized Detector
        49.54% mAP@0.50 · 30+ FPS on T4
```

---

## 📂 Repository Structure

```
Dsl-object-detection/
├── configs/
├── notebooks/
│   └── train_pipeline.ipynb
└── src/
    ├── data/
    │   ├── augmentations.py
    │   ├── dataset.py
    │   └── dual_dataset.py
    ├── losses/
    ├── models/
    │   ├── student_fcos.py
    │   └── teacher_detr.py
    ├── utils/
    │   └── checkpoint.py
    ├── evaluate_student.py
    ├── generate_pseudo_labels.py
    ├── test_loss_function.py
    ├── train_oracle.py
    └── train_student.py
```

| File | Description |
|------|-------------|
| `dual_dataset.py` | Custom SSOD Dual-Stream DataLoader |
| `student_fcos.py` | Lightweight ResNet-50 FCOS+ architecture |
| `teacher_detr.py` | Oracle/Teacher DETR architecture |
| `checkpoint.py` | Cloud-saving and crash recovery logic |
| `train_student.py` | Hardware-resilient student training loop |
| `train_oracle.py` | Teacher model training loop |
| `generate_pseudo_labels.py` | Oracle synthetic data generation |
| `evaluate_student.py` | TorchMetrics mAP evaluation engine |
| `test_loss_function.py` | Unit tests for mathematical regressions |

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/Dsl-object-detection.git
cd Dsl-object-detection
pip install -r requirements.txt
```

Configure your dataset path and hyperparameters in `configs/`.

---

## 🏃 Running

**Train the Teacher (Oracle):**
```bash
python src/train_oracle.py
```

**Generate Pseudo-Labels:**
```bash
python src/generate_pseudo_labels.py
```

**Train the Student:**
```bash
python src/train_student.py
```

**Evaluate:**
```bash
python src/evaluate_student.py
```
