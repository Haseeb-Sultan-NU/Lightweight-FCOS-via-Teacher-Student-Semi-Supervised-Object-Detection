# Edge-Optimized FCOS+ SSOD Distillation

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-Object_Detection-blue?style=for-the-badge)

A production-grade **Cross-Architecture Knowledge Distillation** framework for Semi-Supervised Object Detection. A heavyweight Deformable DETR Oracle transfers spatial reasoning to a lightweight FCOS+ ResNet-50 student via a custom dual-stream pipeline — achieving **49.54% mAP@0.50** on COCO 2017, real-time at **30+ FPS on NVIDIA T4**, trained across **230,000+ image passes per epoch** without a single CUDA OOM crash across **400+ hours of continuous GPU compute**.

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
| **Standard Hardware (Baseline)** | — | **9.71 FPS** (edge deployment target) |

> Latency covers the full forward-pass regression, bypassing the overhead of traditional two-stage models like Faster R-CNN.

---

## 🧠 Training Data Scale

| Stream | Size | Purpose |
|:---|:---|:---|
| **Labeled Anchor Stream** | 117,266 unique images | Human-annotated COCO 2017 ground truth — prevents hallucination |
| **Unlabeled Oracle Stream** | 112,987 images | Teacher-generated pseudo-labels — transfers spatial reasoning |
| **Dual-Stream Oversampling** | ~236,574 passes/epoch | 1:1 mixing ratio (μ = 0.5) forces equal exposure across 56,494 batches |
| **Blind Validation Set** | 5,000 images | Held-out exclusively for TorchMetrics mAP evaluation |
| **Total GPU Compute** | **400+ hours** | Continuous NVIDIA T4 — zero OOM crashes |

---

## 🔬 Experimental Scope

9 primary experimental pipelines executed across 400+ hours of cloud GPU compute:

### Data Scarcity Validation (3 runs)
Benchmarked the framework's label efficiency under extreme annotation budgets:

| Split | Labeled Data | Purpose |
|:---|:---|:---|
| 1% split | ~1,173 images | Minimum viable label scenario |
| 5% split | ~5,863 images | Low-resource deployment simulation |
| 10% split | ~11,727 images | Practical semi-supervised baseline |

### Architectural Ablation Studies (6 runs)
Executed on 30–50% fractional proxy datasets to optimize compute budgets. Parameters tested:

| Parameter | Values Tested |
|:---|:---|
| **NMS Threshold** | Multiple configurations |
| **Input Resolution** | 800px vs 320px Pareto frontier |
| **Pseudo-label Mixing Ratio** | 1:0 · 1:2 · 1:4 |
| **SAGc Confidence Gate** | Dynamic threshold tuning |

---

## 🚀 Key Engineering Features

**Cross-Architecture Knowledge Distillation**
Teacher: high-capacity Deformable DETR Oracle. Student: lightweight anchor-free FCOS+ with ResNet-50 backbone. The student absorbs the teacher's spatial reasoning via pseudo-labels while remaining grounded by human-annotated ground truth in every batch.

**Dual-Stream Distillation Engine**
A custom PyTorch `DataLoader` with a 1:1 mixing ratio (μ = 0.5) that oversamples human-annotated data to match pseudo-labeled data — forcing ~236,574 image passes per epoch across 56,494 discrete batches. Prevents the student from hallucinating on pseudo-labels alone.

**Soft Adaptive Confidence Gate (SAGc)**
A custom-designed confidence filtering mechanism that dynamically suppresses noisy pseudo-labels during training, actively preventing confirmation bias and model drift — with **zero computational overhead at inference time**.

**Hardware-Resilient Training Loop**
Engineered for uninterrupted stability on constrained cloud hardware (T4 GPUs):
- *Active NaN-Blocker* — detects `inf`/`NaN` loss spikes and safely purges the computational graph before weights are corrupted
- *Unified Backward Pass* — optimizes gradient accumulation and dynamically flushes dead VRAM caches to prevent fragmentation and OOM crashes

**Universal Hardware Benchmarking**
Built-in asynchronous CUDA timing and CPU performance counters measure exact FPS and millisecond latency across NVIDIA GPUs, Apple Silicon, and standard CPUs.

**Automated Cloud Recovery**
Epoch and batch-level checkpointing ensures seamless pipeline resumption after cloud disconnections or preemptions.

---

## 🏗️ System Architecture

```
COCO 2017 Dataset (118,287 unique training images)
      │
      ├── Labeled Stream (117K) ──────────────────┐
      │                                           │
      └── Unlabeled Stream (113K)                 │
                │                                 │
                ▼                                 │
        Teacher Model (Deformable DETR)           │
        pseudo-label generation                   │
                │                                 │
                ▼                                 ▼
         Dual-Stream DataLoader
         (μ = 0.5 mixing ratio · 236,574 passes/epoch)
                │
                ▼
        Soft Adaptive Confidence Gate (SAGc)
        (dynamic pseudo-label filtering)
                │
                ▼
        Student Model (FCOS+ · ResNet-50 · Anchor-free)
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
        49.54% mAP@0.50 · 9.71 FPS (standard HW) · 30+ FPS (T4)
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
| `dual_dataset.py` | Custom SSOD Dual-Stream DataLoader (μ = 0.5 mixing ratio) |
| `student_fcos.py` | Lightweight anchor-free FCOS+ with ResNet-50 backbone |
| `teacher_detr.py` | Deformable DETR Oracle/Teacher architecture |
| `checkpoint.py` | Cloud-saving and crash recovery logic |
| `train_student.py` | Hardware-resilient student training loop with SAGc + NaN-Blocker |
| `train_oracle.py` | Teacher model training loop |
| `generate_pseudo_labels.py` | Oracle synthetic data generation |
| `evaluate_student.py` | TorchMetrics mAP evaluation engine |
| `test_loss_function.py` | Unit tests for mathematical regressions |

---

## ⚙️ Setup

```bash
git clone https://github.com/Haseeb-Sultan-NU/Lightweight-FCOS-via-Teacher-Student-Semi-Supervised-Object-Detection.git
cd Lightweight-FCOS-via-Teacher-Student-Semi-Supervised-Object-Detection
pip install -r requirements.txt
```

Configure dataset path and hyperparameters in `configs/`.

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

---

## 📄 License

MIT
