# Learning-Rate and Schedule Tuning Experiment

Cross-validation experiment investigating the impact of base learning rate magnitudes and dynamic scheduling policies on training stability, convergence speed, and generalization error for the CNN surrogate model.

---

## 1. Scientific Question

**How do initial learning rate magnitudes ($\eta \in \{10^{-5}, 10^{-3}, 10^{-1}\}$) and dynamic schedule policies (Constant, Step Decay, Cosine Annealing, Warmup + Cosine) affect convergence and generalization error under the Adam optimizer?**

### Evaluated Scheduling Policies

| Policy Type | Mathematical Formulation | Description & Dynamics |
|:---|:---|:---|
| **Constant** | $\eta_e = \eta_0$ | Fixed learning rate maintained across all 100 epochs |
| **Step Decay (Multi-Step)** | $\eta_e = \eta_0 \cdot \gamma^{\lfloor e / S \rfloor}$ | Geometric step drops by factor $\gamma$ every $S$ epochs ($S=30, \gamma=0.1$ or $S=15, \gamma=0.5$) |
| **Cosine Annealing** | $\eta_e = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{e}{E-1}\pi\right)\right)$ | Smooth monotonic half-cosine decay from $\eta_{\max}$ to $\eta_{\min}$ over $E=100$ epochs |
| **Warmup + Cosine Decay** | 5 epochs linear warmup ($\eta_{\text{start}} \to \eta_{\text{peak}}$), then cosine decay to $\eta_{\min}$ | Linear ramp-up from $10^{-5}$ to $10^{-1}$ to stabilize early moments, followed by cosine annealing |

### Evaluated Search Space (11 Candidates)

| Candidate ID | Schedule Type | Base / Peak $\eta$ | Min $\eta$ | Schedule Parameters |
|:---|:---|:---:|:---:|:---|
| `const_1e-5` | Constant | $10^{-5}$ | — | Initial baseline rate |
| `const_1e-3` | Constant | $10^{-3}$ | — | Intermediate baseline |
| `const_1e-1` | Constant | $10^{-1}$ | — | High aggressive rate |
| `step_s30_1e-3` | Step Decay | $10^{-3}$ | — | Step $S=30$, $\gamma=0.1$ |
| `step_s15_1e-3` | Step Decay | $10^{-3}$ | — | Step $S=15$, $\gamma=0.5$ |
| `step_s30_1e-5` | Step Decay | $10^{-5}$ | — | Step $S=30$, $\gamma=0.1$ |
| `step_s15_1e-5` | Step Decay | $10^{-5}$ | — | Step $S=15$, $\gamma=0.5$ |
| `step_s30_1e-1` | Step Decay | $10^{-1}$ | — | Step $S=30$, $\gamma=0.1$ |
| `step_s15_1e-1` | Step Decay | $10^{-1}$ | — | Step $S=15$, $\gamma=0.5$ |
| `cosine_1e-1` | Cosine Annealing | $10^{-1}$ | $10^{-5}$ | $E=100$ epoch horizon |
| `warmup_cosine_1e-1` | Warmup + Cosine | $10^{-1}$ | $10^{-5}$ | 5 warmup epochs ($10^{-5} \to 10^{-1}$) |

Total evaluations: **55 fold runs** (11 candidates $\times$ 5 folds).

---

## 2. Fixed Experimental Configuration

| Hyperparameter | Value | Description |
|:---|:---|:---|
| **Architecture** | `conv5x5-dense-1024-512-256-128` | Winner from topology search |
| **Optimizer** | Adam | Winner from optimizer search |
| **Loss Function** | SIMM Physics Loss | MSE Data Loss + $\lambda_{\text{SIMM}} \cdot \text{Physics Loss}$ |
| **Physics Weight ($\lambda$)** | $0.10$ | Active for $\lvert\alpha\rvert \le 10^\circ$ |
| **Activation Function** | LeakyReLU | Winner from activation function search |
| **Regularization / Dropout** | None | Winner from regularization search |
| **Global Batch Size** | $64$ | Synchronized across MPI ranks |
| **Epoch Budget** | $100$ | Evaluated at 10-epoch checkpoints |
| **Cross-Validation** | 5-Fold `RandomKFold` | Deterministic shuffle with seed `42` |
| **Selection Metric** | Physical-unit Validation MSE | Sample-weighted mean across 5 folds |
| **Training Dataset** | `dataset/cnn_dataset_train.npz` | 1,713 samples (1,370 train / 343 val per fold) |
| **Held-out Test Dataset** | `dataset/cnn_dataset_test.npz` | 429 samples (untouched, evaluated once on winner) |

## 3. Requirements & Prerequisites

- **Toolchain:** C++20 compliant compiler (`g++` $\ge 11$ or `clang++` $\ge 13$) with an MPI implementation (OpenMPI $\ge 4.0$ or MPICH).
- **Build System:** CMake $\ge 3.16$ or direct `mpicxx` compilation.
- **Dataset Files:** `dataset/cnn_dataset_train.npz` (1,713 samples) and `dataset/cnn_dataset_test.npz` (429 samples), generated via `python3 build_dataset.py --seed 42`.
- **Python Environment:** Python 3.8+ with `numpy` and `matplotlib` for analysis and plotting scripts.
- **Working Directory:** All commands must be run from the **repository root**.

---

## 4. Compilation

Build directly from the repository root:

### Option A: CMake (Recommended)
```bash
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target learning_rate_tuning --parallel
```

### Option B: Direct MPI Compilation (`mpicxx`)
```bash
mkdir -p build/experiments
mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/learning_rate_tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/learning_rate_tuning
```

---

## 5. Running the Experiment

### 1. Quick Verification (Smoke Test)
```bash
mpirun -n 2 ./build/CNN/experiments/learning_rate_tuning --smoke
```

### 2. Full 5-Fold Grid Search (100 Epochs)
```bash
mkdir -p results/cross_validation/learning_rate_tuning
mpirun -n 16 ./build/CNN/experiments/learning_rate_tuning \
  --mode tune \
  --folds 5 \
  --epochs 100 \
  --batch-size 64 \
  --seed 42 \
  --physics-weight 0.10 \
  --diagnostics \
  --results-dir results/cross_validation/learning_rate_tuning
```

### Output Artifacts
- `results/cross_validation/learning_rate_tuning/fold_results.csv`: Per-candidate and per-fold metrics.
- `results/cross_validation/learning_rate_tuning/training_history.csv`: History at 10-epoch validation checkpoints.
- `results/cross_validation/learning_rate_tuning/cv_summary.csv`: Summary statistics across candidates.
- `results/cross_validation/learning_rate_tuning/summary.txt`: Human-readable summary ranking.

---

## 6. Results

Cross-validation summary across 5 folds and 100 epochs:

| Rank | Candidate ID | Schedule Type | Base / Peak $\eta$ | Mean Validation MSE $\pm$ SD | Fold Range ($\text{MSE}_{\text{val}}$) | Status |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **1** | **`const_1e-3`** | **Constant** | **$10^{-3}$** | **$0.003609 \pm 0.001335$** | **0.002092 – 0.006094** | **Selected Winner** |
| 2 | `step_s15_1e-3` | Step Decay ($S=15, \gamma=0.5$) | $10^{-3}$ | $0.004005 \pm 0.000704$ | 0.002826 – 0.004779 | Stable |
| 3 | `step_s30_1e-3` | Step Decay ($S=30, \gamma=0.1$) | $10^{-3}$ | $0.004072 \pm 0.000785$ | 0.002865 – 0.004955 | Stable |
| 4 | `const_1e-5` | Constant (Baseline) | $10^{-5}$ | $0.004419 \pm 0.000790$ | 0.002880 – 0.005001 | Stable |
| 5 | `step_s30_1e-5` | Step Decay ($S=30, \gamma=0.1$) | $10^{-5}$ | $0.005034 \pm 0.000901$ | 0.003434 – 0.005786 | Stable |
| 6 | `step_s15_1e-5` | Step Decay ($S=15, \gamma=0.5$) | $10^{-5}$ | $0.005094 \pm 0.000905$ | 0.003484 – 0.005861 | Stable |
| 7 | `cosine_1e-1` | Cosine Annealing | $10^{-1}$ | $5.58\times 10^{8} \pm 2.25\times 10^{8}$ | $2.09\times 10^8$ – $7.95\times 10^8$ | Diverged |
| 8 | `warmup_cosine_1e-1` | Warmup + Cosine | $10^{-1}$ | $9.05\times 10^{8} \pm 8.87\times 10^{8}$ | $1.21\times 10^8$ – $2.51\times 10^9$ | Diverged |
| 9 | `step_s15_1e-1` | Step Decay ($S=15, \gamma=0.5$) | $10^{-1}$ | $5.01\times 10^{9} \pm 2.96\times 10^{9}$ | $1.53\times 10^9$ – $9.32\times 10^9$ | Diverged |
| 10 | `step_s30_1e-1` | Step Decay ($S=30, \gamma=0.1$) | $10^{-1}$ | $9.45\times 10^{9} \pm 4.73\times 10^{9}$ | $3.20\times 10^9$ – $1.63\times 10^{10}$ | Diverged |
| 11 | `const_1e-1` | Constant | $10^{-1}$ | $2.13\times 10^{17} \pm 3.12\times 10^{17}$ | $1.84\times 10^{16}$ – $8.31\times 10^{17}$ | Diverged |

### Final Untouched Test Set Evaluation
Refitting the winning candidate (**`const_1e-3`**) on the full training dataset (1,713 samples) and evaluating once on the held-out test dataset (`dataset/cnn_dataset_test.npz`, 429 samples) yielded:
- **Final Untouched Test Physical MSE:** **`0.00325384`**

---

## 7. Analysis and Plotting

### Brief Scientific Analysis
1. **Optimal Learning Rate Magnitude ($\eta = 10^{-3}$):** Constant Adam at $\eta = 10^{-3}$ achieved the lowest validation MSE ($0.003609$), representing an **18.3% improvement** over the initial baseline $\eta = 10^{-5}$ ($0.004419$). The larger step size facilitates rapid descent across the non-convex loss surface.
2. **Premature Stalling from Rate Decay:** Within a 100-epoch budget, decaying the rate via step decay (`step_s15_1e-3`, `step_s30_1e-3`) froze weights in suboptimal local basins, increasing error ($0.004005$ and $0.004072$).
3. **Catastrophic Divergence at $\eta \ge 10^{-1}$:** All schedules reaching $\eta = 10^{-1}$ diverged instantly with validation MSE exceeding $10^8 - 10^{17}$. Even 5-epoch linear warmup failed to prevent gradient explosion at $\eta = 10^{-1}$.

### Plot Generation
Generate report figures:
```bash
python3 CNN/experiments/learning_rate_tuning/analysis/plot_report_learning_rate.py
```

Render detailed training diagnostic profiles:
```bash
python3 CNN/analysis/plot_training_diagnostics.py \
  --input results/cross_validation/learning_rate_tuning/learning_rate_tuning/run \
  --output-dir results/cross_validation/learning_rate_tuning/plots \
  --fold 0 \
  --epoch 1 --epoch 10 --epoch 25 --epoch 50 --epoch 100
```

Generated figure artifacts in `Report/Images/Chapter04/learning_rate/`:
- `lr_ranking.png`: Validation MSE ranking of all stable learning rate configurations.
- `lr_convergence.png`: Training and validation MSE learning curves across 100 epochs.
- `lr_schedules.png`: Mathematical profiles of tested constant and dynamic learning rate schedules.
