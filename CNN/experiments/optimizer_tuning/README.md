# Optimizer Tuning Experiment

Cross-validation experiment comparing the performance, convergence speed, and numerical stability of first-order and adaptive optimization algorithms for training the CNN surrogate model under the SIMM physics-informed loss.

---

## 1. Scientific Question

**How do different gradient-based and adaptive optimization algorithms perform when optimizing the non-convex physics-regularized aerodynamic objective?**

The experiment evaluates five distinct optimization algorithms implemented in the framework. Every candidate and fold builds a fresh optimizer instance via `OptimizerRecipe`, guaranteeing isolated internal states with no moment leakage.

### Evaluated Search Space (5 Candidates)

| Candidate ID | Optimizer Algorithm | Mathematical Update Formulation | Hyperparameters | Characteristics |
|:---|:---|:---|:---:|:---|
| `sgd` | Vanilla SGD | $\theta_{t+1} = \theta_t - \gamma g_t$ | None | Standard first-order stochastic gradient descent |
| `sgd_momentum` | SGD with Momentum | $v_t = \mu v_{t-1} + g_t, \quad \theta_{t+1} = \theta_t - \gamma v_t$ | $\mu = 0.9$ | Velocity accumulation along consistent gradient directions |
| `adagrad` | AdaGrad | $G_t = G_{t-1} + g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\gamma}{\sqrt{G_t + \epsilon}} g_t$ | $\epsilon = 10^{-8}$ | Cumulative sum of historical squared coordinate gradients |
| `rmsprop` | RMSprop | $v_t = \rho v_{t-1} + (1 - \rho) g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\gamma}{\sqrt{v_t + \epsilon}} g_t$ | $\rho = 0.9, \epsilon = 10^{-8}$ | Exponentially decaying moving average of squared gradients |
| **`adam`** | **Adam** | $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \; v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \; \theta_{t+1} = \theta_t - \frac{\gamma}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$ | $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ | Adaptive first & second moments with bias correction |

Total evaluations: **25 fold runs** (5 candidates $\times$ 5 folds).

---

## 2. Fixed Experimental Configuration

| Hyperparameter | Value | Description |
|:---|:---|:---|
| **Architecture** | `conv5x5-dense-1024-512-256-128` | Winner from topology search |
| **Learning Rate ($\gamma$)** | $10^{-5}$ | Uniform base learning rate across all algorithms |
| **Loss Function** | SIMM Physics Loss | MSE Data Loss + $\lambda_{\text{SIMM}} \cdot \text{Physics Loss}$ |
| **Physics Weight ($\lambda$)** | $0.25$ | Active for $\lvert\alpha\rvert \le 10^\circ$ |
| **Activation Function** | LeakyReLU | Negative slope $\alpha = 0.05$ |
| **Regularization / Dropout** | None | $L_1 = 0, L_2 = 0, p_{\text{drop}} = 0$ |
| **Global Batch Size** | $64$ | Synchronized across MPI ranks |
| **Epoch Budget** | $100$ | Evaluated at 10-epoch checkpoints |
| **Cross-Validation** | 5-Fold `RandomKFold` | Deterministic shuffle with seed `42` |
| **Selection Metric** | Physical-unit Validation MSE | Sample-weighted mean across 5 folds |
| **Training Dataset** | `dataset/cnn_dataset_train.npz` | 1,713 samples (1,370 train / 343 val per fold) |
| **Held-out Test Dataset** | `dataset/cnn_dataset_test.npz` | 429 samples (untouched, evaluated once on winner) |

## 3. Requirements and Prerequisites

- **Toolchain:** C++20 compliant compiler (`g++` $\ge 11$ or `clang++` $\ge 13$) with an MPI implementation (OpenMPI $\ge 4.0$ or MPICH).
- **Build System:** CMake $\ge 3.16$ or direct `mpicxx` compilation.
- **Dataset Files:** `dataset/cnn_dataset_train.npz` (1,713 samples) and `dataset/cnn_dataset_test.npz` (429 samples), generated via `python3 build_dataset.py --seed 42`.
- **Python Environment:** Python 3.8+ with `numpy`, `matplotlib`, and `pandas` for analysis and plotting scripts.
- **Working Directory:** All commands must be run from the **repository root**.

---

## 4. Compilation

Build directly from the repository root:

### Option A: CMake (Recommended)
```bash
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target optimizer_tuning --parallel
```

### Option B: Direct MPI Compilation (`mpicxx`)
```bash
mkdir -p build/experiments
mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/optimizer_tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/optimizer_tuning
```

---

## 5. Running the Experiment

### 1. Quick Verification (Smoke Test)
```bash
mpirun -n 2 ./build/CNN/experiments/optimizer_tuning --smoke
```

### 2. Full 5-Fold Cross-Validation Comparison
```bash
mkdir -p results/cross_validation/optimizer_comparison
mpirun -n 16 ./build/CNN/experiments/optimizer_tuning \
  --mode compare \
  --folds 5 \
  --epochs 100 \
  --batch-size 64 \
  --seed 42 \
  --diagnostics \
  --results-dir results/cross_validation/optimizer_comparison
```

### Output Artifacts
- `results/cross_validation/optimizer_comparison/fold_results.csv`: Per-candidate and per-fold metrics.
- `results/cross_validation/optimizer_comparison/training_history.csv`: History at 10-epoch checkpoints per fold.
- `results/cross_validation/optimizer_comparison/summary.txt`: Human-readable summary of cross-validation ranking.

---

## 6. Results

Cross-validation summary across 5 folds and 100 epochs at base learning rate $\gamma = 10^{-5}$:

| Rank | Optimizer | Mean Validation MSE $\pm$ SD | Fold Range ($\text{MSE}_{\text{val}}$) | Relative Difference vs Adam | Status |
|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | **Adam** | **$0.004427 \pm 0.000887$** | **0.002879 – 0.005044** | **Baseline (0.0%)** | **Selected Winner** |
| 2 | RMSprop | $0.005216 \pm 0.000653$ | 0.004518 – 0.006173 | +17.8% | Stable |
| 3 | AdaGrad | $0.006185 \pm 0.001238$ | 0.004239 – 0.007439 | +39.7% | Premature decay |
| 4 | SGD + Momentum | $0.007151 \pm 0.001412$ | 0.004791 – 0.008473 | +61.5% | Slow convergence |
| 5 | Vanilla SGD | $0.017927 \pm 0.006229$ | 0.013063 – 0.028647 | +304.9% | Severe stalling |

**Adam is selected as the standard optimizer for all subsequent tuning experiments.**

### Final Untouched Test Set Evaluation
Refitting the winning **Adam** optimizer on the full training dataset (1,713 samples) and evaluating once on the held-out test dataset (`dataset/cnn_dataset_test.npz`, 429 samples) yielded:
- **Test Physical MSE:** **`0.003655`**

---

## 7. Analysis and Plotting

### Brief Scientific Analysis
1. **Superiority of Adaptive Moments:** Adam achieved the lowest validation error ($0.004427$), outperforming RMSprop by 15.1% and SGD+Momentum by 38.1%. The combination of momentum and coordinate-wise adaptive scaling handles the ill-conditioned curvature of deep physics-informed networks effectively.
2. **AdaGrad Learning Rate Decay:** AdaGrad's monotonic accumulation of squared gradients ($G_t$) caused the effective step size to shrink prematurely, stalling optimization in later epochs.
3. **Failure of Vanilla SGD at Small Learning Rates:** Non-adaptive SGD struggled severely at $\gamma = 10^{-5}$, producing four-fold higher error ($0.017927$) and extreme cross-fold variance due to ravines and flat plateaus in parameter space.

### Analysis and Plot Generation
Generate comparison summary and plots:
```bash
python3 CNN/experiments/optimizer_tuning/analyze.py \
  --results-dir results/cross_validation/optimizer_comparison
```

Generate report figures:
```bash
python3 CNN/experiments/optimizer_tuning/analysis/plot_report_figures.py
```

Generated figure artifacts in `Report/Images/Chapter04/optimizer/`:
- `optimizer_boxplots.png`: Distribution of validation MSE across folds for all 5 optimizers.
- `optimizer_convergence.png`: Training objective and validation error curves across epochs.
- `optimizer_ranking.png`: Bar chart comparison of final cross-validated validation MSE.
