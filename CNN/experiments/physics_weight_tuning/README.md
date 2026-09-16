# Physics-Weight (SIMM λ) Tuning Experiment

Cross-validation experiment investigating the influence of the physics-informed residual loss weight ($\lambda_{\text{SIMM}}$) on surrogate model generalization, parameter gradients, and prediction accuracy across angle-of-attack regimes.

---

## 1. Scientific Question

**How strongly should the linear thin-airfoil physical prior ($C_L = 2\pi\alpha$) be enforced in the composite loss function, and does this physics regularizer improve generalization over a purely data-driven model ($\lambda = 0$)?**

The SIMM (Semi-Implicit Multiscale Modeling) physics-informed loss function is defined as:
$$\mathcal{L}(\theta) = \text{MSE}_{\text{data}}(\hat{C}_L, C_L) + \lambda_{\text{SIMM}} \cdot \mathbb{I}_{|\alpha| \le 10^\circ} \cdot \text{MSE}_{\text{physics}}(\hat{C}_L, 2\pi\alpha)$$

where the physical constraint is selectively applied via an indicator mask $\mathbb{I}_{|\alpha| \le 10^\circ}$ to the pre-stall regime ($|\alpha| \le 10^\circ \approx 0.1745\text{ rad}$).

### Evaluated Search Space (7 Candidates)

| Candidate ID | Physics Weight ($\lambda$) | Role in Experiment |
|:---|:---:|:---|
| `lambda_0.0` | $0.00$ | Pure data-driven model (paired reference baseline) |
| `lambda_0.05` | $0.05$ | Weak physical regularization |
| `lambda_0.1` | $0.10$ | Intermediate regularization |
| `lambda_0.25` | $0.25$ | Production default setting |
| `lambda_0.5` | $0.50$ | Strong physical prior |
| `lambda_1.0` | $1.00$ | High physical weight |
| `lambda_2.0` | $2.00$ | Overly strong prior (stress test for loss distortion) |

Total evaluations: **35 fold runs** (7 candidates $\times$ 5 folds).

---

## 2. Fixed Experimental Configuration

| Hyperparameter | Value | Description |
|:---|:---|:---|
| **Architecture** | `conv5x5-dense-1024-512-256-128` | Winner from topology search |
| **Optimizer** | Adam | $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ |
| **Learning Rate ($\eta$)** | $10^{-5}$ | Constant learning rate |
| **Activation Function** | LeakyReLU | Negative slope $\alpha = 0.05$ |
| **Regularization / Dropout** | None | $L_1 = 0, L_2 = 0, p_{\text{drop}} = 0$ |
| **Global Batch Size** | $64$ | Synchronized across MPI ranks |
| **Epoch Budget** | $100$ | Evaluated at 10-epoch checkpoints |
| **Cross-Validation** | 5-Fold `RandomKFold` | Deterministic shuffle with seed `42` |
| **Selection Metric** | Physical-unit Validation MSE | Sample-weighted mean across 5 folds |
| **Decision Rule** | Paired Difference vs $\lambda=0$ | Improved in $\ge 4/5$ folds and $|\Delta| > \text{std}(\Delta)$ |
| **Dataset** | `dataset/cnn_dataset_train.npz` | 1,713 samples (1,370 train / 343 val per fold) |
| **Held-out Test Dataset** | `dataset/cnn_dataset_test.npz` | 429 samples (untouched, evaluated once on winner) |


## 3. Requirements and Prerequisites

- **Toolchain:** C++20 compliant compiler (`g++` $\ge 11$ or `clang++` $\ge 13$) with an MPI implementation (OpenMPI $\ge 4.0$ or MPICH).
- **Build System:** CMake $\ge 3.16$ or direct `mpicxx` compilation.
- **Dataset Files:** `dataset/cnn_dataset_train.npz` (1,713 samples), generated via `python3 build_dataset.py --seed 42`.
- **Python Environment:** Python 3.8+ with `numpy` and `matplotlib` for analysis and plotting scripts.
- **Working Directory:** All commands must be run from the **repository root**.

---

## 4. Compilation

Build directly from the repository root:

### Option A: CMake (Recommended)
```bash
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target physics_weight_tuning --parallel
```

### Option B: Direct MPI Compilation (`mpicxx`)
```bash
mkdir -p build/experiments
mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/physics_weight_tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/physics_weight_tuning
```

---

## 5. Running the Experiment

### 1. Quick Verification (Smoke Test)
```bash
mpirun -n 2 ./build/CNN/experiments/physics_weight_tuning help
```

### 2. Full 5-Fold Parameter Sweep (100 Epochs)
```bash
mkdir -p results/cross_validation/physics_weight_tuning
mpirun -n 4 ./build/CNN/experiments/physics_weight_tuning sweep 100
```
*Note:* Full per-epoch gradient norms, parameter update ratios, and activation diagnostics are recorded under `results/cross_validation/physics_weight_tuning/sweep/`. Use `--no-diagnostics` to disable.

### Output Artifacts
- `results/cross_validation/physics_weight_tuning/sweep_physics.csv`: Aggregate training MSE, validation MSE, and partitioned MSE for small-angle ($\lvert\alpha\rvert \le 10^\circ$) and post-stall ($\lvert\alpha\rvert > 10^\circ$) regimes.
- `results/cross_validation/physics_weight_tuning/sweep/`: Comprehensive diagnostic CSVs per candidate and fold.

---

## 6. Results

Cross-validation performance across 5 folds and 100 epochs, alongside peak training diagnostics for numerical stability assessment.

### Metrics & Diagnostic Definitions:
- **Paired $\Delta$ vs $\lambda=0$:** Mean per-fold validation MSE difference relative to the unregularized ($\lambda=0$) baseline ($\text{MSE}_{\lambda} - \text{MSE}_{0}$). A positive $\Delta$ indicates that physics regularization degraded empirical validation MSE.
- **Val-Train Gap:** Difference between validation MSE and training MSE ($\text{MSE}_{\text{val}} - \text{MSE}_{\text{train}}$); measures the degree of empirical overfitting.
- **Peak Grad Norm:** Largest `maximum_norm` recorded in `gradient_norms.csv` across all layers and folds (scope: `all`).
- **Peak Weight Norm:** Largest `mean_pre_update_norm` in `parameter_update_ratios.csv` for the `weights` scope.
- **Peak Post-Act Var:** Maximum activation `variance` in `activation_statistics.csv` for the `post_activation` phase.
- **Stability Criterion:** A candidate is deemed numerically stable if all monitored diagnostics remain finite and bounded with no late-epoch runaway divergence.

| $\lambda_{\text{SIMM}}$ | Mean Validation MSE | Fold SD | Paired $\Delta$ vs $\lambda=0$ | Mean Train MSE | Val-Train Gap | Peak Grad Norm | Peak Weight Norm | Peak Post-Act Var | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.00** | **0.005601568** | **0.001228485** | **0.0000000** | **0.005183447** | **+0.000418122** | **56.8711** | **15.8807** | **0.196683** | **Selected Winner** |
| 0.05 | 0.005638606 | 0.001270011 | +0.0000370 | 0.005217444 | +0.000421162 | 59.1692 | 15.8804 | 0.198590 | Inferior |
| 0.10 | 0.005630026 | 0.001247489 | +0.0000285 | 0.005214046 | +0.000415979 | 61.4693 | 15.8801 | 0.199118 | Inferior |
| 0.25 | 0.005707657 | 0.001336724 | +0.0001061 | 0.005307874 | +0.000399784 | 68.3797 | 15.8791 | 0.199148 | Inferior |
| 0.50 | 0.006026890 | 0.001334059 | +0.0004253 | 0.005637086 | +0.000389804 | 79.9217 | 15.8782 | 0.201724 | Inferior |
| 1.00 | 0.006582661 | 0.001509834 | +0.0009811 | 0.006226180 | +0.000356481 | 103.0610 | 15.8777 | 0.204872 | Inferior |
| 2.00 | 0.007408295 | 0.001698277 | +0.0018067 | 0.007103552 | +0.000304743 | 149.4420 | 15.8769 | 0.203259 | Severe Distortion |

### Partitioned Error Analysis by Angle Regime ($\lvert\alpha\rvert \le 10^\circ$ vs $\lvert\alpha\rvert > 10^\circ$)

| $\lambda_{\text{SIMM}}$ | $\text{MSE}_{\text{val}}$ ($\lvert\alpha\rvert \le 10^\circ$, 1442 samples) | $\text{MSE}_{\text{val}}$ ($\lvert\alpha\rvert > 10^\circ$, 271 samples) | High/Low Error Ratio |
|:---:|:---:|:---:|:---:|
| **0.00** | **0.004097** | **0.013604** | **3.32** |
| 0.05 | 0.004108 | 0.013777 | 3.35 |
| 0.10 | 0.004106 | 0.013734 | 3.34 |
| 0.25 | 0.004180 | 0.013828 | 3.31 |
| 0.50 | 0.004470 | 0.014308 | 3.20 |
| 1.00 | 0.005021 | 0.014882 | 2.96 |
| 2.00 | 0.005843 | 0.015727 | 2.69 |

---

## 7. Analysis and Plotting

### Brief Scientific Analysis
1. **Data Sufficiency & Redundancy:** The pure data-driven model ($\lambda = 0$) achieved the lowest overall validation MSE ($0.005602$). The dataset already provides sufficient geometric and aerodynamic supervision, making the approximate $2\pi\alpha$ prior redundant.
2. **Gradient Penalty & Distortion:** Increasing $\lambda$ substantially increased peak gradient norms from $56.87$ ($\lambda=0$) up to $149.44$ ($\lambda=2.0$), distorting the loss landscape without improving accuracy.
3. **Partitioned Invariance:** Increasing $\lambda$ degraded accuracy in both the small-angle regime ($|\alpha| \le 10^\circ$) and the high-angle regime ($|\alpha| > 10^\circ$).
4. **Conclusion:** **$\lambda^* = 0.0$ minimizes cross-validated error.** (Note: in ordinary production training, a modest weight of $\lambda=0.10$ is maintained to preserve theoretical physical grounding).

### Analysis & Plot Generation
Perform paired statistical analysis:
```bash
python3 CNN/experiments/physics_weight_tuning/analyze.py \
    results/cross_validation/physics_weight_tuning/sweep_physics.csv
```

Generate report figures:
```bash
python3 CNN/experiments/physics_weight_tuning/analysis/plot_report_figures.py
```

Generated figure artifacts in `Report/Images/Chapter04/physics/`:
- `physics_weight_val_mse.png`: Validation MSE vs $\lambda_{\text{SIMM}}$ across all folds.
- `physics_weight_angle_split.png`: Partitioned MSE comparison for low-angle vs high-angle samples.
- `physics_weight_diagnostics.png`: Gradient norm scaling and stability diagnostics across weights.
