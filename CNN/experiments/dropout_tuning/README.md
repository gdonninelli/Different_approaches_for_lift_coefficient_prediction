# Dropout-Rate Tuning Experiment

Cross-validation experiment investigating whether stochastic regularization via inverted dropout in the dense regression head improves model generalization or suppresses overfitting for lift coefficient ($C_L$) prediction.

---

## 1. Scientific Question

**Does inverted dropout applied after each hidden dense activation improve cross-validated generalization error and which dropout rate $p \in [0, 0.5]$ optimizes performance?**

Dropout randomly sets neuron activations to zero during training with probability $p$ and rescales surviving activations by $1 / (1 - p)$ (inverted dropout). During inference, dropout acts as the identity mapping. 

To ensure exact reproducibility and fair paired comparison, all candidates (including $p=0.0$) instantiate the dropout layer in their blueprint, ensuring identical parameter initialization and layer-seed sequences across all candidates.

### Evaluated Search Space (5 Candidates)

| Candidate ID | Dropout Rate ($p$) | Role in Experiment | Total Fold Evaluations |
|:---|:---:|:---|:---:|
| `rate_0.0` | $0.0$ | Exact deterministic baseline (paired reference) | 5 |
| `rate_0.1` | $0.1$ | Light stochastic masking | 5 |
| `rate_0.2` | $0.2$ | Moderate stochastic regularization | 5 |
| `rate_0.3` | $0.3$ | Standard regularization rate | 5 |
| `rate_0.5` | $0.5$ | Strong classic dropout rate | 5 |

Total evaluations: **25 fold runs** (5 candidates $\times$ 5 folds).

---

## 2. Fixed Experimental Configuration

| Hyperparameter | Value | Description |
|:---|:---|:---|
| **Architecture** | `conv5x5-dense-1024-512-256-128` | Winner from topology search |
| **Optimizer** | Adam | $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ |
| **Learning Rate ($\eta$)** | $10^{-5}$ | Constant learning rate |
| **Loss Function** | SIMM Physics Loss | MSE Data Loss + $\lambda_{\text{SIMM}} \cdot \text{Physics Loss}$ |
| **Physics Weight ($\lambda$)** | $0.25$ | Active for $\lvert\alpha\rvert \le 10^\circ$ |
| **Activation Function** | LeakyReLU | Negative slope $\alpha = 0.05$ |
| **Weight Regularization** | None | $L_1 = 0, L_2 = 0$ |
| **Global Batch Size** | $64$ | Synchronized across MPI ranks |
| **Epoch Budget** | $100$ | Evaluated at 10-epoch checkpoints |
| **Cross-Validation** | 5-Fold `RandomKFold` | Deterministic shuffle with seed `42` |
| **Selection Metric** | Physical-unit Validation MSE | Sample-weighted mean across 5 folds |
| **Decision Rule** | Paired Difference vs $p=0$ | Improved in $\ge 4/5$ folds and $|\Delta| > \text{std}(\Delta)$ |
| **Dataset** | `dataset/cnn_dataset_train.npz` | 1,713 samples (1,370 train / 343 val per fold) |
| **Held-out Test Dataset** | `dataset/cnn_dataset_test.npz` | 429 samples (untouched, evaluated once on winner)

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
cmake --build build/CNN --target dropout_tuning --parallel
```

### Option B: Direct MPI Compilation (`mpicxx`)
```bash
mkdir -p build/experiments
mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/dropout_tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/dropout_tuning
```

---

## 5. Running the Experiment

### 1. Quick Verification (Smoke Test)
Run a short verification test:
```bash
mpirun -n 2 ./build/CNN/experiments/dropout_tuning help
```

### 2. Full 5-Fold Parameter Sweep (100 Epochs)
```bash
mkdir -p results/cross_validation/dropout_tuning
mpirun -n 4 ./build/CNN/experiments/dropout_tuning sweep 100
```
*Note:* Per-epoch gradient norms, parameter update ratios, and activation diagnostics are written automatically to `results/cross_validation/dropout_tuning/sweep/`. Use `--no-diagnostics` to disable.

### Output Artifacts
- `results/cross_validation/dropout_tuning/sweep_dropout.csv`: Aggregate training and validation MSE per candidate and fold.
- `results/cross_validation/dropout_tuning/sweep/`: Per-candidate and per-fold diagnostic CSVs (`gradient_norms.csv`, `parameter_update_ratios.csv`, `activation_statistics.csv`).

---

## 6. Results

The table below reports cross-validation results across all 5 folds and 100 epochs, alongside peak training diagnostics for numerical stability assessment.

### Metrics and Diagnostic Definitions:

- **Paired $\Delta$ vs $p=0$:** Mean per-fold validation MSE difference relative to the deterministic reference ($\text{MSE}_{p} - \text{MSE}_{0}$). A positive $\Delta$ indicates that dropout degraded performance.
- **Val-Train Gap:** Difference between validation MSE and training MSE ($\text{MSE}_{\text{val}} - \text{MSE}_{\text{train}}$); measures the extent of empirical overfitting.
- **Peak Grad Norm:** Largest `maximum_norm` recorded in `gradient_norms.csv` across all layers and folds (scope: `all`).
- **Peak Weight Norm:** Largest `mean_pre_update_norm` in `parameter_update_ratios.csv` for the `weights` scope.
- **Peak Post-Act Var:** Maximum activation `variance` in `activation_statistics.csv` for the `post_activation` phase.
- **Stability Criterion:** A candidate is deemed numerically stable if all monitored diagnostics remain finite and bounded with no late-epoch runaway divergence.

| Dropout Rate ($p$) | Mean Validation MSE | Fold SD | Paired $\Delta$ vs $p=0$ | Mean Train MSE | Val-Train Gap | Peak Grad Norm | Peak Weight Norm | Peak Post-Act Var | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0** | **0.005834597** | **0.001125249** | **0.0000000** | **0.005421198** | **+0.000413398** | **37.8413** | **15.8782** | **0.186070** | **Selected Winner** |
| 0.1 | 0.006686550 | 0.001184206 | +0.0008520 | 0.006390211 | +0.000296338 | 39.0003 | 15.8840 | 0.157550 | Inferior |
| 0.2 | 0.007630465 | 0.000961879 | +0.0017959 | 0.007235922 | +0.000394543 | 38.4087 | 15.8842 | 0.153652 | Inferior |
| 0.3 | 0.008466757 | 0.001275233 | +0.0026322 | 0.008097506 | +0.000369251 | 44.6056 | 15.8879 | 0.151285 | Inferior |
| 0.5 | 0.011764955 | 0.002912000 | +0.0059304 | 0.011353900 | +0.000411055 | 47.4447 | 15.8899 | 0.185163 | Inferior |

---

## 7. Analysis and Plotting

### Brief Scientific Analysis
1. **Selected Configuration ($p^* = 0.0$):** `dropout rate = 0.0` is selected as it achieves the lowest cross-validated validation MSE ($0.005835$). Training is numerically stable and non-explosive: the initial peak gradient norm of $37.84$ decreases steadily to $2.69$ at epoch 100, while the total weight norm shifts by less than $0.1\%$ from initialization ($15.8705 \to 15.8782$).
2. **Runner-Up Stability ($p = 0.1$):** The second-best candidate ($p=0.1$, MSE $0.006687$) passes all numerical stability checks with well-behaved diagnostics (peak grad $39.00 \to 2.26$ at epoch 100). It is rejected purely because its validation error is $14.6\%$ higher than $p=0.0$.
3. **No Overfitting to Remove:** At learning rate $10^{-5}$ over 100 epochs, the baseline generalization gap ($+0.000413$) is well within the fold standard deviation ($0.001125$). Introducing dropout primarily adds stochastic gradient noise at a fixed epoch budget, hindering optimization without providing regularization benefits.

### Methodological Caveats
- **Initialization Sequence:** The inclusion of dropout layers shifts the per-layer RNG initialization sequence, which explains why absolute fold values differ slightly from the topology search baseline. Only paired differences within this sweep isolate the effect of masking.
- **Dataset Splitting:** Data splitting is performed randomly on individual samples rather than airfoil groups; relative paired comparisons remain strictly valid across candidates sharing the identical fold plan.

### Analysis & Plot Generation
Perform paired statistical analysis:
```bash
python3 CNN/experiments/dropout_tuning/analyze.py \
    results/cross_validation/dropout_tuning/sweep_dropout.csv
```

Generate report figures:
```bash
python3 CNN/experiments/dropout_tuning/analysis/plot_report_figures.py
```

Generated figure artifacts in `Report/Images/Chapter04/dropout/`:
- `dropout_val_mse_vs_rate.png`: Validation MSE vs dropout rate showing monotonic error increase.
- `dropout_paired_delta.png`: Paired differences $\Delta$ against $p=0.0$ across all 5 folds.
- `dropout_diagnostics.png`: Peak gradient norm and post-activation variance evolution across training epochs.
