# Weight Regularization (L1/L2) Tuning Experiment

Cross-validation experiment investigating the effect of $L_1$ (Lasso) and $L_2$ (Ridge / Weight Decay) parameter penalties on the large `conv5x5-dense-1024-512-256-128` CNN surrogate model.

---

## 1. Scientific Question

**Does explicitly penalizing weight norms via $L_1$ or $L_2$ regularization prevent overfitting and reduce validation MSE in the large 8.065M parameter CNN architecture?**

The objective function with regularization is defined as:
$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}_{\text{SIMM}}(\theta) + \lambda_1 \sum_{w} |w| + \frac{\lambda_2}{2} \sum_{w} w^2$$

The experiment tests two independent 1D sweeps ($L_1$ alone and $L_2$ alone) against the shared $\lambda = 0$ unpenalized reference model.

### Evaluated Search Space (16 Candidates)

The search space comprises two independent 1D sweeps on the weight penalties:

#### $L_1$ Regularization Grid (8 Candidates)
| Candidate ID | Penalty Type | Value ($\lambda_1$) | Role in Experiment | Total Fold Evaluations |
|:---|:---|:---:|:---|:---:|
| `l1_0.0` | None | $0.0$ | Exact unregularized baseline (paired reference) | 5 |
| `l1_6.75e-7` | $L_1$ (Lasso) | $6.75 \times 10^{-7}$ | Weak sparsity penalty | 5 |
| `l1_2.13e-6` | $L_1$ (Lasso) | $2.13 \times 10^{-6}$ | Moderate-low sparsity penalty | 5 |
| `l1_6.75e-6` | $L_1$ (Lasso) | $6.75 \times 10^{-6}$ | Moderate sparsity penalty | 5 |
| `l1_2.13e-5` | $L_1$ (Lasso) | $2.13 \times 10^{-5}$ | Moderate-high sparsity penalty | 5 |
| `l1_6.75e-5` | $L_1$ (Lasso) | $6.75 \times 10^{-5}$ | Strong sparsity penalty | 5 |
| `l1_2.13e-4` | $L_1$ (Lasso) | $2.13 \times 10^{-4}$ | Very strong sparsity penalty | 5 |
| `l1_6.75e-4` | $L_1$ (Lasso) | $6.75 \times 10^{-4}$ | Extreme sparsity penalty | 5 |

#### $L_2$ Regularization Grid (8 Candidates)
| Candidate ID | Penalty Type | Value ($\lambda_2$) | Role in Experiment | Total Fold Evaluations |
|:---|:---|:---:|:---|:---:|
| `l2_0.0` | None | $0.0$ | Exact unregularized baseline (paired reference) | 5 |
| `l2_1.00e-4` | $L_2$ (Ridge) | $1.00 \times 10^{-4}$ | Weak weight decay penalty | 5 |
| `l2_3.16e-4` | $L_2$ (Ridge) | $3.16 \times 10^{-4}$ | Moderate-low weight decay penalty | 5 |
| `l2_1.00e-3` | $L_2$ (Ridge) | $1.00 \times 10^{-3}$ | Moderate weight decay penalty | 5 |
| `l2_3.16e-3` | $L_2$ (Ridge) | $3.16 \times 10^{-3}$ | Moderate-high weight decay penalty | 5 |
| `l2_1.00e-2` | $L_2$ (Ridge) | $1.00 \times 10^{-2}$ | Strong weight decay penalty | 5 |
| `l2_3.16e-2` | $L_2$ (Ridge) | $3.16 \times 10^{-2}$ | Very strong weight decay penalty | 5 |
| `l2_1.00e-1` | $L_2$ (Ridge) | $1.00 \times 10^{-1}$ | Extreme weight decay penalty | 5 |

Total evaluations: **80 fold runs** (16 candidates $\times$ 5 folds).

---

## 2. Fixed Experimental Configuration

| Hyperparameter | Value | Description |
|:---|:---|:---|
| **Architecture** | `conv5x5-dense-1024-512-256-128` | Winner from topology search |
| **Optimizer** | Adam | $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ |
| **Learning Rate ($\eta$)** | $10^{-5}$ | Constant learning rate |
| **Loss Function** | SIMM Physics Loss | MSE Data Loss + $\lambda_{\text{SIMM}} \cdot \text{Physics Loss}$ |
| **Physics Weight ($\lambda_{\text{SIMM}}$)** | $0.25$ | Active for $\lvert\alpha\rvert \le 10^\circ$ |
| **Activation Function** | LeakyReLU | Negative slope $\alpha = 0.05$ |
| **Dropout** | None | $p_{\text{drop}} = 0$ |
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
cmake --build build/CNN --target regularization_tuning --parallel
```

### Option B: Direct MPI Compilation (`mpicxx`)
```bash
mkdir -p build/experiments
mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/regularization_tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/regularization_tuning
```

---

## 5. Running the Experiment

### 1. Quick Verification (Smoke Test)
```bash
mpirun -n 2 ./build/CNN/experiments/regularization_tuning --smoke
```

### 2. Full 5-Fold Dual-Axis Sweep (100 Epochs)
```bash
mkdir -p results/cross_validation/regularization_tuning
mpirun -n 16 ./build/CNN/experiments/regularization_tuning \
  --epochs 100 --folds 5 --seed 42 \
  --train-path dataset/cnn_dataset_train.npz \
  --results-dir results/cross_validation/regularization_tuning \
  --diagnostics
```

### Output Artifacts
- `results/cross_validation/regularization_tuning/sweep_l1.csv`: Full $L_1$ sweep metrics across all folds.
- `results/cross_validation/regularization_tuning/sweep_l2.csv`: Full $L_2$ sweep metrics across all folds.
- `results/cross_validation/regularization_tuning/training_history_l1.csv`: 10-epoch checkpoint trajectories for $L_1$.
- `results/cross_validation/regularization_tuning/training_history_l2.csv`: 10-epoch checkpoint trajectories for $L_2$.
- Per-candidate diagnostics under `.../l1/candidate_NNN/fold_MMM/` and `.../l2/candidate_NNN/fold_MMM/`.

## 6. Results

Cross-validation performance across 5 folds and 100 epochs on both regularization axes, including weight shrinkage and numerical stability metrics:

### Metrics & Diagnostic Definitions:
- **Paired $\Delta$ vs $\lambda=0$:** Mean per-fold validation MSE difference relative to the unpenalized baseline ($\text{MSE}_{\text{ref}} - \text{MSE}_{\lambda}$). A negative $\Delta$ indicates that regularization increased error.
- **Improved Folds:** Number of folds (out of 5) where the candidate achieved lower validation MSE than $\lambda=0$ ($0/5$ indicates uniform underperformance).
- **Val-Train Gap:** Difference between validation MSE and training MSE ($\text{MSE}_{\text{val}} - \text{MSE}_{\text{train}}$); quantifies shrinkage of generalization gap.
- **$\sum w^2$:** Sum of squared network weights at epoch 100; directly demonstrates $L_1$/$L_2$ weight shrinkage.
- **$\|w - w_0\|$:** Euclidean displacement of weights from initial initialization $w_0$.
- **Peak Grad Norm:** Maximum parameter gradient norm observed during training across all scopes.

### $L_1$ Regularization Axis

| $\lambda_1$ | Mean Val MSE | Fold SD | Paired $\Delta$ vs $\lambda=0$ | Improved Folds | Val-Train Gap | $\sum w^2$ | $\|w - w_0\|$ | Peak Grad Norm | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0** | **0.0044187** | **0.0007936** | **0.0000000** | — | **+0.0005775** | **2992.6** | **0.91** | **22.54** | **Selected Winner** |
| $6.75\times 10^{-7}$ | 0.0044240 | 0.0007880 | $-0.0000053$ | 2 / 5 | +0.0005677 | 2672.4 | 10.17 | 22.54 | Inferior |
| $2.13\times 10^{-6}$ | 0.0044364 | 0.0008001 | $-0.0000178$ | 2 / 5 | +0.0005621 | 2408.7 | 15.38 | 22.54 | Inferior |
| $6.75\times 10^{-6}$ | 0.0044775 | 0.0008110 | $-0.0000588$ | 0 / 5 | +0.0005694 | 2035.2 | 21.45 | 22.54 | Inferior |
| $2.13\times 10^{-5}$ | 0.0045395 | 0.0008318 | $-0.0001209$ | 0 / 5 | +0.0005452 | 1592.8 | 27.87 | 22.54 | Inferior |
| $6.75\times 10^{-5}$ | 0.0046843 | 0.0009019 | $-0.0002656$ | 0 / 5 | +0.0004556 | 1163.5 | 33.63 | 22.54 | Inferior |
| $2.13\times 10^{-4}$ | 0.0050997 | 0.0009500 | $-0.0006810$ | 0 / 5 | +0.0002851 | 815.0 | 38.08 | 22.55 | Inferior |
| $6.75\times 10^{-4}$ | 0.0064617 | 0.0011891 | $-0.0020431$ | 0 / 5 | +0.0001513 | 587.0 | 41.18 | 22.61 | Severe Penalty |

### $L_2$ Regularization Axis

| $\lambda_2$ | Mean Val MSE | Fold SD | Paired $\Delta$ vs $\lambda=0$ | Improved Folds | Val-Train Gap | $\sum w^2$ | $\|w - w_0\|$ | Peak Grad Norm | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0** | **0.0044187** | **0.0007936** | **0.0000000** | — | **+0.0005775** | **2992.6** | **0.91** | **22.54** | **Selected Winner** |
| $1.00\times 10^{-4}$ | 0.0044831 | 0.0008186 | $-0.0000644$ | 1 / 5 | +0.0005715 | 2225.2 | 14.94 | 22.54 | Inferior |
| $3.16\times 10^{-4}$ | 0.0045800 | 0.0008590 | $-0.0001613$ | 0 / 5 | +0.0005587 | 1808.1 | 20.30 | 22.54 | Inferior |
| $1.00\times 10^{-3}$ | 0.0047676 | 0.0009316 | $-0.0003489$ | 0 / 5 | +0.0005171 | 1374.7 | 25.68 | 22.54 | Inferior |
| $3.16\times 10^{-3}$ | 0.0050917 | 0.0009918 | $-0.0006730$ | 0 / 5 | +0.0003939 | 1008.7 | 30.14 | 22.54 | Inferior |
| $1.00\times 10^{-2}$ | 0.0059023 | 0.0010995 | $-0.0014836$ | 0 / 5 | +0.0001915 | 748.4 | 33.48 | 22.55 | Inferior |
| $3.16\times 10^{-2}$ | 0.0069936 | 0.0012170 | $-0.0025749$ | 0 / 5 | +0.0001567 | 621.1 | 35.56 | 22.70 | Inferior |
| $1.00\times 10^{-1}$ | 0.0084531 | 0.0010800 | $-0.0040345$ | 0 / 5 | +0.0001140 | 582.3 | 36.46 | 24.08 | Severe Penalty |

---

## 7. Analysis and Plotting

### Brief Scientific Analysis
1. **Unpenalized Reference is Optimal:** $\lambda^* = 0$ strictly outperforms all non-zero $L_1$ and $L_2$ penalties. Every positive weight penalty degrades validation MSE monotonically.
2. **Active Weight Shrinkage without Generalization Benefit:** Non-zero regularization actively shrinks the weight norm $\sum w^2$ (from 2,992 down to 582) and increases weight displacement from initialization ($\|w - w_0\|$ from 0.91 to ~41). However, training loss increases much faster than any reduction in generalization gap, indicating that regularization hinders optimization rather than suppressing overfitting.
3. **Cross-Validation Consistency:** The cross-fold standard deviation at $\lambda=0$ ($0.000794$) represents only 18% of the mean ($0.004419$), demonstrating high numerical stability.
4. **Conclusion:** **$\lambda_1^* = 0$ and $\lambda_2^* = 0$ are selected.**

### Plot Generation
Generate the full set of figures from the recorded CSVs:
```bash
python3 CNN/experiments/regularization_tuning/plot_results.py \
    --results-dir results/cross_validation/regularization_tuning
```

Generated figure artifacts in `CNN/experiments/regularization_tuning/plots/`:
- `plot1_val_mse_vs_lambda.png`: Validation MSE vs regularization strength on $L_1$ and $L_2$ axes.
- `plot2_paired_diff.png`: Paired differences $\Delta$ against $\lambda=0$ across all 5 folds.
- `plot3_val_curves_epoch.png`: 10-epoch validation MSE checkpoint trajectories.
- `plot4_sum_w2_vs_lambda.png`: Total weight norm $\sum w^2$ shrinkage as a function of penalty coefficient.
- `plot5_diagnostics_over_epochs.png`: Gradient norms and update ratios throughout training.
