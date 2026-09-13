# Activation-Function Tuning Experiment

Cross-validation experiment evaluating classical and piecewise-linear activation functions applied across the convolutional trunk and dense regression head for lift coefficient ($C_L$) prediction.

---

## 1. Scientific Question

**Which activation function provides the lowest physical-unit validation MSE while ensuring stable gradient flow and preventing saturation or dead neurons across all hidden layers?**

The activation function is applied identically after the 2D convolution and after every hidden dense layer (`Dense(1024)`, `Dense(512)`, `Dense(256)`, `Dense(128)`), while the output layer remains linear.

### Evaluated Search Space (6 Candidates)

| Candidate ID | Activation Function | Mathematical Formulation | Hyperparameter |
|:---|:---|:---:|:---:|
| `tanh` | Hyperbolic Tangent (Tanh) | $\sigma(z) = \tanh(z)$ | None |
| `sigmoid` | Logistic Sigmoid | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | None |
| `relu` | Rectified Linear Unit (ReLU) | $\sigma(z) = \max(0, z)$ | None |
| `leakyrelu-alpha-0.01` | Leaky ReLU ($\alpha=0.01$) | $\sigma(z) = \max(\alpha z, z)$ | $\alpha = 0.01$ |
| `leakyrelu-alpha-0.05` | Leaky ReLU ($\alpha=0.05$) | $\sigma(z) = \max(\alpha z, z)$ | $\mathbf{\alpha = 0.05}$ |
| `leakyrelu-alpha-0.1` | Leaky ReLU ($\alpha=0.10$) | $\sigma(z) = \max(\alpha z, z)$ | $\alpha = 0.10$ |

Total evaluations: **30 fold runs** (6 candidates $\times$ 5 folds).

---

## 2. Fixed Experimental Configuration

| Hyperparameter | Value | Description |
|:---|:---|:---|
| **Architecture** | `conv5x5-dense-1024-512-256-128` | Winner from topology search |
| **Optimizer** | Adam | Winner from optimizer search |
| **Learning Rate ($\eta$)** | $10^{-5}$ | Constant learning rate |
| **Loss Function** | SIMM Physics Loss | MSE Data Loss + $\lambda_{\text{SIMM}} \cdot \text{Physics Loss}$ |
| **Physics Weight ($\lambda$)** | $0.10$ | Active for $\lvert\alpha\rvert \le 10^\circ$ |
| **Regularization / Dropout** | None | Winner from regularization search |
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
- **Python Environment:** Python 3.8+ with `numpy` and `matplotlib` for analysis and plotting scripts.
- **Working Directory:** All commands must be run from the **repository root**.

---

## 4. Compilation

Build directly from the repository root:

### Option A: CMake (Recommended)
```bash
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target activation_function_tuning --parallel
```

### Option B: Direct MPI Compilation (`mpicxx`)
```bash
mkdir -p build/experiments
mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/activation_function_tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/activation_function_tuning
```

---

## 5. Running the Experiment

### 1. Quick Verification (Smoke Test)
```bash
mpirun -n 2 ./build/CNN/experiments/activation_function_tuning --smoke
```

### 2. Full 5-Fold Cross-Validation Sweep (100 Epochs)
```bash
mkdir -p results/cross_validation/activation-function-tuning
mpirun -n 16 ./build/CNN/experiments/activation_function_tuning \
  --train-path dataset/cnn_dataset_train.npz \
  --test-path dataset/cnn_dataset_test.npz \
  --results-dir results/cross_validation/activation-function-tuning \
  --diagnostic
```

### Output Artifacts
- `results/cross_validation/activation-function-tuning/fold_results.csv`: Per-candidate and per-fold training and validation MSE.
- `results/cross_validation/activation-function-tuning/training_history.csv`: Checkpoint history every 10 epochs.
- `results/cross_validation/activation-function-tuning/summary.txt`: Full human-readable cross-validation ranking.
- Per-candidate diagnostics under `.../search/candidate_NNN/fold_MMM/`.

---

## 6. Results

Cross-validation performance across 5 folds and 100 epochs, alongside complete numerical stability diagnostics:

| Rank | Activation Function | Mean Validation MSE | Fold SD | Peak Grad Norm | Peak Weight Update Ratio | Non-Finite Events | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **`leakyrelu-alpha-0.05`** | **0.00442742** | **0.000865257** | **20.14** | **$4.50\times 10^{-4}$** | **None** | **Selected Winner** |
| 2 | `leakyrelu-alpha-0.01` | 0.00444772 | 0.000876077 | 20.64 | $4.47\times 10^{-4}$ | None | Stable |
| 3 | `relu` | 0.00445429 | 0.000875881 | 20.81 | $3.31\times 10^{-4}$ | None | Stable |
| 4 | `leakyrelu-alpha-0.1` | 0.00454457 | 0.000965707 | 20.79 | $4.42\times 10^{-4}$ | None | Stable |
| 5 | `tanh` | 0.00546562 | 0.001045450 | 111.51 | $3.01\times 10^{-4}$ | None | Transient Gradient Spikes |
| 6 | `sigmoid` | 0.00897140 | 0.001250090 | 18.22 | $6.15\times 10^{-4}$ | None | Severe Saturation |

### Final Untouched Test Set Evaluation
Refitting the winning candidate (**`LeakyReLU(alpha=0.05)`**) on the full training dataset (1,713 samples) and evaluating once on the held-out test dataset (`dataset/cnn_dataset_test.npz`, 429 samples) yielded:
- **Final Untouched Test Physical MSE:** **`0.00273892`**

---

## 7. Analysis and Plotting

### Brief Scientific Analysis
1. **Piecewise-Linear Advantage:** ReLU and LeakyReLU variants strictly outperformed smooth saturating activations (Tanh and Sigmoid). Sigmoid suffered severe gradient vanishing with validation error over twice as high ($0.008971$). Tanh exhibited large transient gradient spikes (peak norm $111.51$).
2. **LeakyReLU Superiority over Standard ReLU:** LeakyReLU with $\alpha = 0.05$ achieved the lowest validation MSE ($0.004427$) and the lowest fold-to-fold standard deviation ($0.000865$). The non-zero gradient for negative pre-activations ($\alpha = 0.05$) prevents "dying ReLU" units during early optimization.
3. **Healthy Update Ratios:** Parameter update ratios ($r_\theta \approx 4.5\times 10^{-4}$) stayed stable and well-bounded across all layers without exploding or vanishing.

### Plot Generation
Generate the full set of comparison and diagnostic plots:
```bash
python3 CNN/experiments/activation_function_tuning/plot_results.py \
    --results-dir results/cross_validation/activation-function-tuning
```

Generate report figures:
```bash
python3 CNN/experiments/activation_function_tuning/analysis/plot_report_activation.py
```

Generated figure artifacts in `CNN/experiments/activation_function_tuning/plots/`:
- `plot1_val_mse_by_activation.png`: Mean validation MSE with cross-fold standard deviation bars.
- `plot2_val_mse_by_fold.png`: Validation MSE broken down for every candidate and fold.
- `plot3_val_curves_epoch.png`: Validation MSE progression at 10-epoch checkpoints.
- `plot4_diagnostics_over_epochs.png`: Synchronized gradient norms and parameter update ratios over epochs.
