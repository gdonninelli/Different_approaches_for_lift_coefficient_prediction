# Topology Tuning Experiment

Cross-validation experiment that systematically investigates the impact of convolutional feature trunk dimensions and dense regression head architectures on lift coefficient ($C_L$) prediction accuracy.

---

## 1. Scientific Question

Holding the optimizer, learning rate, physics weight and training schedule fixed, **which convolutional kernel size and dense head architecture (width and depth) minimize the physical-unit validation MSE?**

The architecture consists of:
1. **Convolutional Feature Trunk:** A single 2D convolution (`Conv2D(8, KxK, stride K)` with no padding) mapping the $150 \times 150$ Signed Distance Function (SDF) input to 8 feature maps followed by LeakyReLU activation and flattening.
   - $K=5$: produces $8 \times 30 \times 30 = 7,200$ flattened features (total $7,202$ inputs into the dense head when concatenated with $\mathrm{Re}$ and $\alpha$).
   - $K=3$: produces $8 \times 50 \times 50 = 20,000$ flattened features (total $20,002$ inputs into the dense head).
2. **Dense Regression Head:** A sequence of dense layers with widths $(w_1, \dots, w_m)$ and LeakyReLU activations, terminating in a single linear output neuron predicting $C_L$.

### Evaluated Search Space (18 Candidates)

| Candidate Label | Feature Trunk ($K \times K$) | Dense Head Hidden Widths | Total Parameters |
|:---|:---:|:---|:---:|
| `conv5x5-dense-64-32` | $5 \times 5$ | $(64, 32)$ | 463,393 |
| `conv5x5-dense-128-64` *(Baseline)* | $5 \times 5$ | $(128, 64)$ | 930,305 |
| `conv5x5-dense-256-128` | $5 \times 5$ | $(256, 128)$ | 1,877,057 |
| `conv5x5-dense-512-64` | $5 \times 5$ | $(512, 64)$ | 3,724,161 |
| `conv5x5-dense-512-256` | $5 \times 5$ | $(512, 256)$ | 3,819,521 |
| `conv5x5-dense-768-384` | $5 \times 5$ | $(768, 384)$ | 5,832,961 |
| `conv5x5-dense-1024-512` | $5 \times 5$ | $(1024, 512)$ | 7,901,185 |
| `conv5x5-dense-1536-768` | $5 \times 5$ | $(1536, 768)$ | 12,257,281 |
| `conv5x5-dense-2048-1024` | $5 \times 5$ | $(2048, 1024)$ | 16,850,945 |
| `conv5x5-dense-512-256-128` | $5 \times 5$ | $(512, 256, 128)$ | 3,852,545 |
| `conv5x5-dense-512-256-128-64` | $5 \times 5$ | $(512, 256, 128, 64)$ | 3,860,865 |
| `conv5x5-dense-512-256-128-64-32` | $5 \times 5$ | $(512, 256, 128, 64, 32)$ | 3,862,977 |
| `conv5x5-dense-256-128-64-32` | $5 \times 5$ | $(256, 128, 64, 32)$ | 1,887,489 |
| **`conv5x5-dense-1024-512-256-128`** | $5 \times 5$ | $(1024, 512, 256, 128)$ | **8,065,025** |
| `conv5x5-dense-1024-512-256-128-64` | $5 \times 5$ | $(1024, 512, 256, 128, 64)$ | 8,073,345 |
| `conv3x3-dense-128-64` | $3 \times 3$ | $(128, 64)$ | 2,568,705 |
| `conv3x3-dense-256-128` | $3 \times 3$ | $(256, 128)$ | 5,153,409 |
| `conv3x3-dense-512-256` | $3 \times 3$ | $(512, 256)$ | 10,371,969 |

---

## 2. Fixed Experimental Configuration

| Hyperparameter | Value | Description |
|:---|:---|:---|
| **Optimizer** | Adam | $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ |
| **Learning Rate ($\eta$)** | $10^{-5}$ | Constant learning rate |
| **Loss Function** | SIMM Physics Loss | MSE Data Loss + $\lambda_{\text{SIMM}} \cdot \text{Physics Loss}$ |
| **Physics Weight ($\lambda$)** | $0.25$ | Active for $\lvert\alpha\rvert \le 10^\circ$ |
| **Activation Function** | LeakyReLU | $\alpha = 0.05$ |
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
- **Python Environment:** Python 3.8+ with `numpy` and `matplotlib` for analysis and plotting scripts.
- **Working Directory:** All commands must be run from the **repository root**.

---

## 4. Compilation

Build directly from the repository root:

### Option A: CMake (Recommended)
```bash
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target topology_tuning --parallel
```

### Option B: Direct MPI Compilation (`mpicxx`)
```bash
mkdir -p build/experiments
mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/topology_tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/topology_tuning
```

---

## 5. Running the Experiment

### 1. Quick Verification (Smoke Test)
Run a quick 2-rank execution to verify data loading and pipeline integrity:
```bash
mpirun -n 2 ./build/CNN/experiments/topology_tuning
```

### 2. Full 5-Fold Grid Search (90 Fold Evaluations)
Execute the complete search across available MPI ranks:
```bash
mkdir -p results/cross_validation/topology_tuning
mpirun -n 16 ./build/CNN/experiments/topology_tuning \
  | tee results/cross_validation/topology_tuning/run.log
```

### Output Artifacts
The experiment writes execution logs and summary tables:
- `results/cross_validation/topology_tuning/summary.csv`: Aggregated mean and standard deviation of validation MSE per candidate (historically committed under `results/cross_validation/topology_tuning/`).
- `results/cross_validation/topology_tuning/aggregated.csv`: Per-fold validation MSE and sample counts.
- `results/cross_validation/topology_tuning/run_leonardo_51807287.log`: Recorded CINECA Leonardo cluster log.

---

## 6. Results

The grid search evaluated all 18 candidate architectures on the identical 5-fold cross-validation split (90 total fold evaluations).  

### Metrics and Diagnostic Definitions:
- **$\Delta$ vs Baseline (Paired Difference):** The average difference in validation MSE between a candidate and the `{128, 64}` baseline evaluated on the exact same fold splits ($\text{MSE}_{\text{cand}} - \text{MSE}_{\text{base}}$). A **negative $\Delta$** means the candidate achieved a lower error (performed better) than the baseline.
- **Folds Improved:** How many of the 5 cross-validation folds showed better performance than the baseline (e.g., **5/5** indicates consistent superiority across all data splits).
- **Ratio to Mean Baseline ($\text{MSE}_{\text{val}} / \text{MSE}_{\text{mean-predictor}}$):** Compares the model's error against a trivial "dummy" baseline that always predicts the simple dataset average $C_L$ ($\text{MSE}_{\text{mean-predictor}} \approx 0.4974$). A ratio of **$0.0092$** ($0.92\%$) means the neural network eliminates over **$99\%$ of the variance** compared to naive guessing.

| Rank | Candidate Architecture | Mean Validation MSE $\pm$ SD | $\Delta$ vs Baseline | Folds Improved | Ratio to Mean Baseline | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **`conv5x5-dense-1024-512-256-128`** | **$0.004560 \pm 0.000946$** | **$-0.001450$** | **5 / 5** | **0.0092** | **Selected Winner** |
| 2 | `conv5x5-dense-1024-512-256-128-64` | $0.004626 \pm 0.000918$ | $-0.001384$ | 5 / 5 | 0.0093 | Stable |
| 3 | `conv5x5-dense-1024-512` | $0.004851 \pm 0.000911$ | $-0.001158$ | 5 / 5 | 0.0098 | Stable |
| 4 | `conv5x5-dense-512-256-128-64` | $0.004872 \pm 0.001125$ | $-0.001138$ | 5 / 5 | 0.0098 | Stable |
| 5 | `conv5x5-dense-512-256-128` | $0.004898 \pm 0.000970$ | $-0.001112$ | 5 / 5 | 0.0098 | Stable |
| 6 | `conv5x5-dense-768-384` | $0.004932 \pm 0.001099$ | $-0.001077$ | 5 / 5 | 0.0099 | Stable |
| 7 | `conv5x5-dense-512-256` | $0.005052 \pm 0.001067$ | $-0.000958$ | 5 / 5 | 0.0102 | Stable |
| 8 | `conv5x5-dense-512-256-128-64-32` | $0.005061 \pm 0.001031$ | $-0.000949$ | 5 / 5 | 0.0102 | Stable |
| 9 | `conv5x5-dense-512-64` | $0.005145 \pm 0.001014$ | $-0.000865$ | 5 / 5 | 0.0103 | Stable |
| 10 | `conv5x5-dense-1536-768` | $0.005307 \pm 0.001177$ | $-0.000703$ | 4 / 5 | 0.0107 | Stable |
| 11 | `conv5x5-dense-256-128` | $0.005447 \pm 0.001060$ | $-0.000563$ | 4 / 5 | 0.0110 | Stable |
| 12 | `conv5x5-dense-256-128-64-32` | $0.005517 \pm 0.001095$ | $-0.000493$ | 3 / 5 | 0.0111 | Stable |
| 13 | `conv5x5-dense-2048-1024` | $0.005673 \pm 0.001466$ | $-0.000337$ | 3 / 5 | 0.0114 | Overparameterized |
| 14 | `conv5x5-dense-128-64` *(Baseline)* | $0.006010 \pm 0.000823$ | — | — | 0.0121 | Baseline |
| 15 | `conv3x3-dense-256-128` | $0.006157 \pm 0.001051$ | $+0.000147$ | 2 / 5 | 0.0124 | Suboptimal |
| 16 | `conv3x3-dense-128-64` | $0.006330 \pm 0.001099$ | $+0.000321$ | 1 / 5 | 0.0127 | Suboptimal |
| 17 | `conv5x5-dense-64-32` | $0.006425 \pm 0.001292$ | $+0.000415$ | 2 / 5 | 0.0129 | Underparameterized |
| 18 | `conv3x3-dense-512-256` | $0.006459 \pm 0.001318$ | $+0.000449$ | 2 / 5 | 0.0130 | Suboptimal |

### Final Untouched Test Set Evaluation
Refitting the winning candidate (**`conv5x5-dense-1024-512-256-128`**) on the full training dataset (1,713 samples) and evaluating once on the held-out test dataset (`dataset/cnn_dataset_test.npz`, 429 samples) yielded:
- **Test Physical MSE:** **`0.002805`**

---

## 7. Analysis and Plotting

### Brief Scientific Analysis
1. **Trunk Kernel Size ($5\times5$ vs $3\times3$):** The $5\times5$ kernel strictly outperformed the $3\times3$ kernel in every paired comparison. The larger receptive field ($5\times5$ with stride 5) effectively compresses the $150\times150$ SDF while retaining macroscopic airfoil shape characteristics, avoiding the over-dimensional flattened feature map ($20,000$ features) of the $3\times3$ trunk.
2. **Dense Head Depth and Width:** Performance improved consistently by simultaneously increasing both depth (4 hidden layers) and width ($1024 \to 512 \to 256 \to 128$). The first dense layer is the primary capacity driver. Adding depth beyond 4 layers (e.g. 5 layers) or increasing 2-layer width beyond 1024 neurons ($2048 \times 1024$) led to saturation and degradation due to excess parameter count (~16.8M parameters).
3. **Winner Consistency:** The winning configuration improved validation MSE on **5 out of 5 folds** with a large paired margin ($\Delta = -0.001450 \pm 0.000318$), representing a **24.1% error reduction** over the baseline.

### Generating Report Plots
To reproduce the figures in the report:
```bash
python3 CNN/experiments/topology_tuning/analysis/plot_topology_tuning.py
```

Generated figure artifacts in `Report/Images/Chapter04/topology/`:
- `topology_ranking.png`: Bar chart of all 18 architectures sorted by validation MSE with cross-fold error bars.
- `topology_kernel_paired.png`: Paired comparison highlighting the consistent performance advantage of $5\times5$ over $3\times3$ convolution kernels.
- `topology_history.png`: 10-epoch validation checkpoint trajectories across folds for representative candidates.
