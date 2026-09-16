# Final Production Training and Diagnostics

Production training configuration and developmental trial analysis combining the optimal architectural and optimization hyperparameters established during the tuning campaign.

---

## 1. Scientific Question

**Core Question:** **Integrating all winning tuning selections, how does the production model perform with balanced batching ($B=257$), robust "20/20" early stopping, and extended training budgets on a 90/10 data split, and how is test error partitioned across angle-of-attack regimes ($|\alpha| \le 10^\circ$ vs $|\alpha| > 10^\circ$)?**

The production training integrates:
1. **Winning Architecture:** `conv5x5-dense-1024-512-256-128` (8,065,025 parameters).
2. **Winning Activation:** `LeakyReLU(alpha=0.05)`.
3. **Winning Optimizer & Schedule:** Adam with constant learning rate $\eta = 10^{-3}$.
4. **Regularization Policy:** $L_1 = 0$, $L_2 = 0$, $p_{\text{drop}} = 0$, with physics weight $\lambda_{\text{SIMM}} = 0.10$ for pre-stall physical consistency.
5. **Exact Balanced Batching:** 6 exact batches of size $257$ ($1542 = 6 \times 257$), eliminating the unbalanced mini-batch tail of size 6 observed at $B=64$.
6. **Robust "20/20" Early Stopping:** Minimum 20 epochs and a patience of 20 consecutive epochs where $\text{MSE}_{\text{val}} > 1.15 \times \text{MSE}_{\text{train}}$, automatically restoring the best validation checkpoint.

---

## 2. Fixed Experimental Configuration

| Hyperparameter | Value | Description |
|:---|:---|:---|
| **Architecture** | `conv5x5-dense-1024-512-256-128` | $5\times 5$ trunk + 4-layer dense head |
| **Total Parameters** | 8,065,025 | 208 conv + 8,064,817 dense |
| **Optimizer** | Adam | $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ |
| **Learning Rate ($\eta$)** | $10^{-3}$ | Constant effective learning rate |
| **Loss Function** | SIMM Physics Loss | MSE Data Loss + $\lambda_{\text{SIMM}} \cdot \text{Physics Loss}$ |
| **Physics Weight ($\lambda$)** | $0.10$ | Active for pre-stall $\lvert\alpha\rvert \le 10^\circ$ |
| **Activation Function** | LeakyReLU | Negative slope $\alpha = 0.05$ |
| **Gradient Clipping** | $1.0$ | Componentwise clipping |
| **Batch Construction** | Balanced ($B=257$) | 6 exact batches per epoch |
| **Stopping Policy** | Robust 20/20 Rule | Ratio $\rho = 0.15$, min 20 epochs, patience 20 |
| **Maximum Budget** | $1,200$ epochs | 6 updates/epoch ($5,160$ updates completed) |
| **Data Partition** | 90 / 10 Split | 1,542 train / 171 validation / 429 held-out test |
| **Seed** | `42` | Deterministic initialization |

## 3. Requirements and Prerequisites

- **Toolchain:** C++20 compliant compiler (`g++` $\ge 11$ or `clang++` $\ge 13$) with an MPI implementation (OpenMPI $\ge 4.0$ or MPICH).
- **Build System:** CMake $\ge 3.16$ or Makefile (`make cnn`).
- **Dataset Files:** `dataset/cnn_dataset_train.npz` (1,713 samples) and `dataset/cnn_dataset_test.npz` (429 samples), generated via `python3 build_dataset.py --seed 42`.
- **Python Environment:** Python 3.8+ with `numpy` and `matplotlib` for analysis and plotting scripts.
- **Working Directory:** All commands must be run from the **repository root**.

---

## 4. Compilation

Build the production executable from the repository root:

### Option A: CMake / Makefile (Recommended)
```bash
make cnn
# or directly with CMake:
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target cnn_executable --parallel
```

---

## 5. Running Production Training

### Standard Execution Across MPI Ranks
Execute the production training driver from the repository root:
```bash
mkdir -p results/ordinary-training
mpirun -n 16 ./build/CNN/cnn_executable \
  --train-path dataset/cnn_dataset_train.npz \
  --test-path dataset/cnn_dataset_test.npz \
  --results-dir results/ordinary-training \
  --epochs 1200 \
  --batch-size 257 \
  --seed 42 \
  --lr 1e-3 \
  --physics-weight 0.10
```

### SLURM Cluster Submission
Use the template SLURM script:
```bash
sbatch slurm.sh
```

### Output Artifacts
- `epoch_metrics.csv`: Full epoch-by-epoch physical training and validation MSE.
- `final_metrics.csv`: Restored checkpoint metrics and evaluation summary.
- `test_metrics.csv`: Held-out test performance partitioned by angle regimes.
- `gradient_norms.csv` & `parameter_update_ratios.csv`: Diagnostic stability histories.

---

## 6. Results

### Iterative Developmental Progression

| Stage / Trial | Batch Size ($B$) | Budget | Completed Epochs | Selected Epoch | Train MSE | Val MSE | Test MSE | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `trial 1` | 64 | 200 | 200 | 162 | 0.001957 | 0.001714 | 0.001647 | Unbalanced tail (24 $\times$ 64 + 6) |
| `trial 2` | 64 | 200 | 191 | 139 | 0.002081 | 0.001719 | 0.001894 | Balanced (17 $\times$ 62 + 8 $\times$ 61) |
| `trial 3` | 257 | 200 | 200 | 183 | 0.002614 | 0.002325 | 0.001998 | Exact batches ($6 \times 257$) |
| `trial 4` | 257 | 834 | 319 | 318 | 0.001518 | 0.001425 | 0.001512 | Extended budget (early stop) |
| 20/20 policy, seed 0 | 257 | 834 | 834 | 820 | 0.000389 | 0.000552 | 0.000420 | Robust stopping allows convergence |
| 20/20 policy, seed 1 | 257 | 834 | 159 | 139 | 0.002924 | 0.004752 | 0.002381 | Intermediate seed |
| 20/20 policy, seed 42 | 257 | 834 | 834 | 797 | 0.000395 | 0.000435 | 0.000487 | Selected checkpoint at epoch 797 |
| **`trial 5` (Production)** | **257** | **1200** | **860** | **797** | **0.000395** | **0.000435** | **0.000487** | **Final Production Model** |

### Partitioned Test Set Performance by Angle of Attack

| Test Subset | Sample Count | Physical MSE | Physical RMSE | Error Difference vs Low-Angle | High / Low Error Ratio |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Overall Test Set** | **429** | **0.000486795** | **0.02206** | — | — |
| Pre-Stall ($\lvert\alpha\rvert \le 10^\circ$) | 363 | 0.000401836 | 0.02005 | Baseline | 1.00 |
| Post-Stall ($\lvert\alpha\rvert > 10^\circ$) | 66 | 0.000954069 | 0.03089 | +0.000552233 | 2.374 |

---

## 7. Analysis and Plotting

### Brief Scientific Analysis
1. **Impact of Batch Balancing:** Balanced batching ($B=257$, 6 exact batches) eliminated gradient variance spikes caused by small sample tails, stabilizing Adam's second-moment estimates.
2. **Impact of 20/20 Stopping Policy:** The sensitive `first_ratio` policy caused premature termination at epochs 5–6 with poor MSE (>0.009). The robust "20/20" policy allowed training to navigate transient fluctuations, achieving a **20-fold error reduction** (test MSE $0.000487$).
3. **High-Angle Challenge:** The post-stall regime ($|\alpha| > 10^\circ$) exhibited $2.374\times$ higher MSE ($0.000954$ vs $0.000402$). Although high-angle samples represent only 15.38% of the test set, they account for 30.15% of the total squared error due to non-linear flow separation dynamics.

### Plot Generation
Generate report figures:
```bash
python3 CNN/experiments/final_training/analysis/plot_report_final_training.py
```

Generate batch balancing and stability comparison plots:
```bash
python3 CNN/experiments/final_training/analysis/plot_trial_batch_instability.py
```

Generated figure artifacts in `Report/Images/Chapter04/final_training/`:
- `final_training_history.png`: Linear and logarithmic physical MSE training curves from epoch 1 to stopping epoch 860.
- `test_mse_by_angle.png`: Comparison of test physical MSE partitioned into low-angle ($|\alpha|\le 10^\circ$) and high-angle ($|\alpha|> 10^\circ$) regimes.
- `final_trial1_trial2_mse_comparison.png`: Batching excursion comparison between unbalanced ($B=64$) and balanced ($B=257$) regimes.
- `final_trial1_trial2_gradient_update.png`: Pre-clipping RMS gradient norms and dense update ratios across layers.
