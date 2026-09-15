# Ice Acceleration Model on Airplane Wing

This project reproduces the CNN-based approach from the AMSC and NAML Project paper for predicting aerodynamic coefficients on airfoil profiles. The core idea is to replace traditional CFD simulations with a data-driven surrogate model: given the airfoil geometry (encoded as a Signed Distance Function) and flight conditions (Reynolds number and Angle of Attack), the model predicts the resulting aerodynamic forces.

Unlike a standard purely data-driven neural network, this implementation adopts a **physics-informed cost function** (SIMM loss). The loss combines a standard MSE term on the predicted coefficients with a physics-based regularization term that enforces the known linear relationship between the lift coefficient and the angle of attack in the small-angle regime. This physical prior acts as a soft constraint, guiding the model toward physically consistent predictions and improving generalization.

The pipeline is composed of two main modules:
- **SDF Generator** — converts raw airfoil boundary coordinates into a grid-based Signed Distance Function representation.
- **CNN Model** — a custom C++/MPI CNN framework built from scratch, performing manual forward/backward passes with distributed gradient averaging via MPI.

---

## Repository Structure

```text
.
├── Makefile                          # Top-level build automation (CNN, SDF, experiments)
├── build_dataset.py                  # Dataset preparation and splitting utility
├── download_notebook.py              # Kaggle dataset download utility
├── training.sh                       # Training helper script
├── PR_HISTORY.md                     # Development logs and pull request history
├── assets/                           # Figures and diagrams for documentation
│   ├── SDF.png
│   ├── cnn_architecture.png
│   └── training trend.png
│
├── SDF/                              # Geometry preprocessing module
│   ├── README.md                     # Dedicated SDF documentation and usage guide
│   ├── main.cpp                      # Parallel SDF generator CLI
│   ├── SDFGenerator.hpp / .cpp       # Winding number and segment projection kernels
│   ├── visualization.ipynb           # SDF field visualization notebook
│   └── data/                         # Airfoil boundary .dat files
│
├── CNN/                              # Custom C++/MPI Neural Network Framework
│   ├── CMakeLists.txt                # Builds core library, CLI, tests, and all experiments
│   ├── main.cpp                      # Production CLI & single-training driver
│   ├── src/
│   │   ├── core/                     # Tensor storage and PINN SIMM Loss
│   │   ├── data/                     # Dataset loading (.npz) and batch construction
│   │   ├── layers/                   # Conv2D, Dense, LeakyReLU, Flatten, Concat, Dropout
│   │   ├── model/                    # CNNModel and ModelFactory
│   │   ├── optimizers/               # Adam, SGD, AdaGrad, RMSprop and LRSchedulers
│   │   ├── training/                 # Trainer (balanced batching, 20/20 early stopping, diagnostics)
│   │   └── tuning/                   # CrossValidator (K-Fold), SearchSpace, TrialConfig
│   │
│   ├── experiments/                  # Specialized hyperparameter tuning drivers
│   │   ├── topology_tuning/          # Conv kernel sizes & dense head depth/width sweeps
│   │   ├── dropout_tuning/           # Inverted dropout rate ablation (p in [0, 0.5])
│   │   ├── optimizer_tuning/         # Adam vs RMSprop vs AdaGrad vs SGD+Momentum vs SGD
│   │   ├── physics_weight_tuning/    # SIMM thin-airfoil loss weight (lambda) ablation
│   │   ├── regularization_tuning/    # L1 and L2 weight penalty grid search
│   │   ├── activation_function_tuning/# LeakyReLU vs ReLU vs Tanh vs Sigmoid
│   │   ├── learning_rate_tuning/     # LR magnitude & schedule (step, cosine, warmup)
│   │   └── final_training/           # Final production 90/10 training & stopping policy
│   │
│   ├── analysis/                     # Diagnostics and metric visualization scripts
│   │   ├── plot_training_diagnostics.py
│   │   └── plot_training_physical_mse.py
│   └── tests/                        # Unit and distributed regression tests
│       └── test_cross_validation.cpp
│
├── dataset/                          # NPZ dataset directory (train/test splits)
├── results/                          # Output logs, metrics CSVs, and diagnostic artifacts
├── docs/                             # Additional technical documentation
│   ├── cross_validation.md
│   └── training_diagnostics_plots.md
└── Report/                           # LaTeX source files for the project report
```

---

## Quickstart and Compilation

The repository provides a top-level `Makefile` that wraps CMake and MPI builds.

### Prerequisites
- **C++ Compiler**: Supporting C++20 (for CNN) and C++17 (for SDF).
- **MPI Library**: OpenMPI, MPICH, or Intel MPI (`mpic++`, `mpirun`).
- **Build Tools**: CMake $\ge 3.16$, `make`.
- **Python 3**: With `numpy` and `matplotlib` (for datasets and diagnostics plotting).

### 1. Build Everything via Makefile
From the repository root:
```bash
# Build the CNN executable, all 8 experiment binaries, unit tests, and the SDF generator
make all

# Or build specific targets:
make cnn              # Configures and compiles CNN + all experiments into build/CNN/
make sdf              # Compiles the SDF generator into build/sdf/sdfgen
make test             # Runs CTest suite (serial and 2-rank MPI)
make clean            # Removes all build directories
make help             # Displays all available targets
```

### 2. Build via CMake Directly
```bash
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --parallel
ctest --test-dir build/CNN --output-on-failure
```
*Note: This automatically builds `cnn_executable`, `cnn_tests`, and each standalone experiment binary into `build/CNN/experiments/<experiment_name>`.*

---

## Dataset Setup

The CNN models train on `.npz` archives containing $150 \times 150$ SDF matrices, scalar pairs $(\mathrm{Re}, \alpha)$, and corresponding CFD lift coefficients $C_L$.

Download the pre-generated dataset from Kaggle:
[SDF Symmetric Airfoil High Reynolds Number](https://www.kaggle.com/datasets/giulioenzodonninelli/sdf-symmetric-airfoil-high-reynolds-number)

```bash
mkdir -p dataset
kaggle datasets download -d giulioenzodonninelli/sdf-symmetric-airfoil-high-reynolds-number -p dataset --unzip
```

Alternatively, to regenerate the dataset from raw airfoil coordinate files using the SDF generator:
```bash
make dataset
```

---

## Modules and Documentation

### 1. Geometry Preprocessing: SDF Generator
The SDF generator converts 2D boundary polygons from `.dat` files into $150 \times 150$ Signed Distance Field matrices using an exact parallel winding-number and ray-casting algorithm.
- **Detailed Documentation**: [SDF Module Guide (SDF/README.md)](SDF/README.md)
- **Run Standalone**:
  ```bash
  cd SDF
  mpirun -np 4 ../build/sdf/sdfgen
  ```

---

### 2. Hyperparameter Tuning Experiments
All systematic ablation sweeps are isolated in dedicated subprojects under `CNN/experiments/`. Each subproject contains its own driver, parameter grid, and documentation.

| Experiment | Focus and Key Questions | Documentation and Guide |
| :--- | :--- | :---: |
| **Topology Tuning** | Kernel size ($5\times5$ vs $3\times3$) & Dense head width/depth | [topology_tuning/README.md](CNN/experiments/topology_tuning/README.md) |
| **Dropout Tuning** | Impact of inverted dropout ($p \in \{0, 0.1, 0.2, 0.3, 0.5\}$) on generalization | [dropout_tuning/README.md](CNN/experiments/dropout_tuning/README.md) |
| **Optimizer Tuning**| Adam vs. RMSprop vs. AdaGrad vs. SGD+Momentum vs. Plain SGD | [optimizer_tuning/README.md](CNN/experiments/optimizer_tuning/README.md) |
| **Physics Weight** | SIMM thin-airfoil loss penalty weight ($\lambda \in [0, 2]$) | [physics_weight_tuning/README.md](CNN/experiments/physics_weight_tuning/README.md) |
| **Weight Regularization**| $L_1$ Lasso sparsity and $L_2$ Ridge shrinkage penalties | [regularization_tuning/README.md](CNN/experiments/regularization_tuning/README.md) |
| **Activation Functions**| LeakyReLU (slopes $\alpha \in \{0.01, 0.05, 0.1\}$), ReLU, Tanh, Sigmoid | [activation_function_tuning/README.md](CNN/experiments/activation_function_tuning/README.md) |
| **Learning Rate & Schedule**| Constant, Step Decay, Cosine Annealing, and Warmup schedules | [learning_rate_tuning/README.md](CNN/experiments/learning_rate_tuning/README.md) |
| **Final Production** | Balanced batching, "20/20" early stopping policy, and final model evaluation | [final_training/README.md](CNN/experiments/final_training/README.md) |

#### Running an Individual Experiment:
After building with `make cnn`, execute any experiment binary directly with `mpirun`:
```bash
# Example: Run the optimizer comparison across 4 MPI ranks
mpirun -n 4 ./build/CNN/experiments/optimizer_tuning --results-dir results/optimizer_tuning

# Example: Run the learning rate schedule sweep
mpirun -n 4 ./build/CNN/experiments/learning_rate_tuning --results-dir results/learning_rate_tuning
```

---

## Running the Main Production CNN Model

From the repository root, run the primary executable with customizable CLI options:

### 5-Fold Cross-Validation
```bash
mpirun -n 4 ./build/CNN/cnn_executable --cross-validate --folds 5 --epochs 100 --batch-size 64
```

### Production Training
```bash
mpirun -n 4 ./build/CNN/cnn_executable \
  --epochs 200 \
  --batch-size 257 \
  --learning-rate 1e-3 \
  --activation leakyrelu \
  --alpha 0.05 \
  --diagnostics \
  --results-dir results \
  --experiment production \
  --run-name seed_42
```

Command line arguments summary:
- `--epochs N`: Total training epochs.
- `--batch-size B`: Global batch size (divided evenly across MPI ranks).
- `--learning-rate LR`: Base learning rate.
- `--activation ACT`: Activation function (`leakyrelu`, `relu`, `tanh`, `sigmoid`).
- `--alpha VAL`: LeakyReLU negative slope (default: 0.05).
- `--dropout P`: Dropout probability on dense head (default: 0.0).
- `--diagnostics`: Enables comprehensive per-epoch metric logging.
- `--cross-validate`: Runs $K$-fold cross-validation instead of single training.

---

## Diagnostics and Analysis Plotting

When diagnostics are enabled, the training engine exports structured telemetry into `results/<experiment>/<run>/`:
- `epoch_metrics.csv`: SIMM loss, physical training MSE, physical validation MSE, and effective learning rate per epoch.
- `gradient_norms.csv`: Synchronized RMS, max, and layer-wise gradient norms before clipping.
- `parameter_update_ratios.csv`: Actual parameter movement relative to weight magnitude ($r_\theta$).
- `activation_statistics.csv`: Pre- and post-activation mean, variance, min, max per layer.
- `activation_histograms.csv`: Bounded 64-bin activation distributions over $[-10, 10]$.
- `final_metrics.csv` & `test_metrics.csv`: Restored checkpoint errors and $|\alpha| \le 10^\circ$ regime splits.

Generate diagnostic plots without a display server:
```bash
python3 CNN/analysis/plot_training_diagnostics.py \
  --input results/production/seed_42 \
  --output-dir results/production/seed_42/plots
```

See [docs/training_diagnostics_plots.md](docs/training_diagnostics_plots.md) for detailed descriptions of each diagnostic figure.

---

## CINECA Leonardo Cluster Deployment

Instructions for deploying and executing on the CINECA Leonardo supercomputer:

### 1. Environment Setup & Compilation
Load the required modules on the **login node**:
```bash
module purge
module load python/3.11.7
module load gcc/11.3.0
module load intel-oneapi-compilers
module load intel-oneapi-mpi

make cnn
```

### 2. Slurm Job Submission
Create and submit a Slurm script `run_cnn.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=cnn_training
#SBATCH --account=<project_account>
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=01:00:00
#SBATCH --output=cnn_%j.out
#SBATCH --error=cnn_%j.err

cd $SLURM_SUBMIT_DIR

module purge
module load python/3.11.7
module load gcc/11.3.0
module load intel-oneapi-compilers
module load intel-oneapi-mpi

srun ./build/CNN/cnn_executable --batch-size 257 --epochs 800 --learning-rate 1e-3 --diagnostics
```

Submit and monitor:
```bash
sbatch run_cnn.sh        # Submit job
squeue -u <username>     # Check queue status
tail -f cnn_*.out        # Follow execution log
```
