# SDF Generator for Airfoil Boundaries

This project generates Signed Distance Function (SDF) matrices from airfoil boundary data. The program reads airfoil coordinates from `.dat` files, reconstructs the airfoil boundary, and computes the SDF values on a grid. The output is saved as a matrix in a text file.

## Repository Structure

```text
SDF/
├── main.cpp                      # Main entry point for the Signed Distance Function execution
├── README.md                     # SDF documentation
├── sdfgen                        # Compiled executable (generated)
├── SDFGenerator.cpp              
├── SDFGenerator.hpp              # Logic for generating the SDF matrices from airfoil boundary coordinates
├── visualization.ipynb           # Notebook used for plotting/visualizing generated SDF outputs
└── data/                         # Geometry profiles and boundary data
    ├── n0012_matrix.txt
    ├── naca0006_matrix.txt
    ├── naca0008_matrix.txt
    ├── naca0010_matrix.txt
    ├── naca0018_matrix.txt
    └── rotate_profiles.py        # Python script to handle profile rotation or preprocessing
```

## Prerequisites
- C++ compiler supporting C++17;
- MPI library for parallel processing.

## Data Format
Input files should be `.dat` files containing whitespace-separated `x` and `z` coordinates per line.

## Compilation
To compile the code, navigate to the SDF directory and run:
```bash
mpic++ main.cpp SDFGenerator.cpp -o sdfgen -std=c++17
```

## How to Run
1. Create a `data` directory in the same location as the compiled `sdfgen` executable.
2. Place your airfoil `.dat` files (e.g., `n0012.dat`) inside the `data` directory.
3. Execute the application using `mpirun` with the desired number of processes. For example, to run with 4 processes:
    ```bash
    mpirun -np 4 ./sdfgen
    ```

## Output
For each input `.dat` file (e.g., `airfoil.dat`), a corresponding SDF matrix file is generated: `airfoil_matrix.txt`. This file contains the SDF grid values in a space-separated matrix format. Positive values are outside the airfoil, negative values are inside.