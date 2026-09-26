<h1 align="center">
  <img src="assets/PARTE.svg" alt="PARTE" width="320"><br>
</h1>

# Plane-Assisted Robust Transformation Estimation
C++ implementation and Python bindings for Plane-Assisted Robust Transformation Estimation (PARTE).

<p align="center">
  <img src="assets/PARTE-showcase.gif" alt="PARTE point-cloud registration">
</p>



**NOTE**: This implementation of PARTE uses [CLIPPER+](https://github.com/ariarobotics/clipperp) for graph-based outlier rejection. Positive integer node weights are represented by duplicating vertices before running the solver. CLIPPER+ is an approximate maximum-clique solver, so the returned clique is not guaranteed to be globally optimal.


## Build

#### Requirements
- CMake 3.21 or newer
- A C++20 compiler with OpenMP support
- Eigen 5.0.0 (automatically downloaded)
- Open3D 0.19.0 (automatically downloaded)

#### PARTE
Initialize and clone the submodules:
```sh
git submodule update --init --recursive
```

Run this command to download Eigen and Open3D requirements and configure the project:
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

You can also use the following command to use your system's Open3D and Eigen:
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_OPEN3D=ON -DUSE_SYSTEM_EIGEN=ON
cmake --build build
```


To build the python bindings, install nanobind using `pip install nanobind` and run the following command:
```sh
cmake -S . -B build -DBUILD_PYTHON_MODULE=ON
cmake --build build
```


Run the algorithm on two point clouds:
```sh
build/parte assets/source.ply assets/target.ply 0.05 # voxel-size of 5cm
```
or using the python demo (requires numpy and open3d installed)
```sh
python python/demo.py
```

Note that PARTE assumes that the input point clouds are already downsampled, pass `--raw` if using the original point clouds.

## Run the benchmarks
Download and extract the benchmark datasets from [https://parte.pages.dev/](https://parte.pages.dev/).
The layout of the benchmark datasets is as follows:
```text
datasets/
  indoor.txt
  outdoor.txt
  additional.txt
  scans/
    3DMatch/
      7-scenes-redkitchen/
        cloud_bin_0.ply
        cloud_bin_1.ply
        ...
        voxel_size
  benchmarks/
    3DMatch/
      7-scenes-redkitchen/gt.log
```

Run indoor benchmarks:
```sh
build/parte-benchmark datasets/indoor.txt results/indoor
```

Run outdoor benchmarks:
```sh
build/parte-benchmark datasets/outdoor.txt results/outdoor
```

Run ETH and RESSO benchmarks:
```sh
build/parte-benchmark datasets/additional.txt results/additional
```


The `results/` directory will contain the estimated transformation of each pair in `parte.log` alongside a summary under `summary.txt`.

