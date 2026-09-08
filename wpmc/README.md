# Weighted Parallel Maximum Clique Library

This library provides a simple extension of the PMC library to support **node-weighted graphs**. It is provided as part of the [PARTE: Plane-assisted Robust Transformation Estimation]() implementation.

Important differences between [PMC by Jingnan Shi](https://github.com/jingnanshi/pmc) and this library are:

- This library extends the k-core computation, greedy coloring, and branch-and-bound algorithms to support node-weighted graphs. All algorithms are implemented in such a way that they are equivalent to the original PMC algorithms when all node weights are equal.

- This library only implements the dense version of Algorithm 0, corresponding to the `pmcx_maxclique` implementation.

- This library does not implement the dynamic graph reductions intended to reduce the memory footprint. By avoiding explicit copies of the graph for each thread, its memory footprint remains practical for highly parallelized runs.

- This library assumes that the input graph is weighted and uses a default weight of one for all nodes in unweighted graphs. It is generally slower for unweighted graphs but can be up to 3× faster for weighted graphs.

- There are multiple minor bug fixes and improvements to the original code, including:
  - Fixes for multiple concurrency issues in `pmcx_maxclique`.
  - Improvements to `pmcx_maxclique::branch` that reduce the number of recursive calls.


## Build instructions
Requirements: CMake 3.24 or newer, a C++20 compiler, and OpenMP.
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```



## References

- PMC paper: Parallel Maximum Clique Algorithms with Applications to Network Analysis and Storage on [arXiv](https://arxiv.org/pdf/1302.6256)
- PMC implementation: [PMC by Ryan A. Rossi](https://github.com/ryanrossi/pmc)
- Forked reimplementation we used: [PMC by Jingnan Shi](https://github.com/jingnanshi/pmc)
- PARTE paper: [PARTE: Plane-assisted Robust Transformation Estimation]()
