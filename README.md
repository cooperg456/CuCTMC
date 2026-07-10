# CuCTMC
Some parallel tools for analysis of the Chemical Master Equation type CTMCs

## Installation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=<Debug|Release> -DINSTALL_MODELS=<ON|OFF>
cmake --build build --parallel $(nproc)
cmake --install build --prefix </path/to/install>
```

## Uninstall

```bash
xargs rm < build/install_manifest.txt
```
