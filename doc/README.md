# Building MPL Documentation

MPL uses a documentation pipeline combining [Doxygen](https://www.doxygen.nl/) for C++ API reference extraction and [Sphinx](https://www.sphinx-doc.org/) (with [Breathe](https://breathe.readthedocs.io/) and the [Furo](https://pradyunsg.me/furo/) theme) for generating HTML documentation.

## Prerequisites & Conda Environment

A reproducible Conda environment definition is provided in `doc/environment.yml`. It includes all necessary dependencies: Python, Sphinx, Breathe, Furo theme, Sphinx extensions, Doxygen, and Graphviz (`dot`).

### 1. Create and Activate the Environment

Create the `mpl-doc` Conda environment:

```bash
conda env create -f doc/environment.yml
```

Activate the environment before configuring or building the documentation:

```bash
conda activate mpl-doc
```

To update an existing environment when dependencies change:

```bash
conda env update -f doc/environment.yml --prune
```

## Building the Documentation

With the `mpl-doc` Conda environment activated:

### 2. Configure CMake

Enable documentation building with `-DMPL_BUILD_DOCUMENTATION=ON`:

```bash
cmake -B cmake-build-documentation -DMPL_BUILD_DOCUMENTATION=ON
```

### 3. Build Documentation Target

Generate the complete documentation suite:

```bash
cmake --build cmake-build-documentation --target documentation
```

This target runs:
1. Doxygen XML generation into `<build-dir>/doxygen/xml/`.
2. Sphinx HTML build into `<build-dir>/sphinx/html/`.

### 4. Inspect Output

Open the generated HTML documentation in any web browser:

```bash
# Linux
xdg-open cmake-build-documentation/sphinx/html/index.html

# macOS
open cmake-build-documentation/sphinx/html/index.html
```

## Cleaning Build Artifacts

To remove generated documentation files (Doxygen XML and Sphinx HTML output):

```bash
cmake --build cmake-build-documentation --target documentation_clean
```
