# sgiga

**S**econd-**G**radient **I**so**G**eometric **A**nalysis — a small nonlinear FEM/IGA
package for continuum mechanics, focused on:

- **Hyperelasticity** (Ogden-type first-gradient continua, compressible and incompressible)
- **Second-gradient continuum theory**, including beam-network material models used to
  represent pantographic sheets and orthotropic blocks (mechanical metamaterials)
- **Isogeometric analysis**: B-spline/Bezier discretization in 1D/2D/3D, alongside a
  classical Lagrange discretization for comparison
- A small set of nonlinear solvers (Newton, Euler backward, generalized-α) built around
  a sparse-assembly `Model`/`Coo` container

This is a trimmed, cleaned-up personal fork of a larger research codebase (see
[Acknowledgments](#acknowledgments)), kept to the parts needed to run the 5 examples
below.

## Installation

Requires Python ≥3.10.

```bash
git clone <this-repo-url>
cd sgiga
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Examples

All examples live in `examples/` and are run directly with Python from the repo root;
each writes VTK/ParaView output to a gitignored `results/` folder (`continuum_batch.py`
writes to `output/` instead — both are ignored by `.gitignore`).

```bash
python examples/continuum.py
```

| Script | What it demonstrates |
| --- | --- |
| `continuum.py` | First-gradient hyperelastic (Ogden) cube under prescribed face displacement, B-spline (IGA) discretization, static Newton solve. |
| `continuum_lagrange.py` | Same first-gradient hyperelastic formulation on a 2D rectangle, using a classical Lagrange discretization instead of B-splines, for comparison. |
| `continuum_second_gradient.py` | Second-gradient continuum (`Pantobox_beam_network` material) cube under tension — the microstructure-aware theory behind the pantographic/metamaterial examples below. |
| `continuum_batch.py` | Batch-style driver for first-gradient hyperelastic solves (Treloar rubber test data) over a larger 2D mesh, with solutions serialized via `dill`. |
| `pantographic_ortho_block.py` | 3D pantographic/orthotropic block under tension and torsion — a second-gradient beam-network metamaterial on a larger IGA mesh. |

Output `.vtu`/`.pvd` files can be opened in [ParaView](https://www.paraview.org/) to
visualize the deformed configuration.

## Project structure

```
sgiga/
  discretization/   B-spline, Lagrange, and mesh (1D/2D/3D) discretization
  math/              small linear-algebra and numerical-derivative utilities
  model/             continuum/material models, forces, constraints, rigid bodies
  solver/            Newton, Euler backward, generalized-alpha solvers
  utility/           sparse-matrix assembly container, VTK post-processing
examples/            the 5 runnable examples described above
doc/dictionary.md     glossary of the variable-naming conventions used throughout
```

## Acknowledgments

This package started as `cardillo2`, a research FEM/IGA code co-developed at the
Institute of Nonlinear Mechanics, University of Stuttgart, with Jonas Harsch, Giuseppe
Capobianco, and Simon Eugster. This repository is my own, heavily trimmed personal fork
— curated down to the second-gradient/IGA continuum mechanics I worked on, cleaned up,
and modernized for a current Python/SciPy/meshio toolchain. It is not the original
project and does not track its history.

## License

MIT — see [LICENSE](LICENSE).
