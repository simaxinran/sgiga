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
[Acknowledgments](#acknowledgments)), kept to the parts needed to run the 4 examples
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
each writes VTK/ParaView output to a gitignored `results/` folder.

```bash
python examples/continuum.py
```

| Script | What it demonstrates |
| --- | --- |
| `continuum.py` | First-gradient hyperelastic (Ogden) cube under prescribed face displacement, B-spline (IGA) discretization, static Newton solve. |
| `continuum_lagrange.py` | Same first-gradient hyperelastic formulation on a 2D rectangle, using a classical Lagrange discretization instead of B-splines, for comparison. |
| `continuum_second_gradient.py` | Second-gradient continuum (`Pantobox_beam_network` material) cube under tension — the microstructure-aware theory behind the pantographic/metamaterial examples below. |
| `pantographic_ortho_block.py` | 3D pantographic/orthotropic block under tension and torsion — a second-gradient beam-network metamaterial on a larger IGA mesh. See [Stilz et al. (2023)](https://doi.org/10.1016/j.ijsolstr.2023.112358). |

Output `.vtu`/`.pvd` files can be opened in [ParaView](https://www.paraview.org/) to
visualize the deformed configuration.

### Gradient boundary conditions

Second-gradient continua support constraining the displacement *gradient* on a
boundary, not just the displacement itself (needed e.g. to prescribe a flat/clamped
edge in the pantographic ortho-block). Both are implemented as Lagrange-multiplier
constraints in `sgiga/model/bilateral_constraints/implicit/displacement_gradient_constraint.py`:

- `Displacement_constraint` — standard Dirichlet condition on the displacement field.
- `Gradient_constraint` — constrains a component of the displacement gradient
  (∂u/∂X) on a boundary surface.

`examples/pantographic_ortho_block.py` wires both up through its
`weak_constraint_srf()` helper (inside `ortho_block_cuboid()`), which builds a shared
boundary `Mesh2D` for the Lagrange multipliers and instantiates either constraint type
per boundary surface depending on whether a gradient direction (`_X`) or a plain
displacement is passed in.

## Visuals

Renders and clips made in ParaView from the examples above:

![pantographic_ortho_block.py: gradient boundary condition](docs/videos/panto_ortho_block_gradient_boundary.gif)

*Pantographic ortho-block under a gradient boundary condition (see
[Gradient boundary conditions](#gradient-boundary-conditions) above). Full-quality
clip: [panto_ortho_block_gradient_boundary.mp4](docs/videos/panto_ortho_block_gradient_boundary.mp4).*

<!-- ![continuum.py: deformed cube](docs/images/continuum_cube.png) -->

## Project structure

```
sgiga/
  discretization/   B-spline, Lagrange, and mesh (1D/2D/3D) discretization
  math/              small linear-algebra and numerical-derivative utilities
  model/             continuum/material models, forces, constraints, rigid bodies
  solver/            Newton, Euler backward, generalized-alpha solvers
  utility/           sparse-matrix assembly container, VTK post-processing
examples/            the 4 runnable examples described above
docs/images/         screenshots/renders embedded above
docs/videos/         short clips embedded above (GitHub inlines short .mp4/.gif well)
doc/dictionary.md     glossary of the variable-naming conventions used throughout
```

## References

- M. Stilz, S. R. Eugster, J. Harsch, F. Gutmann, G. Ganzenmüller, S. Hiermaier,
  "A second-gradient elasticity model and isogeometric analysis for the pantographic
  ortho-block," *International Journal of Solids and Structures*, vol. 280, 112358,
  2023. [doi:10.1016/j.ijsolstr.2023.112358](https://doi.org/10.1016/j.ijsolstr.2023.112358)
  — theory and formulation behind `examples/pantographic_ortho_block.py`.

## Acknowledgments

This package started as `cardillo`, a research FEM/IGA code co-developed at the
Institute of Nonlinear Mechanics, University of Stuttgart, with Jonas Harsch, Giuseppe
Capobianco, and Simon Eugster. This repository is my own, heavily trimmed personal fork
— curated down to the second-gradient/IGA continuum mechanics I worked on, cleaned up,
and modernized for a current Python/SciPy/meshio toolchain. It is not the original
project and does not track its history.

## License

MIT — see [LICENSE](LICENSE).
