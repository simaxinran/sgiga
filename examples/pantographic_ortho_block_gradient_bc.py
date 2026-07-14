"""Pantographic ortho-block deformed by a pure gradient boundary condition.

Both end faces are fully clamped at *zero* displacement (u = 0 everywhere on
them -- they never physically move). The only thing driving deformation is a
prescribed, time-varying target for the out-of-plane displacement gradient
d(u_z)/dZ on the top face (the bottom face keeps its gradient target at zero,
acting as the anchor). This is a boundary condition that only exists in
second-gradient theory: since the boundary displacement is identically zero
everywhere, a classical (first-gradient) continuum under this same boundary
condition would have the trivial zero solution -- there is nothing for it to
respond to. Here the block visibly deforms anyway, driven purely by the
second-gradient boundary term.

Reference:
    M. Stilz, S. R. Eugster, J. Harsch, F. Gutmann, G. Ganzenmueller, S. Hiermaier,
    "A second-gradient elasticity model and isogeometric analysis for the
    pantographic ortho-block," International Journal of Solids and Structures,
    vol. 280, 112358, 2023. https://doi.org/10.1016/j.ijsolstr.2023.112358
"""
import numpy as np
import pathlib
import pickle
from sgiga.discretization.mesh3D import Mesh3D, cube
from sgiga.discretization.mesh2D import Mesh2D
from sgiga.discretization.B_spline import Knot_vector
from sgiga.model.continuum import Second_gradient, Pantobox_beam_network
from sgiga.solver import Newton
from sgiga.model import Model
from sgiga.model.bilateral_constraints.implicit import (
    Displacement_constraint,
    Gradient_constraint,
)


def save_solution(sol, filename):
    with open(filename, mode="wb") as f:
        pickle.dump(sol, f)


def ortho_block_gradient_bc():
    # build mesh
    degrees = (3, 3, 3)
    QP_shape = (3, 3, 3)
    element_shape = (2, 2, 6)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)

    mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=2, basis="B-spline", nq_n=3)
    # Gradient_constraint needs per-surface boundary-element indices and shape
    # functions, which Mesh3D doesn't build by default (these two calls are
    # commented out in Mesh3D.__init__ since most examples don't need them)
    mesh.boundary_elements()
    mesh.shape_functions_boundary()

    # material parameters
    u_l = 1.0  # 1e-3
    u_Pa = 1.0  # 1e9
    Lx = 70.0 * u_l  # block length in x direction in mm
    Ly = 70.0 * u_l  # block length in y direction in mm
    Lz = 210.0 * u_l  # block length in z direction in mm
    nx = 3  # number of unit cells in x-direction
    nxp = 1  # number of sheets per length Lx including boundary
    p = Lx / np.sqrt(2) / nx  # distance between pivots along a beam
    a = 1.0 * p / 5 * u_l  # beam thickness in d2 direction in mm
    b = 1.0 * p / 5 * u_l  # beam thickness in d3 direction in mm
    Yb = 50.0 * u_Pa  # in GPa
    Gb = Yb / (2 + 0.8)
    rp = 0.45 * u_l  # pivot radius in mm
    hp = 1.5 * u_l  # pivot length in mm
    Jn = a**3 * b / 12  # second moment of area I_d2
    Jg = a * b**3 / 12  # second moment of area I_d3
    Jt = 2.25 * (a / 2) ** 4  # torsional moment of area I_d1
    Ke = Yb * a * b / p**2 * np.sqrt(2) * 2 * nxp  # extensional stiffness
    Kg = Yb * Jg / p**2 * np.sqrt(2) * 2 * nxp  # geodesic bending stiffness
    Kn = Yb * Jn / p**2 * np.sqrt(2) * 2 * nxp  # normal bending stiffness
    Kt = Gb * Jt / p**2 * np.sqrt(2) * 2 * nxp  # torsional stiffness
    Kp = Gb * np.pi * rp**4 / 2 / hp / p**3 * np.sqrt(2) * 2 * nxp
    Ks = Kp
    Kc = Gb * np.pi * rp**4 / 2 / hp / p**3 * np.sqrt(2) * 2 * nxp

    # reference configuration is a cube
    cube_shape = (Lx, Ly, Lz)
    Z = cube(cube_shape, mesh, Greville=True)

    mat = Pantobox_beam_network(
        Ke, Ks * 0.0, Kg, Kn, Kt, Kc * 0.0, numerical_derivative=False
    )
    density = 1e-3

    # no strong (Dirichlet elimination) boundary conditions here -- every
    # constraint below is applied weakly via a Lagrange multiplier field
    continuum = Second_gradient(density, mat, mesh, Z, z0=Z)

    # Lagrange-multiplier mesh living on the bottom/top (z=const) faces,
    # parametrized by the same (Xi, Eta) as those faces
    la_mesh_z = Mesh2D((Xi, Eta), QP_shape[1:], derivative_order=0, nq_n=1)

    # prescribed target for d(u_z)/dZ on the top face -- the sole driver of
    # deformation, since every displacement below is fixed at zero
    dg_top = lambda t, qe: t * 0.3

    #    6-------7
    #   /|      /|   |z /y
    #  / |     / |   | /
    # 4--|----5  |   |/---x
    # |  2----|--3
    # | /     | /
    # 0-------1
    # surface4 = [0, 1, 2, 3] => bottom, z = 0
    # surface5 = [4, 5, 6, 7] => top, z = z_max
    displacement_constraints = [
        # both end faces fully clamped at zero displacement -- they never move
        Displacement_constraint(continuum, la_mesh_z, srf_id=4, x=0),
        Displacement_constraint(continuum, la_mesh_z, srf_id=4, x=1),
        Displacement_constraint(continuum, la_mesh_z, srf_id=4, x=2),
        Displacement_constraint(continuum, la_mesh_z, srf_id=5, x=0),
        Displacement_constraint(continuum, la_mesh_z, srf_id=5, x=1),
        Displacement_constraint(continuum, la_mesh_z, srf_id=5, x=2),
    ]
    gradient_constraints = [
        # bottom face: gradient target stays at 0 (flat/unchanged) -- anchor
        Gradient_constraint(continuum, la_mesh_z, con_id=4, x=2, _X=2),
        # top face: only the out-of-plane gradient d(u_z)/dZ is driven to a
        # nonzero target -- this alone is what deforms the block
        Gradient_constraint(continuum, la_mesh_z, con_id=5, x=2, _X=2, disp=dg_top),
    ]

    model = Model()
    model.add(continuum)
    for constraint in displacement_constraints:
        model.add(constraint)
    for constraint in gradient_constraints:
        model.add(constraint)
    model.assemble()

    n_load_steps = 10
    tol = 1.0e-6
    max_iter = 20
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)

    file_name = pathlib.Path(__file__).stem
    file_path = pathlib.Path(__file__).parent / "results" / file_name / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    export_path = file_path.parent / "sol"

    sol = solver.solve()
    sol.z = np.array([continuum.z(t, sol.q[i]) for i, t in enumerate(sol.t)])
    save_solution(sol, str(export_path))

    continuum.post_processing(sol.t, sol.q, file_path, binary=True)


if __name__ == "__main__":
    ortho_block_gradient_bc()
