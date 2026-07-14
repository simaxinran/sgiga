"""Pantographic ortho-block under uniaxial tension.

Second-gradient continuum (beam-network) model of an orthotropic pantographic
block, clamped at the bottom face and pulled in tension at the top face via a
strong (Dirichlet elimination) boundary condition.

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
from sgiga.discretization.B_spline import Knot_vector
from sgiga.model.continuum import Second_gradient, Pantobox_beam_network
from sgiga.solver import Newton
from sgiga.model import Model


def save_solution(sol, filename):
    with open(filename, mode="wb") as f:
        pickle.dump(sol, f)


def ortho_block_tension():
    # build mesh
    degrees = (3, 3, 3)
    QP_shape = (3, 3, 3)
    element_shape = (2, 2, 6)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)

    mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=2, basis="B-spline", nq_n=3)

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

    # boundary conditions: clamp bottom face (z=0), pull top face (z=Lz) in
    # tension along z while keeping it flat in x/y
    #    6-------7
    #   /|      /|   |z /y
    #  / |     / |   | /
    # 4--|----5  |   |/---x
    # |  2----|--3
    # | /     | /
    # 0-------1
    # surface4 = [0, 1, 2, 3] => bottom, z = 0
    # surface5 = [4, 5, 6, 7] => top, z = z_max
    cDOF_bottom = mesh.surface_qDOF[4].ravel()
    cDOF_top_x = mesh.surface_qDOF[5][0]
    cDOF_top_y = mesh.surface_qDOF[5][1]
    cDOF_top_z = mesh.surface_qDOF[5][2]
    cDOF_fixed = np.concatenate((cDOF_bottom, cDOF_top_x, cDOF_top_y))
    cDOF = np.concatenate((cDOF_fixed, cDOF_top_z))

    b_fixed = lambda t: Z[cDOF_fixed]
    b_tension = lambda t: Z[cDOF_top_z] + t * 50.0 * u_l
    b = lambda t: np.concatenate((b_fixed(t), b_tension(t)))

    continuum = Second_gradient(density, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    model = Model()
    model.add(continuum)
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
    ortho_block_tension()
