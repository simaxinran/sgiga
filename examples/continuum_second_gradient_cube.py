"""Second-gradient continuum (Pantobox_beam_network) cube under tension.

B-spline (IGA) discretization, static Newton solve -- the microstructure-aware
theory behind the pantographic/metamaterial examples.
"""
import pathlib
import pickle
from sgiga.discretization.mesh3D import Mesh3D, cube
from sgiga.discretization.B_spline import Knot_vector
from sgiga.model.continuum import Second_gradient, Pantobox_beam_network
from sgiga.solver import Newton
from sgiga.model import Model
import numpy as np


def save_solution(sol, filename):
    with open(filename, mode="wb") as f:
        pickle.dump(sol, f)


def continuum_second_gradient_cube():
    # build mesh
    degrees = (3, 3, 3)
    QP_shape = (3, 3, 3)
    element_shape = (2, 2, 6)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)

    mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=2, basis="B-spline", nq_n=3)

    # material model
    l = 70.0  # in mm
    L = 3 * l
    a = 1.0  # in mm
    b = 1.0  # in mm
    Yb = 50.0  # in GPa
    Gb = Yb / (2 + 0.8)
    rp = 0.45  # in mm
    hp = 1.5  # in mm
    Jn = a**3 * b / 12
    Jg = a * b**3 / 12
    Jt = 0.196 * a**3 * b
    nf = 1
    p = l / np.sqrt(2) / nf
    Ke = Yb * a * b / p
    Kg = Yb * Jg / p
    Kn = Yb * Jn / p
    Kt = Gb * Jt / p
    Kp = Gb * np.pi * rp**4 / 2 / hp / p**2
    Ks = Kp * 0
    Kc = Ks
    nsH = nf / l

    # reference configuration is a cube
    W = L / 3
    B = L / 3
    H = L
    cube_shape = (W, B, H)
    Z = cube(cube_shape, mesh, Greville=False)

    mat = Pantobox_beam_network(
        Ke * nsH, Ks * nsH, Kg * nsH, Kn * nsH, Kt * nsH, Kc * nsH
    )
    density = 1.0e-3

    # boundary conditions: clamp bottom face, pull top face in tension
    cDOF1 = mesh.surface_qDOF[4].ravel()
    cDOF3 = mesh.surface_qDOF[5][0]
    cDOF4 = mesh.surface_qDOF[5][1]
    cDOF2 = mesh.surface_qDOF[5][2]
    cDOF134 = np.concatenate((cDOF1, cDOF3, cDOF4))
    cDOF = np.concatenate((cDOF134, cDOF2))
    b1 = lambda t: Z[cDOF134]
    b2 = lambda t: Z[cDOF2] + t * 30.0
    b = lambda t: np.concatenate((b1(t), b2(t)))

    continuum = Second_gradient(density, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    model = Model()
    model.add(continuum)
    model.assemble()

    n_load_steps = 20
    tol = 1.0e-5
    max_iter = 10
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)

    file_name = pathlib.Path(__file__).stem
    file_path = pathlib.Path(__file__).parent / "results" / file_name / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    export_path = file_path.parent / "sol"

    sol = solver.solve()
    save_solution(sol, str(export_path))

    continuum.post_processing(sol.t, sol.q, file_path, binary=True)


if __name__ == "__main__":
    continuum_second_gradient_cube()
