"""First-gradient hyperelastic (Ogden) cube under prescribed face displacement.

B-spline (IGA) discretization, static Newton solve.
"""
import pathlib
import pickle
from sgiga.discretization.mesh3D import Mesh3D, cube
from sgiga.discretization.B_spline import Knot_vector
from sgiga.model.continuum import Ogden1997_compressible, First_gradient
from sgiga.solver import Newton
from sgiga.model import Model
import numpy as np


def save_solution(sol, filename):
    with open(filename, mode="wb") as f:
        pickle.dump(sol, f)


def continuum_cube():
    # build mesh
    degrees = (3, 3, 3)
    QP_shape = (3, 3, 3)
    element_shape = (2, 2, 2)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)

    mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=2, basis="B-spline", nq_n=3)

    # reference configuration is a unit cube
    cube_shape = (1, 1, 1)
    Z = cube(cube_shape, mesh, Greville=False)

    # material model
    mu1 = 0.3
    mu2 = 0.5
    mat = Ogden1997_compressible(mu1, mu2)
    density = 1e-2

    # boundary conditions: clamp bottom face, pull top face in tension
    cDOF1 = mesh.surface_qDOF[4].ravel()
    cDOF2 = mesh.surface_qDOF[5][2]
    cDOF = np.concatenate((cDOF1, cDOF2))
    b1 = lambda t: Z[cDOF1]
    b2 = lambda t: Z[cDOF2] + t * 1.0
    b = lambda t: np.concatenate((b1(t), b2(t)))

    continuum = First_gradient(density, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    model = Model()
    model.add(continuum)
    model.assemble()

    n_load_steps = 10
    tol = 1.0e-5
    max_iter = 10
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)

    file_name = pathlib.Path(__file__).stem
    file_path = pathlib.Path(__file__).parent / "results" / file_name / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    export_path = file_path.parent / "sol"

    sol = solver.solve()
    save_solution(sol, str(export_path))

    continuum.post_processing(sol.t, sol.q, file_path)


if __name__ == "__main__":
    continuum_cube()
