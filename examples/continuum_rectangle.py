"""First-gradient hyperelastic (Ogden) 2D rectangle under prescribed edge displacement.

B-spline (IGA) discretization, static Newton solve. Compare with
continuum_lagrange_rectangle.py, which solves the same problem with a
classical Lagrange discretization.
"""
import pathlib
from sgiga.discretization.mesh2D import Mesh2D, rectangle
from sgiga.discretization.B_spline import Knot_vector
from sgiga.model.continuum import Ogden1997_compressible, First_gradient
from sgiga.solver import Newton
from sgiga.model import Model
import numpy as np


def continuum_rectangle():
    # build mesh
    degrees = (3, 3)
    QP_shape = (3, 3)
    element_shape = (4, 4)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    knot_vectors = (Xi, Eta)

    mesh = Mesh2D(knot_vectors, QP_shape, derivative_order=1, basis="B-spline", nq_n=2)

    # reference configuration is a rectangle
    rectangle_shape = (4, 2)
    Z = rectangle(rectangle_shape, mesh, Greville=True)

    # material model
    mu1 = 0.3
    mu2 = 0.5
    mat = Ogden1997_compressible(mu1, mu2, dim=2)

    # boundary conditions: clamp left edge, pull right edge in tension
    cDOF1 = mesh.edge_qDOF[0].reshape(-1)
    cDOF2 = mesh.edge_qDOF[1][0]
    cDOF = np.concatenate((cDOF1, cDOF2))
    b1 = lambda t: Z[cDOF1]
    b2 = lambda t: Z[cDOF2] + t * 1
    b = lambda t: np.concatenate((b1(t), b2(t)))

    continuum = First_gradient(1, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    model = Model()
    model.add(continuum)
    model.assemble()

    n_load_steps = 10
    tol = 1.0e-6
    max_iter = 10
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)
    sol = solver.solve()

    file_name = pathlib.Path(__file__).stem
    file_path = pathlib.Path(__file__).parent / "results" / file_name / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    continuum.post_processing(sol.t, sol.q, file_path)


if __name__ == "__main__":
    continuum_rectangle()
