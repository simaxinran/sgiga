"""Incompressible first-gradient hyperelastic (Ogden) cube under traction.

Classical Lagrange discretization (not B-spline/IGA), mixed displacement/
pressure formulation via an incompressibility constraint, static Newton solve.
"""
import pathlib
from sgiga.discretization.lagrange import Node_vector
from sgiga.discretization.mesh3D import Mesh3D, cube
from sgiga.model.continuum import Ogden1997_incompressible, First_gradient
from sgiga.solver import Newton
from sgiga.model import Model
from sgiga.model.force_distr2D import Force_distr2D
from sgiga.model.bilateral_constraints.implicit.incompressibility import (
    Incompressibility,
)
import numpy as np


def continuum_lagrange_cube():
    # build mesh
    degrees = (1, 1, 1)
    QP_shape = (2, 2, 2)
    element_shape = (2, 2, 2)

    Xi = Node_vector(degrees[0], element_shape[0])
    Eta = Node_vector(degrees[1], element_shape[1])
    Zeta = Node_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)

    mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=1, basis="lagrange", nq_n=3)

    # reference configuration is a unit cube
    cube_shape = (1, 1, 1)
    Z = cube(cube_shape, mesh, Fuzz=False)

    # incompressible material model, with a coarser pressure field
    mu1 = 0.3
    mat = Ogden1997_incompressible(mu1)
    Xi_la = Node_vector(degrees[0] - 1, element_shape[0])
    Eta_la = Node_vector(degrees[1] - 1, element_shape[1])
    Zeta_la = Node_vector(degrees[2] - 1, element_shape[2])
    knot_vectors_la = (Xi_la, Eta_la, Zeta_la)
    la_mesh = Mesh3D(knot_vectors_la, QP_shape, derivative_order=0, basis="lagrange", nq_n=1)

    density = 1e-2

    # boundary conditions: clamp bottom face, apply traction on top face
    cDOF = mesh.surface_qDOF[4].ravel()
    b = lambda t: Z[cDOF]

    continuum = First_gradient(density, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    model = Model()
    model.add(continuum)
    model.add(Incompressibility(continuum, la_mesh))

    F = lambda t, xi, eta: np.array([0, 0, 0.2]) * t
    model.add(Force_distr2D(F, continuum, 5))

    model.assemble()

    n_load_steps = 10
    tol = 1.0e-5
    max_iter = 20
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)
    sol = solver.solve()

    file_name = pathlib.Path(__file__).stem
    file_path = pathlib.Path(__file__).parent / "results" / file_name / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    continuum.post_processing(sol.t, sol.q, file_path)


if __name__ == "__main__":
    continuum_lagrange_cube()
