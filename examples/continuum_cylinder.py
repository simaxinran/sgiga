"""First-gradient hyperelastic (Ogden) cylinder twisted about its axis.

B-spline (IGA) discretization fitted to a cylindrical reference shape, static
Newton solve.
"""
import pathlib
from sgiga.discretization.mesh3D import Mesh3D
from sgiga.discretization.B_spline import Knot_vector, fit_B_spline_volume
from sgiga.discretization.indexing import flat3D
from sgiga.model.continuum import Ogden1997_compressible, First_gradient
from sgiga.solver import Newton
from sgiga.model import Model
from sgiga.math.algebra import A_IK_basic_z
import numpy as np


def continuum_cylinder():
    # build mesh
    degrees = (3, 3, 3)
    QP_shape = (3, 3, 3)
    element_shape = (2, 2, 4)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)

    mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=1, basis="B-spline", nq_n=3)

    def cylinder(xi, eta, zeta, R=1, H=3):
        xi_ = 2 * xi - 1
        eta_ = 2 * eta - 1
        if np.abs(xi_) > np.abs(eta_):
            r = np.sqrt(1 + eta_**2)
        else:
            r = np.sqrt(1 + xi_**2)
        x = R / r * xi_
        y = R / r * eta_
        z = zeta * H
        return x, y, z

    nxi, neta, nzeta = 10, 10, 10
    xi = np.linspace(0, 1, num=nxi)
    eta = np.linspace(0, 1, num=neta)
    zeta = np.linspace(0, 1, num=nzeta)

    n3 = nxi * neta * nzeta
    knots = np.zeros((n3, 3))
    Pw = np.zeros((n3, 3))
    for i, xii in enumerate(xi):
        for j, etai in enumerate(eta):
            for k, zetai in enumerate(zeta):
                idx = flat3D(i, j, k, (nxi, neta, nzeta))
                knots[idx] = xii, etai, zetai
                Pw[idx] = cylinder(xii, etai, zetai)

    cDOF_ = np.array([], dtype=int)
    qc = np.array([], dtype=float).reshape((0, 3))
    X, Y, Z_ = fit_B_spline_volume(mesh, knots, Pw, qc, cDOF_)
    Z = np.concatenate((X, Y, Z_))

    # material model
    mu1 = 0.3
    mu2 = 0.5
    mat = Ogden1997_compressible(mu1, mu2)
    density = 1e-2

    # boundary conditions: clamp bottom face, twist top face about the z-axis
    cDOF1 = mesh.surface_qDOF[0].reshape(-1)
    cDOF2 = mesh.surface_qDOF[1].reshape(-1)
    cDOF = np.concatenate((cDOF1, cDOF2))
    b1 = lambda t: Z[cDOF1]

    def b2(t, phi0=np.pi, h=0.25):
        cDOF2_xyz = cDOF2.reshape(3, -1).T
        out = np.zeros_like(Z)
        phi = t * phi0
        R = A_IK_basic_z(phi)
        th = t * np.array([0, 0, h])
        for DOF in cDOF2_xyz:
            out[DOF] = R @ Z[DOF] + th
        return out[cDOF2]

    b = lambda t: np.concatenate((b1(t), b2(t)))

    continuum = First_gradient(density, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    model = Model()
    model.add(continuum)
    model.assemble()

    n_load_steps = 10
    tol = 1.0e-5
    max_iter = 10
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)
    sol = solver.solve()

    file_name = pathlib.Path(__file__).stem
    file_path = pathlib.Path(__file__).parent / "results" / file_name / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    continuum.post_processing(sol.t, sol.q, file_path)


if __name__ == "__main__":
    continuum_cylinder()
