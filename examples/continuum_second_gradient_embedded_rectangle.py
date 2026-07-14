"""Pantographic sheet embedded in 3D space, under combined tension/torsion.

Uses Pantosheet_beam_network, which the original authors flagged
`# TODO: untested!` in sgiga/model/continuum/second_gradient_material.py.

KNOWN ISSUE (partially fixed): originally produced NaN from the first Newton
iteration. Root cause found and fixed: the mesh was built with `basis="lagrange"`
but used B-spline `Knot_vector` objects + `rectangle(..., Greville=True)` (a
B-spline-specific point placement), producing degenerate near-zero-area
elements in the *reference* configuration. Now uses `Node_vector` +
`Greville=False`, matching every other Lagrange example in this repo, and no
longer produces NaN.

However, the Newton solve still fails to converge (residual grows/oscillates
starting at load step 2), independent of load-step size or the y-displacement
magnitude -- ruled out as a step-size or large-deformation instability. Also
verified independently (via numerical differentiation, see git history/PR
discussion) that both Displacement_constraint's analytic Jacobian and
Second_gradient.f_pot_q_el's analytic tangent are individually correct for
this material, so the remaining bug (if any) is likely in the global
assembly of the augmented (K, W_g; W_g^T, 0) system for this specific
weakly-constrained edge scenario, or a genuine conditioning issue in the
never-validated Pantosheet_beam_network formulation itself. Left for future
debugging.
"""
import numpy as np
import pathlib
import pickle
from sgiga.discretization.mesh2D import Mesh2D, rectangle
from sgiga.discretization.mesh1D import Mesh1D
from sgiga.discretization.lagrange import Node_vector
from sgiga.model.continuum import Second_gradient, Pantosheet_beam_network
from sgiga.solver import Newton
from sgiga.model import Model
from sgiga.math.algebra import A_IK_basic_x
from sgiga.model.bilateral_constraints.implicit import Displacement_constraint


def save_solution(sol, filename):
    with open(filename, mode="wb") as f:
        pickle.dump(sol, f)


def continuum_second_gradient_embedded_rectangle():
    # build mesh
    degrees = (3, 3)
    QP_shape = (3, 3)
    element_shape = (3, 9)

    Xi = Node_vector(degrees[0], element_shape[0])
    Eta = Node_vector(degrees[1], element_shape[1])
    knot_vectors = (Xi, Eta)

    mesh = Mesh2D(knot_vectors, QP_shape, derivative_order=2, basis="lagrange", nq_n=3)

    # material parameters
    u_l = 1.0  # 1e-3
    u_Pa = 1.0  # 1e9
    Lx = 25.0 * u_l  # block length in x direction in mm
    Ly = 75.0 * u_l  # block length in y direction in mm
    a = 0.6 * u_l  # beam thickness in d2 direction in mm
    b = 0.6 * u_l  # beam thickness in d3 direction in mm
    Yb = 90.0 * u_Pa  # in GPa
    Gb = Yb / (2 + 0.6)
    rp = 0.3 * u_l  # pivot radius in mm
    hp = 1.25 * u_l  # pivot length in mm
    Jn = a**3 * b / 12  # second moment of area I_d2
    Jg = a * b**3 / 12  # second moment of area I_d3
    Jt = 2.25 * a**4  # torsional moment of area I_d1
    nx = 10  # number of unit cells in x-direction
    p = Lx / np.sqrt(2) / nx  # distance between pivots along a beam
    Ke = Yb * a * b / p  # extensional stiffness
    Kg = Yb * Jg / p  # geodesic bending stiffness
    Kn = Yb * Jn / p  # normal bending stiffness
    Kt = Gb * Jt / p  # torsional stiffness

    rectangle_shape = (Lx, Ly)
    Z = rectangle(rectangle_shape, mesh, Greville=False)

    mat = Pantosheet_beam_network(Ke, 0, Kg, Kn, Kt)

    # boundary conditions: strong displacement on one edge
    cDOF1 = mesh.edge_qDOF[2][0]
    cDOF2 = mesh.edge_qDOF[2][1]
    cDOF3 = mesh.edge_qDOF[2][2]
    cDOF = np.concatenate((cDOF1, cDOF2, cDOF3))
    b1 = lambda t: Z[cDOF1]
    b2 = lambda t: Z[cDOF2]
    b3 = lambda t: Z[cDOF3]
    b = lambda t: np.concatenate((b1(t), b2(t), b3(t)))

    # torsion (currently unused, kept for reference)
    def bt(t, phi0=0.5 * np.pi, h=-20):
        cDOF2_xyz = cDOF3.reshape(3, -1).T
        out = np.zeros_like(Z)
        phi = t * phi0
        R = A_IK_basic_x(phi)
        th = t * np.array([1, 0, 0]) * h
        for DOF in cDOF2_xyz:
            out[DOF] = R @ (Z[DOF] - [0, Ly / 2, 0]) + th + [0, Ly / 2, 0]
        return out[cDOF3]

    continuum = Second_gradient(1, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    # weak constraint on the opposite edge's displacement
    la_mesh_x = Mesh1D(Xi, 3, derivative_order=0, basis="lagrange", nq_n=1)
    disp = lambda t, q: 20 * t

    displacement_constraint_x = Displacement_constraint(
        continuum, la_mesh_x, edge_id=3, x=0, disp=0
    )
    displacement_constraint_y = Displacement_constraint(
        continuum, la_mesh_x, edge_id=3, x=1, disp=disp
    )

    model = Model()
    model.add(continuum)
    model.add(displacement_constraint_x)
    model.add(displacement_constraint_y)
    model.assemble()

    n_load_steps = 10
    tol = 1.0e-8
    max_iter = 20
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)

    file_name = pathlib.Path(__file__).stem
    file_path = pathlib.Path(__file__).parent / "results" / file_name / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    export_path = file_path.parent / "sol"

    sol = solver.solve()
    sol.z = np.array([continuum.z(t, sol.q[i]) for i, t in enumerate(sol.t)])
    save_solution(sol, str(export_path))

    continuum.post_processing(sol.t, sol.q, file_path)
    continuum.post_processing_la(
        sol.t,
        la_mesh_x,
        sol.la_g[:, model.contributions[-1].la_gDOF],
        sol.q,
        continuum.mesh.edge_qDOF[3].ravel(),
        file_path,
    )


if __name__ == "__main__":
    continuum_second_gradient_embedded_rectangle()  # known broken, see module docstring
