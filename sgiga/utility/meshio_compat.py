"""Compatibility shim for meshio >= 5.

meshio 5.0 dropped the VTK_BEZIER_* entries from its `topological_dimension`
lookup table (used by `meshio.CellBlock`/`meshio.Mesh`), while keeping the
underlying VTK type-ID table (`meshio._vtk_common.meshio_to_vtk_type`) and the
.vtu writer itself fully generic and untouched. That means Bezier-cell export
(used for our B-spline/IGA meshes) only needs these 7 dimension entries
restored to work correctly again.
"""
from meshio._mesh import topological_dimension

topological_dimension.update(
    {
        "VTK_BEZIER_CURVE": 1,
        "VTK_BEZIER_TRIANGLE": 2,
        "VTK_BEZIER_QUADRILATERAL": 2,
        "VTK_BEZIER_TETRAHEDRON": 3,
        "VTK_BEZIER_HEXAHEDRON": 3,
        "VTK_BEZIER_WEDGE": 3,
        "VTK_BEZIER_PYRAMID": 3,
    }
)
