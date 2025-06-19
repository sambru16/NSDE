from scipy.sparse import lil_matrix
import numpy as np
from GaussianQuadrature import GaussianQuadrature
from mesh import QuadMesh
from materialModel import MaterialModel

def assembleSystem(mesh: QuadMesh, material_model: MaterialModel, gauss: GaussianQuadrature):
    """
    Assemble the global stiffness matrix for the FEM system.
    """
    num_nodes = len(mesh.get_nodes())
    stiffness_matrix = lil_matrix((num_nodes, num_nodes))

    for element in mesh.get_elements():
        element_nodes = mesh.get_nodes()[element]

        # Local stiffness matrix
        ke = gauss.localStiffnessMatrix(material_model.material_tensor, element_nodes)
        
        for i, ni in enumerate(element):
            for j, nj in enumerate(element):
                stiffness_matrix[ni, nj] += ke[i, j]

    return stiffness_matrix