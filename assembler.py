from scipy.sparse import lil_matrix
import numpy as np
from GaussianQuadrature import GaussianQuadrature

def assembleSystem(mesh, material_model):
    """
    Assemble the global stiffness matrix and load vector for the FEM system.
    """
    num_nodes = len(mesh.get_nodes())
    stiffness_matrix = lil_matrix((num_nodes, num_nodes))
    load_vector = np.zeros(num_nodes)

    gauss = GaussianQuadrature()

    for element in mesh.get_elements():
        element_nodes = mesh.get_nodes()[element]
        # Get conductivity tensor for this element from the material model
        conductivity = material_model.material_tensor

        # Use only the 2x2 part for 2D problems
        conductivity_2d = np.array(conductivity)[:2, :2]

        # Local stiffness matrix
        # TODO: function should be able to take 3x3 tensor
        ke = gauss.localStiffnessMatrix(conductivity_2d, element_nodes)

        # Local load vector (assuming body force/source term is zero unless specified)
        fe = gauss.loadVector(lambda x, y: 0.0, element_nodes)
        
        for i, ni in enumerate(element):
            for j, nj in enumerate(element):
                stiffness_matrix[ni, nj] += ke[i, j]
            load_vector[ni] += fe[i]

    return stiffness_matrix, load_vector