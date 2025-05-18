from input import InputData
from mesh import QuadMesh
from exportRes import EXPORT
import numpy as np
from typing import Final
from materialModel import MaterialModel
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def assemble_system(mesh):
    """
    Moved out of materialModel.py
    May be useful, nut does not belong here in my opinion
    """
    num_nodes = len(mesh.get_nodes())
    stiffness_matrix = lil_matrix((num_nodes, num_nodes))
    load_vector = np.zeros(num_nodes)

    # Placeholder for element matrix assembly
    for element in mesh.get_elements():
        element_nodes = mesh.get_nodes()[element]
        ke = np.zeros((4, 4))  # Replace with actual computation
        fe = np.zeros(4)       # Replace with actual computation
        for i, ni in enumerate(element):
            for j, nj in enumerate(element):
                stiffness_matrix[ni, nj] += ke[i, j]
            load_vector[ni] += fe[i]

    return stiffness_matrix, load_vector

def main():
    # Read input data
    input_file = "example_input.json"
    try:
        input_data: Final[InputData] = InputData(input_file)
    except Exception as e:
        print(f"Error reading input: {e}")
        return

    # Extract data from input
    length = input_data.LENGTH
    width = input_data.WIDTH
    tensor = input_data.TENSOR
    boundary_conditions = input_data.BOUNDARY_CONDITION
    print(f"Length: {length}, Width: {width}, Tensor: {tensor}")

    # Generate mesh
    mesh = QuadMesh(length, width, input_data.CX, input_data.CY)

    # Initialize material model
    material_model = MaterialModel(tensor)

    # TODO: implement actual stiffness matrix and load vector
    stiffness_matrix, load_vector = assemble_system(mesh, material_model.material_tensor, boundary_conditions)

    # Solve the system
    U = spsolve(stiffness_matrix.tocsr(), load_vector)
    print("Solution vector:", U)

    # Export results
    geom = mesh.get_nodes()
    connec_plot = mesh.get_elements()
    exporter = EXPORT(4, len(connec_plot), len(geom), 2, U, geom, connec_plot, 1)
    try:
        result_message = exporter.writeResults()
        print(result_message)
    except Exception as e:
        print(f"Error exporting results: {e}")

if __name__ == "__main__":
    main()
