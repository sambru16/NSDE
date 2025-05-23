from input import InputData
from mesh import QuadMesh
from exportRes import EXPORT
import numpy as np
from typing import Final
from materialModel import MaterialModel
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from assembler import assembleSystem
from boundaryConditions import applyBoundaryConditions

def main():
    # Read input data
    try:
        input_data: Final[InputData] = InputData()
    except Exception as e:
        print(f"Error reading input: {e}")
        return

    # Extract data from input
    length = input_data.LENGTH
    width = input_data.WIDTH
    tensor = input_data.TENSOR
    boundary_conditions = input_data.DIRICHLET_BOUNDARY_CONDITIONS
    print(f"Length: {length}, Width: {width}, Tensor: {tensor}")

    # Generate mesh
    mesh = QuadMesh(length, width, input_data.CX, input_data.CY)

    # Initialize material model
    material_model = MaterialModel(tensor)

    # returns the stiffness matrix and load vector
    stiffness_matrix, load_vector = assembleSystem(mesh, material_model)
    # apply boundary conditions
    applyBoundaryConditions(mesh, boundary_conditions, stiffness_matrix, load_vector)
    #print("Stiffness matrix:", stiffness_matrix)
    #print("Load vector:", load_vector)


    # Solve the system
    U = spsolve(stiffness_matrix.tocsr(), load_vector)
    print("Solution vector:", U)

    # Export results
    geom = mesh.get_nodes()
    connec_plot = mesh.get_elements()
    exporter = EXPORT(4, len(connec_plot), len(geom), 2, U, geom, connec_plot, 1)
    try:
        print(exporter.writeResults())
    except Exception as e:
        print(f"Error exporting results: {e}")

if __name__ == "__main__":
    main()
