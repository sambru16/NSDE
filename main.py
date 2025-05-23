from input import InputData
from mesh import QuadMesh
from exportRes import EXPORT
from typing import Final
from materialModel import MaterialModel
from scipy.sparse.linalg import spsolve
from assembler import assembleSystem
import matplotlib.pyplot as plt
from boundaryConditions import BoundaryCondition

def plot_quadmesh(mesh, U):
    """
    Plots the result of the FEM simulation using pcolormesh.
    Prints the value in every box (cell) of the plot.
    """
    cx, cy = mesh.cx, mesh.cy
    nodes = mesh.get_nodes()

    # Reshape the nodes and solution vector for plotting
    X = nodes[:, 0].reshape((cy + 1, cx + 1))
    Y = nodes[:, 1].reshape((cy + 1, cx + 1))
    Z = U.reshape((cy + 1, cx + 1))  # Reshape U to match the mesh grid

    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
    fig.colorbar(c, ax=ax)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('FEM-Results')
    plt.grid(True)
    plt.tight_layout()

    # Print the value in every box (cell)
    #for i in range(cy):
    #    for j in range(cx):
    #        # Compute the center of the cell
    #        x_center = 0.25 * (X[i, j] + X[i+1, j] + X[i, j+1] + X[i+1, j+1])
    #        y_center = 0.25 * (Y[i, j] + Y[i+1, j] + Y[i, j+1] + Y[i+1, j+1])
    #        # Get the values at the four corners of the cell
    #        z_vals = [Z[i, j], Z[i+1, j], Z[i, j+1], Z[i+1, j+1]]
    #        # Arithmetic mean for the cell value
    #        cell_val = sum(z_vals) / 4
    #        ax.text(x_center, y_center, f"{cell_val:.2f}", color='white', ha='center', va='center', fontsize=8, weight='bold')

    plt.show()

def main():
    # Read input data
    try:
        input_data: Final[InputData] = InputData()
    except Exception as e:
        print(f"Error reading input: {e}")
        return

    # Extract data from input
    print(f"Length: {input_data.LENGTH}, Width: {input_data.WIDTH}, Tensor: {input_data.TENSOR}")

    # Generate mesh
    mesh = QuadMesh(input_data.LENGTH, input_data.WIDTH, input_data.CX, input_data.CY)

    # Initialize material model
    material_model = MaterialModel(input_data.TENSOR)

    # returns the stiffness matrix and load vector
    stiffness_matrix, load_vector = assembleSystem(mesh, material_model)
    # apply boundary conditions
    input_data.boundary.apply(mesh, stiffness_matrix, load_vector)
    #applyBoundaryConditions(mesh, input_data.DIRICHLET_BOUNDARY_CONDITIONS, stiffness_matrix, load_vector)

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

    plot_quadmesh(mesh, U)


if __name__ == "__main__":
    main()
