from input import InputData
from mesh import QuadMesh
from exportRes import EXPORT
from materialModel import MaterialModel
from scipy.sparse.linalg import spsolve
from assembler import assembleSystem
import matplotlib.pyplot as plt

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
    plt.show()

def main(input_data: InputData = None):
    # Extract data from input
    print(f"Length: {input_data.LENGTH}, Width: {input_data.WIDTH}, Tensor: {input_data.TENSOR}")

    # Generate mesh
    mesh = QuadMesh(input_data.LENGTH, input_data.WIDTH, input_data.CX, input_data.CY)

    # Initialize material model
    material_model = MaterialModel(input_data.TENSOR)

    # returns the stiffness matrix and load vector
    stiffness_matrix = assembleSystem(mesh, material_model, input_data.gauss)
    # calculate load vector with boundary conditions
    load_vector = input_data.boundary.apply(mesh, stiffness_matrix, input_data.gauss)
    # Solve the system
    U = spsolve(stiffness_matrix.tocsr(), load_vector)
    print("Solution vector:", U)

    # Export results
    geom = mesh.get_nodes()
    connec_plot = mesh.get_output_elements()
    exporter = EXPORT(4, len(connec_plot), len(geom), 2, U, geom, connec_plot, 1)
    try:
        print(exporter.writeResults())
    except Exception as e:
        print(f"Error exporting results: {e}")

    # Plot Results
    plot_quadmesh(mesh, U)


if __name__ == "__main__":
    main(InputData())
