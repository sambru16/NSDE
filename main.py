from input import InputData
from exportRes import EXPORT
import numpy as np
from typing import Final

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

    # Example data for export (replace with actual simulation results)
    nodesPerEl = 4
    nElements = 2
    nNodes = 4
    dim = 2
    U = np.array([0.0, 0.1, 0.2, 0.3])  # Example results vector
    geom = np.array([[0, 0], [length, 0], [0, width], [length, width]])
    connec_plot = np.array([[0, 1, 3, 2]])
    n_dof_p_node = 1

    # Export results
    exporter = EXPORT(nodesPerEl, nElements, nNodes, dim, U, geom, connec_plot, n_dof_p_node)
    try:
        result_message = exporter.writeResults()
        print(result_message)
    except Exception as e:
        print(f"Error exporting results: {e}")

if __name__ == "__main__":
    main()
