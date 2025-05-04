import json
from typing import Final

class InputData:
    def __init__(self, file_path):
        """
        Reads input data from a JSON file-
        
        Parameters:
            file_path (str): Path to the input JSON file.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {file_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in input file.")

        # Extract and store data as final member variables
        try:
            geometry = data["geometry"]
            self.LENGTH: Final[float] = geometry["length"]
            self.WIDTH: Final[float] = geometry["width"]
        except KeyError as e:
            raise KeyError(f"Missing geometry information: {e}")

        if self.LENGTH <= 0 or self.WIDTH <= 0:
            raise ValueError("Geometry dimensions must be positive.")

        try:
            self.TENSOR: Final[list] = data["material"]["tensor"]
        except KeyError as e:
            raise KeyError(f"Missing material tensor: {e}")

        if not isinstance(self.TENSOR, list) or len(self.TENSOR) != 2 or len(self.TENSOR[0]) != 2:
            raise ValueError("Material tensor must be a 2x2 matrix.")

        try:
            self.BOUNDARY_CONDITION: Final[dict] = data["boundary_conditions"]
        except KeyError as e:
            raise KeyError(f"Missing boundary conditions: {e}")

        if "Dirichlet" not in self.BOUNDARY_CONDITION or "Neumann" not in self.BOUNDARY_CONDITION:
            raise KeyError("Boundary conditions must include 'Dirichlet' and 'Neumann'.")

        try:
            mesh = data["mesh"]
            self.CX: Final[int] = mesh["cx"]
            self.CY: Final[int] = mesh["cy"]
        except KeyError as e:
            raise KeyError(f"Missing mesh resolution information: {e}")

        if self.CX <= 0 or self.CY <= 0:
            raise ValueError("Mesh resolution (cx, cy) must be positive integers.")

# Example usage
if __name__ == "__main__":
    input_file = "example_input.json"
    try:
        input_data = InputData(input_file)
        print("Input data read successfully.")
        print(f"Length: {input_data.LENGTH}")
        print(f"Width: {input_data.WIDTH}")
        print(f"Tensor: {input_data.TENSOR}")
        print(f"Boundary Conditions: {input_data.BOUNDARY_CONDITION}")
        print(f"CX: {input_data.CX}")
        print(f"CY: {input_data.CY}")
    except Exception as e:
        print(f"Error: {e}")
