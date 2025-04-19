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

        try:
            self.TENSOR: Final[list] = data["material"]["tensor"]
        except KeyError as e:
            raise KeyError(f"Missing material tensor: {e}")

        try:
            self.BOUNDARY_CONDITION: Final[dict] = data["boundary_conditions"]
        except KeyError as e:
            raise KeyError(f"Missing boundary conditions: {e}")

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
    except Exception as e:
        print(f"Error: {e}")
