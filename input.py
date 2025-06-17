from typing import Final
import InputSettings as InputSettings
import numpy as np
from boundaryConditions import BoundaryCondition

class InputData:
    def __init__(self, settings: dict = None):
        """
        Validates input data from InputSettings.py constants or from a provided dictionary.
        """
        if settings is not None:
            # Use only the provided dictionary
            self.LENGTH: Final[float] = settings["LENGTH"]
            self.WIDTH: Final[float] = settings["WIDTH"]
            if self.LENGTH <= 0 or self.WIDTH <= 0:
                raise ValueError("Geometry dimensions must be positive.")

            self.CX: Final[int] = settings["CX"]
            self.CY: Final[int] = settings["CY"]
            if self.CX <= 0 or self.CY <= 0:
                raise ValueError("Mesh resolution (cx, cy) must be positive integers.")

            self.TENSOR: Final[list] = settings["TENSOR"]
            if not isinstance(self.TENSOR, list) or len(self.TENSOR) < 2 or len(self.TENSOR[0]) < 2:
                raise ValueError("Material tensor must be a 2x2 matrix.")
            tensor_array = np.array(self.TENSOR)
            if tensor_array.shape != (2, 2):
                raise ValueError("Material tensor must be a 2x2 matrix.")
            if not np.all(np.linalg.eigvals(tensor_array) > 0):
                raise ValueError("Material tensor must be positive definite.")

            dirichlet_bc = settings.get("DIRICHLET_BC", [])
            neumann_bc = settings.get("NEUMANN_BC", [])
            inside_bc = settings.get("INSIDE_BC", [])
        else:
            # Use only the InputSettings module
            self.LENGTH: Final[float] = InputSettings.LENGTH
            self.WIDTH: Final[float] = InputSettings.WIDTH
            if self.LENGTH <= 0 or self.WIDTH <= 0:
                raise ValueError("Geometry dimensions must be positive.")

            self.CX: Final[int] = InputSettings.CX
            self.CY: Final[int] = InputSettings.CY
            if self.CX <= 0 or self.CY <= 0:
                raise ValueError("Mesh resolution (cx, cy) must be positive integers.")

            self.TENSOR: Final[list] = InputSettings.TENSOR
            if not isinstance(self.TENSOR, list) or len(self.TENSOR) < 2 or len(self.TENSOR[0]) < 2:
                raise ValueError("Material tensor must be a 2x2 matrix.")
            tensor_array = np.array(self.TENSOR)
            if not np.all(np.linalg.eigvals(tensor_array) > 0):
                raise ValueError("Material tensor must be positive definite.")

            dirichlet_bc = getattr(InputSettings, "DIRICHLET_BC", [])
            neumann_bc = getattr(InputSettings, "NEUMANN_BC", [])
            inside_bc = getattr(InputSettings, "INSIDE_BC", [])

        self.boundary = BoundaryCondition(
            dirichlet_bc, neumann_bc, inside_bc,
            [self.LENGTH / self.CX, self.WIDTH / self.CY])

# Example usage
if __name__ == "__main__":
    try:
        input_data = InputData()
        print("Input data validated successfully.")
        print(f"Length: {input_data.LENGTH}")
        print(f"Width: {input_data.WIDTH}")
        print(f"CX: {input_data.CX}")
        print(f"CY: {input_data.CY}")
        print(f"Tensor: {input_data.TENSOR}")
        print(f"Boundary Conditions: {input_data.boundary}")
    except Exception as e:
        print(f"Error: {e}")