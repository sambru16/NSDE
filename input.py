from typing import Final
import InputSettings
import numpy as np

class InputData:
    def __init__(self):
        """
        Validates input data from InputSettings.py constants.
        """
        # Geometry
        self.LENGTH: Final[float] = InputSettings.LENGTH
        self.WIDTH: Final[float] = InputSettings.WIDTH
        if self.LENGTH <= 0 or self.WIDTH <= 0:
            raise ValueError("Geometry dimensions must be positive.")

        # Mesh
        self.CX: Final[int] = InputSettings.CX
        self.CY: Final[int] = InputSettings.CY
        if self.CX <= 0 or self.CY <= 0:
            raise ValueError("Mesh resolution (cx, cy) must be positive integers.")

        # Material tensor
        self.TENSOR: Final[list] = InputSettings.TENSOR
        if not isinstance(self.TENSOR, list) or len(self.TENSOR) < 2 or len(self.TENSOR[0]) < 2:
            raise ValueError("Material tensor must be at least a 2x2 matrix.")
        
        # Boundary conditions (Dirichlet)
        self.DIRICHLET_BOUNDARY_CONDITIONS = {}
        for side in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
            found = False
            for bc in InputSettings.DIRICHLET_BC:
                if side.lower() in bc:
                    self.DIRICHLET_BOUNDARY_CONDITIONS[side] = bc[side.lower()]
                    found = True
                    break
            if not found:
                continue
        
        # Boundary conditions (Neumann)
        self.NEUMANN_BOUNDARY_CONDITIONS = {}
        for side in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
            found = False
            for bc in InputSettings.NEUMANN_BC:
                if side.lower() in bc:
                    self.NEUMANN_BOUNDARY_CONDITIONS[side] = bc[side.lower()]
                    found = True
                    break
            if not found:
                continue
        # Collisions in Neumann conditions
        # TODO: Look if they are the same
        if "LEFT" in self.NEUMANN_BOUNDARY_CONDITIONS and "TOP" in self.NEUMANN_BOUNDARY_CONDITIONS:
            print("Neumann conditions collide at top-left corner. Arithmetic middle will be used.")
        if "RIGHT" in self.NEUMANN_BOUNDARY_CONDITIONS and "BOTTOM" in self.NEUMANN_BOUNDARY_CONDITIONS:
            print("Neumann conditions collide at bottom-right corner. Arithmetic middle will be used.")
        if "LEFT" in self.NEUMANN_BOUNDARY_CONDITIONS and "BOTTOM" in self.NEUMANN_BOUNDARY_CONDITIONS:
            print("Neumann conditions collide at bottom-left corner. Arithmetic middle will be used.")
        if "RIGHT" in self.NEUMANN_BOUNDARY_CONDITIONS and "TOP" in self.NEUMANN_BOUNDARY_CONDITIONS:
            print("Neumann conditions collide at top-right corner. Arithmetic middle will be used.")

        # Inside boundary conditions
        self.INSIDE_BOUNDARY_CONDITIONS = InputSettings.INSIDE_BC
        


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
        print(f"Boundary Conditions: {input_data.DIRICHLET_BOUNDARY_CONDITIONS}")
    except Exception as e:
        print(f"Error: {e}")