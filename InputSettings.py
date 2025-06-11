from typing import Final

# Diameter of the system
LENGTH: Final[float] = 100.0
WIDTH: Final[float] = 50.0
# Number of intervals in the x and y directions
CX: Final[int] = 100
CY: Final[int] = 50

#Material tensor
TENSOR: Final[list[list[float]]] = [
    [1.0, 0.0],
    [0.0, 1.0]
]

# Boundary conditions
# Dirichlet boundary conditions
# Usage: {"side": lambda x: value}
DIRICHLET_BC: Final[list[dict]] = [
    {"right": lambda x: -3},
    {"left": lambda x: 2.0},
    {"top": lambda y: 0.0},
    {"bottom": lambda y: 0.0}
]

# Inside boundary conditions
# type: Dirichlet, Neumann, or None
INSIDE_BC: Final[list[dict]] = [
    {
        "type": "None",
        "x_range": [0.0, 90.0],
        "y": lambda x: x,
        "value": lambda x, y: 4.0
    }
]
