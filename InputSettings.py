from typing import Final

# Diameter of the system
LENGTH: Final[float] = 100.0
WIDTH: Final[float] = 50.0
# Number of intervals in the x and y directions
CX: Final[int] = 100
CY: Final[int] = 50

TENSOR: Final[list[list[float]]] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
]

# Boundary conditions
# Dirichlet boundary conditions
DIRICHLET_BC: Final[list[dict]] = [
    {"right": lambda x: -3},
    {"left": lambda x: 0.0},
    {"top": lambda y: 0.0},
    {"bottom": lambda y: 0.0}
]

# Neumann boundary conditions
NEUMANN_BC: Final[list[dict]] = [
    {"right": -1.0},
    {"left": 2.0},
    {"top": 1.0},
    {"bottom": 1.0}
]

# Inside boundary conditions
INSIDE_BC: Final[list[dict]] = [
    {
        "type": "Dirichlet",
        "x_range": [0.0, 90.0],
        "y": lambda x: 25.0 + 0.0005 * (x - 50.0) ** 3,
        "value": lambda x, y: 4.0
    }
]

