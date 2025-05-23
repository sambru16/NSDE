from typing import Final

# Diameter of the system
LENGTH: Final[float] = 10.0
WIDTH: Final[float] = 5.0
# Number of intervals in the x and y directions
CX: Final[int] = 10
CY: Final[int] = 5

TENSOR: Final[list[list[float]]] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
]

# Boundary conditions
# Dirichlet boundary conditions
DIRICHLET_BC: Final[list[dict]] = [
    {"right": lambda x: 3.0-x},
    {"left": lambda x: 5.0},
    {"top": lambda y: 8.0},
    {"bottom": lambda y: 0.0}
]

# Neumann boundary conditions
NEUMANN_BC: Final[list[dict]] = [
    {"right": 0.0},
    {"left": 2.0},
    {"top": 1.0},
    {"bottom": 0.0}
]

# Inside boundary conditions
INSIDE_BC: Final[list[dict]] = [
    {
        "type": "Dirichlet",
        "start": [[2.0, 1.0], [2.0, 4.0]],
        "end": [[8.0, 1.0], [8.0, 4.0]],
        "value": lambda x, y: 0.0
    }
]

