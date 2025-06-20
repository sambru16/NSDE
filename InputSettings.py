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
    {"right": lambda y: -3},
    {"left": lambda y: 2.0},
    {"top": lambda x: 0.0},
    {"bottom": lambda x: 0.0}
]

# Neumann boundary conditions
# Usage: {"side": value} 
NEUMANN_BC: Final[list[dict]] = [
    {"right": lambda y: -1.0},
    {"left": lambda y: 2.0},
    {"top": lambda x: 1.0},
    {"bottom": lambda x: 1.0}
]

# Inside boundary conditions
# type: Dirichlet, Neumann, or None
INSIDE_BC: Final[list[dict]] = [
    {
        "x_range": [0.0, 90.0],
        "y": lambda x: 25.0 + 0.0005 * (x - 50.0) ** 3,
        "value": lambda x, y: 4.0
    }
]

# Gauss order
ORDER: Final[int] = 2