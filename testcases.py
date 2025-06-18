from main import main
from input import InputData

def test():
    """
    Test cases for simulation with different parameters.
    """
    test_cases = [
        { # Source test
            "LENGTH": 100.0,
            "WIDTH": 100.0,
            "CX": 10,
            "CY": 10,
            "TENSOR": [
                [1.0, 0.0],
                [0.0, 1.0]
            ],
            "DIRICHLET_BC": [
                {"top": lambda y: 0.0},
                {"bottom": lambda y: y}
            ],
        },
        { # Source test
            "LENGTH": 10.0,
            "WIDTH": 10.0,
            "CX": 10,
            "CY": 10,
            "TENSOR": [
                [1.0, 0.0],
                [0.0, 1.0]
            ],
            "DIRICHLET_BC": [
                {"top": lambda y: 0.0},
            ],
            "INSIDE_BC": [
                {
                    "x_range": [0.0, 10.0],
                    "y": lambda x: 5.0,
                    "value": lambda x, y: 1.0
                }
            ]
        },
        { # Source test
            "LENGTH": 5.0,
            "WIDTH": 5.0,
            "CX": 5,
            "CY": 5,
            "TENSOR": [
                [1.0, 0.0],
                [0.0, 1.0]
            ],
            "DIRICHLET_BC": [
                {"right": lambda x: 0.0},
                {"left": lambda x: 0.0},
                {"top": lambda y: 0.0},
                {"bottom": lambda y: 0.0}
            ],
            "INSIDE_BC": [
                {
                    "x_range": [0.0, 4.0],
                    "y": lambda x: 2.5,
                    "value": lambda x, y: 1.0
                }
            ]
        },
        { # Condensator term with inside BC
            "LENGTH": 100.0,
            "WIDTH": 50.0,
            "CX": 100,
            "CY": 50,
            "TENSOR": [
                [1.0, 0.0],
                [0.0, 1.0]
            ],
            "DIRICHLET_BC": [
                {"right": lambda x: 50},
                {"left": lambda x: -50},
            ],
            "NEUMANN_BC": [
                {"top": 0.0},
                {"bottom": 0.0}
            ],
            "INSIDE_BC": [
                {
                    "x_range": [20.0, 80.0],
                    "y": lambda x: x/2,
                    "value": lambda x, y: 4.0
                }
            ]
        },
        { # Quadratic plate of Dirichlet BC
            "LENGTH": 50.0,
            "WIDTH": 50.0,
            "CX": 50,
            "CY": 50,
            "TENSOR": [
                [1.0, 0.0],
                [0.0, 1.0]
            ],
            "DIRICHLET_BC": [
                {"right": lambda x: 3},
                {"left": lambda x: 3},
                {"top": lambda y: 6},
                {"bottom": lambda y: 6}
            ]
        },
        { # Condensator term
            "LENGTH": 100.0,
            "WIDTH": 50.0,
            "CX": 100,
            "CY": 50,
            "TENSOR": [
                [1.0, 0.0],
                [0.0, 1.0]
            ],
            "DIRICHLET_BC": [
                {"right": lambda x: 50},
                {"left": lambda x: -50},
            ],
            "NEUMANN_BC": [
                {"top": 0.0},
                {"bottom": 0.0}
            ]
        }
    ]

    for idx, case in enumerate(test_cases):
        print(f"\n--- Running Test Case {idx+1} ---")
        try:
            input_data = InputData(case)
            main(input_data)
        except Exception as e:
            print(f"Error in test case {idx+1}: {e}")

test()