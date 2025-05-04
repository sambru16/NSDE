import numpy as np
from scipy.sparse import lil_matrix

class MaterialModel:
    def __init__(self, mesh, material_tensor, boundary_conditions):
        self.mesh = mesh
        self.material_tensor = np.array(material_tensor)
        self.boundary_conditions = boundary_conditions
        self.stiffness_matrix = None
        self.load_vector = None

    def assemble_system(self):
        num_nodes = len(self.mesh.get_nodes())
        self.stiffness_matrix = lil_matrix((num_nodes, num_nodes))
        self.load_vector = np.zeros(num_nodes)

        for element in self.mesh.get_elements():
            element_nodes = self.mesh.get_nodes()[element]
            ke, fe = self.compute_element_matrices(element_nodes)
            for i, ni in enumerate(element):
                for j, nj in enumerate(element):
                    self.stiffness_matrix[ni, nj] += ke[i, j]
                self.load_vector[ni] += fe[i]

        self.apply_boundary_conditions()

    def compute_element_matrices(self, element_nodes):
        # Numerical integration for stiffness matrix and load vector
        ke = np.zeros((4, 4))
        fe = np.zeros(4)
        # ... Integration logic here ...
        return ke, fe

    def apply_boundary_conditions(self):
        # Apply Dirichlet conditions
        for bc in self.boundary_conditions["Dirichlet"]:
            node, value = bc["node"], bc["value"]
            self.stiffness_matrix[node, :] = 0
            self.stiffness_matrix[node, node] = 1
            self.load_vector[node] = value

        # Apply Neumann conditions
        for bc in self.boundary_conditions["Neumann"]:
            node, value = bc["node"], bc["value"]
            self.load_vector[node] += value

        # Apply line-based boundary conditions
        if "Line" in self.boundary_conditions:
            for line_bc in self.boundary_conditions["Line"]:
                start = np.array(line_bc["start"])
                end = np.array(line_bc["end"])
                value = line_bc["value"]
                bc_type = line_bc["type"]

                # Find nodes close to the line
                for i, node in enumerate(self.mesh.get_nodes()):
                    if self.is_point_on_line(node, start, end):
                        if bc_type == "Dirichlet":
                            self.stiffness_matrix[i, :] = 0
                            self.stiffness_matrix[i, i] = 1
                            self.load_vector[i] = value
                        elif bc_type == "Neumann":
                            self.load_vector[i] += value

    def is_point_on_line(self, point, start, end, tolerance=1e-6):
        # Check if a point is close to a line segment
        line_vec = end - start
        point_vec = point - start
        proj_length = np.dot(point_vec, line_vec) / np.linalg.norm(line_vec)
        proj_point = start + proj_length * (line_vec / np.linalg.norm(line_vec))
        distance = np.linalg.norm(point - proj_point)
        return distance < tolerance and 0 <= proj_length <= np.linalg.norm(line_vec)
