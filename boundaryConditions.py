import numpy as np

class BoundaryCondition:
    def __init__(self, neuman, dirichlet):
        self.neuman_ = neuman
        self.dirichlet_ = dirichlet


def isPointOnFunction(point, start, end, tolerance=1e-6):
  line_vec = end - start
  point_vec = point - start
  line_len = np.linalg.norm(line_vec)
  if line_len == 0:
    return np.linalg.norm(point - start) < tolerance
  proj_length = np.dot(point_vec, line_vec) / line_len
  proj_point = start + proj_length * (line_vec / line_len)
  distance = np.linalg.norm(point - proj_point)
  return distance < tolerance and 0 <= proj_length <= line_len

def applyBoundaryConditions(mesh, boundary_conditions, stiffness_matrix, load_vector):
    """
    Apply Dirichlet and Neumann boundary conditions to the system.
    boundary_conditions: dict with keys "LEFT", "RIGHT", "TOP", "BOTTOM", each mapping to a lambda or value.
    """
    nodes = np.array(mesh.get_nodes())
    num_nodes = len(nodes)

    # Get mesh bounds
    x_min, y_min = np.min(nodes, axis=0)
    x_max, y_max = np.max(nodes, axis=0)

    # Helper to apply a condition to a node
    def apply_dirichlet(node_idx, value):
        stiffness_matrix[node_idx, :] = 0
        stiffness_matrix[node_idx, node_idx] = 1
        load_vector[node_idx] = value

    def apply_neumann(node_idx, value):
        load_vector[node_idx] += value

    # For each side, find nodes and apply boundary conditions
    for side in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
        if side in boundary_conditions:
            cond = boundary_conditions[side]
            # Find nodes on this side
            if side == "LEFT":
                mask = np.isclose(nodes[:,0], x_min)
                param = nodes[mask][:,1]  # y coordinate
                length = y_max - y_min
            elif side == "RIGHT":
                mask = np.isclose(nodes[:,0], x_max)
                param = nodes[mask][:,1]  # y coordinate
                length = y_max - y_min
            elif side == "BOTTOM":
                mask = np.isclose(nodes[:,1], y_min)
                param = nodes[mask][:,0]  # x coordinate
                length = x_max - x_min
            elif side == "TOP":
                mask = np.isclose(nodes[:,1], y_max)
                param = nodes[mask][:,0]  # x coordinate
                length = x_max - x_min
            else:
                continue

            node_indices = np.where(mask)[0]
            for idx, p in zip(node_indices, param):
                # Normalize parameter along the side to [0, 1]
                t = (p - (y_min if side in ["LEFT", "RIGHT"] else x_min)) / (length if length != 0 else 1)
                # Evaluate lambda or use value
                if callable(cond):
                    value = cond(t * length)
                else:
                    value = cond
                apply_dirichlet(idx, value)

    # Neumann conditions can be handled similarly if needed
    # ...existing code for Neumann or line-based conditions...