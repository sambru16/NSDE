import numpy as np

class BoundaryCondition:
    def __init__(self, dirichlet):
        self.__dirichlet_corner = [0, 0, 0, 0]
        self.dirichlet_ = dirichlet

    def apply(self, mesh, stiffness_matrix, load_vector):
        """
        Apply Dirichlet conditions to the stiffness matrix and load vector using mesh info.
        """
        nodes = np.array(mesh.get_nodes())
        x_min, y_min = np.min(nodes, axis=0)
        x_max, y_max = np.max(nodes, axis=0)

        # Corner indices
        bottom_left  = np.where(np.isclose(nodes[:,0], x_min) & np.isclose(nodes[:,1], y_min))[0][0]
        bottom_right = np.where(np.isclose(nodes[:,0], x_max) & np.isclose(nodes[:,1], y_min))[0][0]
        top_left     = np.where(np.isclose(nodes[:,0], x_min) & np.isclose(nodes[:,1], y_max))[0][0]
        top_right    = np.where(np.isclose(nodes[:,0], x_max) & np.isclose(nodes[:,1], y_max))[0][0]

        dirichlet_dict = self.dirichlet_

        # Prepare to collect corner values from each side
        corner_values = {
            "bottom_left": [],
            "bottom_right": [],
            "top_left": [],
            "top_right": []
        }

        # Apply Dirichlet conditions
        for side in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
            if side in dirichlet_dict:
                cond = dirichlet_dict[side]
                if side == "LEFT":
                    self.__incrementCornerState(3)
                    mask = np.isclose(nodes[:,0], x_min)
                    param = nodes[mask][:,1]  # y coordinate
                    length = y_max - y_min
                elif side == "RIGHT":
                    self.__incrementCornerState(1)
                    mask = np.isclose(nodes[:,0], x_max)
                    param = nodes[mask][:,1]  # y coordinate
                    length = y_max - y_min
                elif side == "BOTTOM":
                    self.__incrementCornerState(0)
                    mask = np.isclose(nodes[:,1], y_min)
                    param = nodes[mask][:,0]  # x coordinate
                    length = x_max - x_min
                elif side == "TOP":
                    self.__incrementCornerState(2)
                    mask = np.isclose(nodes[:,1], y_max)
                    param = nodes[mask][:,0]  # x coordinate
                    length = x_max - x_min
                else:
                    continue

                node_indices = np.where(mask)[0]
                for idx, p in zip(node_indices, param):
                    t = (p - (y_min if side in ["LEFT", "RIGHT"] else x_min)) / (length if length != 0 else 1)
                    if callable(cond):
                        value = cond(t * length)
                    else:
                        value = cond
                    stiffness_matrix[idx, :] = 0
                    stiffness_matrix[idx, idx] = 1
                    load_vector[idx] = value

                    # Store corner values for later collision handling
                    if idx == bottom_left:
                        corner_values["bottom_left"].append(value)
                    if idx == bottom_right:
                        corner_values["bottom_right"].append(value)
                    if idx == top_left:
                        corner_values["top_left"].append(value)
                    if idx == top_right:
                        corner_values["top_right"].append(value)

        # Handle corner cases: if a corner has more than one value, assign arithmetic mean and print message
        for name, idx in zip(
            ["bottom_left", "bottom_right", "top_left", "top_right"],
            [bottom_left, bottom_right, top_left, top_right]
        ):
            vals = corner_values[name]
            if len(vals) > 1:
                if not all(np.isclose(vals[0], v) for v in vals[1:]):
                    mean_val = np.mean(vals)
                    print(f"Dirichlet conditions collide at {name.replace('_', '-')} corner. Arithmetic middle will be used.")
                    stiffness_matrix[idx, :] = 0
                    stiffness_matrix[idx, idx] = 1
                    load_vector[idx] = mean_val

    def __getCornerIndices(self,nodes, x_min, y_min, x_max, y_max):
        """
        Get the corner indices of the mesh.
        x_min, y_min: minimum coordinates of the mesh
        x_max, y_max: maximum coordinates of the mesh
        """
        bottom_left  = np.where(np.isclose(nodes[:,0], x_min) & np.isclose(nodes[:,1], y_min))[0][0]
        bottom_right = np.where(np.isclose(nodes[:,0], x_max) & np.isclose(nodes[:,1], y_min))[0][0]
        top_left     = np.where(np.isclose(nodes[:,0], x_min) & np.isclose(nodes[:,1], y_max))[0][0]
        top_right    = np.where(np.isclose(nodes[:,0], x_max) & np.isclose(nodes[:,1], y_max))[0][0]


    def __applyDirichlet(self, stiffness_matrix, load_vector):
        bottom_left = [None, None]
        bottom_right = [None, None]
        top_right = [None, None]
        top_left = [None, None]
        for d in self.dirichlet_:
            if isinstance(d, dict):
                for side, value in d.items():
                    if side == "LEFT":
                        self.__applyConditions(stiffness_matrix, load_vector, 0, value)
                        bottom_left[0] = value
                        top_left[0] = value
                    elif side == "RIGHT":
                        self.__applyConditions(stiffness_matrix, load_vector, 0, value)
                        bottom_right[0] = value
                        top_right[0] = value
                    elif side == "TOP":
                        self.__applyConditions(stiffness_matrix, load_vector, -1, value)
                        top_left[1] = value
                        top_right[1] = value
                    elif side == "BOTTOM":
                        self.__applyConditions(stiffness_matrix, load_vector, -1, value)
                        bottom_left[1] = value
                        bottom_right[1] = value

        # Handle corner cases
        if (bottom_left[0] != bottom_left[1]
            and bottom_left[0] is not None
            and bottom_left[1] is not None):
            print("Dirichlet conditions collide at bottom-left corner. Arithmetic middle will be used.")
            self.__applyConditions(stiffness_matrix, load_vector, 0, (bottom_left[0] + bottom_left[1]) / 2)
        if (bottom_right[0] != bottom_right[1]
            and bottom_right[0] is not None
            and bottom_right[1] is not None):
            print("Dirichlet conditions collide at bottom-right corner. Arithmetic middle will be used.")
            self.__applyConditions(stiffness_matrix, load_vector, -1, (bottom_right[0] + bottom_right[1]) / 2)
        if (top_left[0] != top_left[1]
            and top_left[0] is not None
            and top_left[1] is not None):
            print("Dirichlet conditions collide at top-left corner. Arithmetic middle will be used.")
            self.__applyConditions(stiffness_matrix, load_vector, 0, (top_left[0] + top_left[1]) / 2)
        if (top_right[0] != top_right[1]
            and top_right[0] is not None
            and top_right[1] is not None):
            print("Dirichlet conditions collide at top-right corner. Arithmetic middle will be used.")
            self.__applyConditions(stiffness_matrix, load_vector, -1, (top_right[0] + top_right[1]) / 2)
            
                    
    def __incrementCornerState(self, side_index):
        """
        Increment the corner state for the given side index.
        side_index: index of the side
        0: BOTTOM,
        1: RIGHT,
        2: TOP,
        3: LEFT
        """
        self.__dirichlet_corner[side_index] += 1
        self.__dirichlet_corner[(side_index + 1) % 4] += 1

    def __applyConditions(self, stiffness_matrix, load_vector, position, value):
        """
        Apply the conditions to the stiffness matrix and load vector.
        position: index of the node
        value: value to be applied
        """
        stiffness_matrix[position, :] = 0
        stiffness_matrix[position, position] = 1
        load_vector[position] = value


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