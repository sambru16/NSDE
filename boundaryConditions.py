import numpy as np
from GaussianQuadrature import GaussianQuadrature

class BoundaryCondition:
    def __init__(self, dirichlet, neumann, inside, element_lenght):
        self.dirichlet_ = dirichlet
        self.neumann_ = neumann
        self.inside_ = inside
        self.element_lenght_ = element_lenght

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

        # Prepare to collect corner values from each side
        corner_values = {
            "bottom_left": [],
            "bottom_right": [],
            "top_left": [],
            "top_right": []
        }

        # -----------------------------------------
        # Apply Dirichlet conditions
        # -----------------------------------------
        for d in self.dirichlet_:
            side = next(iter(d))
            cond = d[side]
            if side == "left":
                mask = np.isclose(nodes[:,0], x_min)
                param = nodes[mask][:,1]  # y coordinate
                length = y_max - y_min
            elif side == "right":
                mask = np.isclose(nodes[:,0], x_max)
                param = nodes[mask][:,1]  # y coordinate
                length = y_max - y_min
            elif side == "bottom":
                mask = np.isclose(nodes[:,1], y_min)
                param = nodes[mask][:,0]  # x coordinate
                length = x_max - x_min
            elif side == "top":
                mask = np.isclose(nodes[:,1], y_max)
                param = nodes[mask][:,0]  # x coordinate
                length = x_max - x_min
            else:
                continue

            node_indices = np.where(mask)[0]
            for idx, p in zip(node_indices, param):
                t = (p - (y_min if side in ["left", "right"] else x_min)) / (length if length != 0 else 1)
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
                    print(f"Dirichlet conditions collide at {name.replace('_', '-')} corner. Arithmetic middle [value: {mean_val}] will be used.")
                    load_vector[idx] = mean_val

        # -----------------------------------------
        # Apply Neumann conditions with Gaussian quadrature, skipping Dirichlet nodes
        # -----------------------------------------
        gauss = GaussianQuadrature()
        for n in self.neumann_:
            side = next(iter(n))
            cond = n[side]
            if side == "left":
                mask = np.isclose(nodes[:,0], x_min)
                coords = nodes[mask][:,1]
                axis = 1
            elif side == "right":
                mask = np.isclose(nodes[:,0], x_max)
                coords = nodes[mask][:,1]
                axis = 1
            elif side == "bottom":
                mask = np.isclose(nodes[:,1], y_min)
                coords = nodes[mask][:,0]
                axis = 0
            elif side == "top":
                mask = np.isclose(nodes[:,1], y_max)
                coords = nodes[mask][:,0]
                axis = 0
            else:
                continue
            node_indices = np.where(mask)[0]
            # Sort nodes along the boundary for correct edge pairing
            sort_idx = np.argsort(coords)
            node_indices = node_indices[sort_idx]
            coords = coords[sort_idx]
            # Integrate over each edge between two nodes
            for i in range(len(node_indices) - 1):
                idx1 = node_indices[i]
                idx2 = node_indices[i+1]
                p1 = nodes[idx1]
                p2 = nodes[idx2]
                edge_length = np.linalg.norm(p2 - p1)
                # Define function for quadrature along the edge
                def g(s):
                    # s in [0, 1], interpolate position
                    pt = p1 + s * (p2 - p1)
                    if callable(cond):
                        if axis == 0:
                            return cond(pt[0])
                        else:
                            return cond(pt[1])
                    else:
                        return cond
                integral = gauss.gaussianQuadrature(g, 0, 1) * edge_length
                load_vector[idx1] += 0.5 * integral
                load_vector[idx2] += 0.5 * integral
                if idx == bottom_left:
                    corner_values["bottom_left"].append(value)
                if idx == bottom_right:
                    corner_values["bottom_right"].append(value)
                if idx == top_left:
                    corner_values["top_left"].append(value)
                if idx == top_right:
                    corner_values["top_right"].append(value)

        # Handle Neumann corner along with dirichlet cases
        for name, idx in zip(
            ["bottom_left", "bottom_right", "top_left", "top_right"],
            [bottom_left, bottom_right, top_left, top_right]
        ):
            vals = corner_values[name]
            if len(vals) > 1:
                if not all(np.isclose(vals[0], v) for v in vals[1:]):
                    mean_val = np.mean(vals)
                    print(f"Neumann conditions collide at {name.replace('_', '-')} corner. Arithmetic middle [value: {mean_val}] will be used.")
                    load_vector[idx] += mean_val

        # -----------------------------------------
        # apply inside conditions
        # -----------------------------------------
        nodes = np.array(mesh.get_nodes())
        elements = mesh.get_elements() if hasattr(mesh, "get_elements") else []
        for inside in self.inside_:
            btype = inside.get("type", "").lower()
            # New format: x_range, y(x), value(x, y)
            if "x_range" in inside and "y" in inside and "value" in inside:
                x_range = inside["x_range"]
                y_func = inside["y"]
                value_func = inside["value"]
                x_min, x_max = x_range
                # Discretize the line into segments between mesh nodes in x
                x_vals = [x_min + i * (x_max - x_min) / max(1, mesh.cx) for i in range(mesh.cx + 1)]
                for i in range(len(x_vals) - 1):
                    x0, x1 = x_vals[i], x_vals[i+1]
                    y0, y1 = y_func(x0), y_func(x1)
                    p0 = np.array([x0, y0])
                    p1 = np.array([x1, y1])
                    # For each element, check if the segment [p0, p1] crosses the element
                    for elem in elements:
                        elem_nodes = nodes[list(elem)]
                        x_e_min, y_e_min = np.min(elem_nodes, axis=0)
                        x_e_max, y_e_max = np.max(elem_nodes, axis=0)
                        # Check if either p0 or p1 is inside the element bounding box
                        for pt in [p0, p1]:
                            if (x_e_min <= pt[0] <= x_e_max) and (y_e_min <= pt[1] <= y_e_max):
                                # Find nearest node in this element to pt
                                dists = np.linalg.norm(elem_nodes - pt, axis=1)
                                local_idx = np.argmin(dists)
                                global_idx = elem[local_idx]
                                x, y = nodes[global_idx]
                                if btype == "dirichlet":
                                    stiffness_matrix[global_idx, :] = 0
                                    stiffness_matrix[global_idx, global_idx] = 1
                                    v = value_func(x, y)
                                    load_vector[global_idx] = v
                                elif btype == "neumann":
                                    v = value_func(x, y)
                                    load_vector[global_idx] += v
                                break  # Only apply to one node per element per segment
