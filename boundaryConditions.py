import numpy as np
from GaussianQuadrature import GaussianQuadrature

class BoundaryCondition:
    def __init__(self, dirichlet, neumann, inside, element_lenght):
        self.dirichlet_ = dirichlet if dirichlet is not None else []
        self.neumann_ = neumann if neumann is not None else []
        self.inside_ = inside  if inside is not None else []
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
        # Track which corners have Dirichlet set
        dirichlet_corners = {
            "bottom_left": len(corner_values["bottom_left"]) > 0,
            "bottom_right": len(corner_values["bottom_right"]) > 0,
            "top_left": len(corner_values["top_left"]) > 0,
            "top_right": len(corner_values["top_right"]) > 0,
        }
        for name, idx in zip(
            ["bottom_left", "bottom_right", "top_left", "top_right"],
            [bottom_left, bottom_right, top_left, top_right]
        ):
            vals = corner_values[name]
            # If Dirichlet is set at this corner, ignore Neumann collision and do not print
            if dirichlet_corners[name]:
                continue
            if len(vals) > 1:
                if not all(np.isclose(vals[0], v) for v in vals[1:]):
                    mean_val = np.mean(vals)
                    print(f"Neumann conditions collide at {name.replace('_', '-')} corner. Arithmetic middle [value: {mean_val}] will be used.")
                    load_vector[idx] += mean_val

        # -----------------------------------------
        # apply inside source term (robust, conservative, correct: each quad point to its element)
        # -----------------------------------------
        elements = mesh.get_elements() if hasattr(mesh, "get_elements") else []
        total_source = 0.0  # For diagnostic print
        for inside in self.inside_:
            if "x_range" in inside and "y" in inside and "value" in inside:
                x_min, x_max = inside["x_range"]
                y_func = inside["y"]
                value_func = inside["value"]
                x_vals = [x_min + i * (x_max - x_min) / mesh.cx for i in range(mesh.cx + 1)]
                for i in range(len(x_vals) - 1):
                    x0, x1 = x_vals[i], x_vals[i+1]
                    y0, y1 = y_func(x0), y_func(x1)
                    p0 = np.array([x0, y0])
                    p1 = np.array([x1, y1])
                    segment_length = np.linalg.norm(p1 - p0)
                    gauss_points, gauss_weights = np.polynomial.legendre.leggauss(4)
                    for s, w in zip((gauss_points + 1) / 2, gauss_weights / 2):  # [0,1]
                        pt = p0 + s * (p1 - p0)
                        v = value_func(pt[0], pt[1])
                        # Find the unique element containing pt
                        for elem in elements:
                            elem_nodes = nodes[list(elem)]
                            x_e_min, y_e_min = np.min(elem_nodes, axis=0)
                            x_e_max, y_e_max = np.max(elem_nodes, axis=0)
                            if not (x_e_min <= pt[0] <= x_e_max and y_e_min <= pt[1] <= y_e_max):
                                continue
                            x1e, y1e = elem_nodes[0]
                            x2e, y2e = elem_nodes[1]
                            x3e, y3e = elem_nodes[2]
                            x4e, y4e = elem_nodes[3]
                            dx = x2e - x1e
                            dy = y4e - y1e
                            if dx == 0 or dy == 0:
                                continue
                            xi = 2 * (pt[0] - x1e) / dx - 1
                            eta = 2 * (pt[1] - y1e) / dy - 1
                            if not (-1 <= xi <= 1 and -1 <= eta <= 1):
                                continue
                            N = np.array([
                                0.25 * (1 - xi) * (1 - eta),
                                0.25 * (1 + xi) * (1 - eta),
                                0.25 * (1 + xi) * (1 + eta),
                                0.25 * (1 - xi) * (1 + eta)
                            ])
                            for local_idx, global_idx in enumerate(elem):
                                contrib = v * N[local_idx] * w * segment_length
                                load_vector[global_idx] += contrib
                                total_source += contrib
                            break  # Only one element contains pt
        print(f"Total inside source applied to load vector: {total_source}")
