## May be usable

def is_point_on_line(point, start, end, tolerance=1e-6):
  line_vec = end - start
  point_vec = point - start
  line_len = np.linalg.norm(line_vec)
  if line_len == 0:
      return np.linalg.norm(point - start) < tolerance
  proj_length = np.dot(point_vec, line_vec) / line_len
  proj_point = start + proj_length * (line_vec / line_len)
  distance = np.linalg.norm(point - proj_point)
  return distance < tolerance and 0 <= proj_length <= line_len

def apply_boundary_conditions(mesh, boundary_conditions, stiffness_matrix, load_vector):
    # Apply Dirichlet conditions
    for bc in boundary_conditions.get("Dirichlet", []):
        node, value = bc["node"], bc["value"]
        stiffness_matrix[node, :] = 0
        stiffness_matrix[node, node] = 1
        load_vector[node] = value

    # Apply Neumann conditions
    for bc in boundary_conditions.get("Neumann", []):
        node, value = bc["node"], bc["value"]
        load_vector[node] += value

    # Apply line-based boundary conditions
    if "Line" in boundary_conditions:
        for line_bc in boundary_conditions["Line"]:
            start = np.array(line_bc["start"])
            end = np.array(line_bc["end"])
            value = line_bc["value"]
            bc_type = line_bc["type"]
            for i, node in enumerate(mesh.get_nodes()):
                if is_point_on_line(node, start, end):
                    if bc_type == "Dirichlet":
                        stiffness_matrix[i, :] = 0
                        stiffness_matrix[i, i] = 1
                        load_vector[i] = value
                    elif bc_type == "Neumann":
                        load_vector[i] += value