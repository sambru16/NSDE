import numpy as np

class ShapeFunctions:
    def shape_functions(xi, eta):
        N = np.array([
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta)
        ])
        
        dN_dxi_deta = np.array([
            [-0.25 * (1 - eta), -0.25 * (1 - xi)],  
            [ 0.25 * (1 - eta), -0.25 * (1 + xi)],
            [ 0.25 * (1 + eta),  0.25 * (1 + xi)],
            [-0.25 * (1 + eta),  0.25 * (1 - xi)]
        ])
        
        return N, dN_dxi_deta

    def jacobi_matrix(dN_dxi_deta, element_coords):
        J = np.zeros((2, 2))
        for i in range(4):
            J[0, 0] += dN_dxi_deta[i, 0] * element_coords[i, 0]
            J[0, 1] += dN_dxi_deta[i, 1] * element_coords[i, 0]
            J[1, 0] += dN_dxi_deta[i, 0] * element_coords[i, 1]
            J[1, 1] += dN_dxi_deta[i, 1] * element_coords[i, 1]
        return J

    def shape_function_derivatives_global(dN_dxi_deta, J):
        J_inv = np.linalg.inv(J)
        dN_dx = np.dot(dN_dxi_deta, J_inv.T)
        return dN_dx





