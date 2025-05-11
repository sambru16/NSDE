import numpy

class ShapeFunctions:
    def shape_functions(xi, eta):
        return numpy.array([
            (1 - xi) * (1 - eta) / 4,
            (1 + xi) * (1 - eta) / 4,
            (1 + xi) * (1 + eta) / 4,
            (1 - xi) * (1 + eta) / 4
        ])
    
    def shape_functions_deriviative(xi, eta):
        return numpy.array([
            [-(1 - eta) / 4, -(1 - xi) / 4],  
            [ (1 - eta) / 4, -(1 + xi) / 4],
            [ (1 + eta) / 4,  (1 + xi) / 4],
            [-(1 + eta) / 4,  (1 - xi) / 4]
        ])
    
    def coordinate_transformation(N, coordinates):
        return N.T @ coordinates

    def jacobi_matrix(dN_dx, coordinates):
        return dN_dx.T @ coordinates
        
    def shape_function_derivatives_global(dN_dx, J):
        return dN_dx @ numpy.linalg.inv(J)