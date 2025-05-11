import numpy
import ShapeFunctions

class GaussianQuadrature:
    """
    Class to perform 1D and 2D Gaussian quadrature over intervals, rectangles, and quadrilaterals.
    """
    def __init__(self, order=2):
        if order < 1:
            raise ValueError("Order must be > 0")
        self.order = order
        self.points, self.weights = self.gaussLegendrePointsWeights()

    def gaussLegendrePointsWeights(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Computes Gauss-Legendre quadrature points and weights for the interval [-1, 1].

        Returns
        -------
        tuple of (ndarray, ndarray)
            Quadrature points and corresponding weights.
        """
        Pn = numpy.polynomial.legendre.Legendre.basis(self.order)
        points = numpy.sort(Pn.roots())
        dPn = Pn.deriv()
        weights = 2 / ((1 - points**2) * (dPn(points)**2))
        return points, weights
    
    def gaussianQuadrature(self, f: callable, a: float, b: float) -> float:
        """
        Integrates a 1D function over [a, b] using Gaussian quadrature.

        Parameters
        ----------
        f : callable
            Function of one variable.
        a, b : float
            Integration interval.

        Returns
        -------
        float
            Approximated integral.
        """
        if not callable(f):
            raise TypeError("Argument 'f' must be a callable function.")
        try:
            values = f(((b - a) * self.points + (b + a)) / 2)
            return numpy.sum(values * self.weights) * (b - a) / 2
        except Exception as e:
            raise ValueError(f"Function call f(x) failed or the return value is not numeric: {e}")

    def gaussianQuadratureRectangle(self, f: callable, x0: float, x1: float, y0: float, y1: float) -> float:
        """
        Integrates a 2D function over an axis-aligned rectangle.

        Parameters
        ----------
        f : callable
            Function of two variables.
        x0, x1 : float
            Horizontal bounds.
        y0, y1 : float
            Vertical bounds.

        Returns
        -------
        float
            Approximated integral.
        """
        if not callable(f):
            raise TypeError("Argument 'f' must be a callable function.")
        x, y = numpy.meshgrid(((x1 - x0) * self.points + (x1 + x0)) / 2, ((y1 - y0) * self.points + (y1 + y0)) / 2)
        try:
            values = f(x, y)
            return numpy.sum(values * numpy.outer(self.weights, self.weights)) * (x1 - x0) * (y1 - y0) / 4
        except Exception as e:
            raise ValueError(f"Function call f(x, y) failed or the return value is not numeric: {e}")

    def gaussianQuadratureQuadrilateral(self, f: callable, vertices) -> float:
        """
        Integrates a 2D function over an arbitrary quadrilateral using Gaussian quadrature.

        Parameters
        ----------
        f : callable
            Function of two variables.
        vertices : array-like of shape (4, 2)
            Coordinates of the quadrilateral corners in counter-clockwise order.

        Returns
        -------
        float
            Approximated integral.
        """
        if not callable(f):
            raise TypeError("Argument 'f' must be a callable function.")
        vertices = numpy.asarray(vertices)
        if vertices.shape != (4, 2):
            raise TypeError("vertices must be a (4, 2) array representing 4 corner points in 2D.")
        integral = 0.0
        for i, xi in enumerate(self.points):
            for j, eta in enumerate(self.points):
                w = self.weights[i] * self.weights[j]
                N = ShapeFunctions.shape_functions(xi, eta)
                dN_dxi_eta = ShapeFunctions.shape_functions_deriviative(xi, eta)
                x, y = N @ vertices
                J = ShapeFunctions.jacobi_matrix(dN_dxi_eta, vertices)  # (2, 2)
                detJ = numpy.linalg.det(J)
                try:
                    f_val = f(x, y)
                    integral += f_val * w * detJ
                except Exception as e:
                    raise ValueError(f"Function call f(x, y) failed or the return value is not numeric: {e}")
        return integral
        
    def localStiffnessMatrix(self, conductivity, vertices):
        """
        Computes the local stiffness matrix for a 4-node quadrilateral element.

        Parameters
        ----------
        conductivity : ndarray of shape (2, 2)
            Conductivity tensor (e.g., identity for isotropic).
        vertices : array-like of shape (4, 2)
            Coordinates of the element nodes.

        Returns
        -------
        ndarray of shape (4, 4)
            Local stiffness matrix.
        """   
        vertices = numpy.asarray(vertices)
        if vertices.shape != (4, 2):
            raise TypeError("vertices must be a (4, 2) array representing 4 corner points in 2D.")
        K = numpy.zeros((4, 4))
        for i, xi in enumerate(self.points):
            for j, eta in enumerate(self.points):
                dN_dxi_eta = ShapeFunctions.shape_functions_deriviative(xi, eta)
                J = ShapeFunctions.jacobi_matrix(dN_dxi_eta, vertices)
                dN_dx = ShapeFunctions.shape_function_derivatives_global(dN_dxi_eta, J)
                K += self.weights[i] * self.weights[j] * numpy.linalg.det(J) * (dN_dx @ conductivity @ dN_dx.T)
        return K

    def loadVector(self, f: callable, vertices):
        """
        Computes the load vector for a 4-node quadrilateral element.

        Parameters
        ----------
        f : callable
            Function of two variables.
        vertices : array-like of shape (4, 2)
            Coordinates of the element nodes.

        Returns
        -------
        ndarray of shape (4,)
            Local load vector
        """   
        if not callable(f):
            raise TypeError("Argument 'f' must be a callable function.")
        vertices = numpy.asarray(vertices)
        if vertices.shape != (4, 2):
            raise TypeError("vertices must be a (4, 2) array representing 4 corner points in 2D.")
        integral = 0.0
        for i, xi in enumerate(self.points):
            for j, eta in enumerate(self.points):
                w = self.weights[i] * self.weights[j]
                N = ShapeFunctions.shape_functions(xi, eta)
                dN_dxi_eta = ShapeFunctions.shape_functions_deriviative(xi, eta)
                x, y = N @ vertices
                J = ShapeFunctions.jacobi_matrix(dN_dxi_eta, vertices)  # (2, 2)
                detJ = numpy.linalg.det(J)
                try:
                    integral += N * f(x, y) * w * detJ
                except Exception as e:
                    raise ValueError(f"Function call f(x, y) failed or the return value is not numeric: {e}")
        return integral


# # How to use: 
# # example function
# f = lambda x, y: x + y
# gauss1 = GaussianQuadrature()
# gauss2 = GaussianQuadrature(10)
# # # 1st-order Gaussian quadrature of function f(x, y) over the rectangle [[0, 0], [2, 1], [0, 1], [2, 0]]
# print(gauss1.gaussianQuadratureRectangle(f, 0, 2, 0, 1))
# # # 20th-order Gaussian quadrature of the same function over the same rectangle
# print(gauss2.gaussianQuadratureRectangle(f, 0, 2, 0, 1))
# # Define 4 points (x_i, y_i)
# quad = [[0, 0], [2, 0], [2, 1], [0, 1]]
# # Gaussian quadrature of function f over the defined quadrilateral (CCW corner order is required)
# print(gauss1.gaussianQuadratureQuadrilateral(f, quad))
# print(gauss2.gaussianQuadratureQuadrilateral(f, quad))
# # Conductivity Matrix example
# conductivity = numpy.array([[1.0, 0.0],[0.0, 1.0]])
# # Calculation of local stiffness matrix
# print(gauss1.localStiffnessMatrix(conductivity, quad))
# print(gauss2.localStiffnessMatrix(conductivity, quad))
# # Calculation of load vector
# print(gauss1.loadVector(f, quad))
# print(gauss2.loadVector(f, quad))