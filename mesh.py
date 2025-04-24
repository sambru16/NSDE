import numpy as np

class QuadMesh:
    def __init__(self, length, height, cx, cy):
        self.length = length    #Height of the domain
        self.height = height    #Length of the domain
        self.cx = cx            #Number of cells in x
        self.cy = cy            #Number of cells in y

        self.gen_mesh()

    def gen_mesh(self):
        #Step in x and y
        dx = self.length / self.cx
        dy = self.height / self.cy

        #Generating nodes
        self.nodes = []
        for i in range(self.cy + 1):
            for j in range(self.cx + 1):
                x = j * dx
                y = i * dy
                self.nodes.append([x, y])
        self.nodes = np.array(self.nodes)

        #Defining elements (4 nodes = 1 element)
        #Order: down left -> down right -> up right -> up left
        self.elements = []
        for i in range(self.cy):
            for j in range(self.cx):
                n0 = i * (self.cx + 1) + i
                n1 = n0 + 1
                n2 = n1 + self.nx + 1
                n3 = n0 + self.nx + 1
                self.elements.append([n0, n1, n2, n3])
            self.elements = np.array(self.elements)

    #Get node coordinates
    def get_nodes(self):
        return self.nodes
    
    #Get elements
    def get_elements(self):
        return self.elements
