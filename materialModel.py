import numpy as np

class MaterialModel:
    def __init__(self, material_tensor):
        self.material_tensor = np.array(material_tensor)
