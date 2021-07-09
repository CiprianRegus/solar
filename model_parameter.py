import math

class ModelParameter():
    
    """ 
        This class will be used as a data structure for passing dataset records to the prediction model
    """

    def __init__(self, name, value, mean=0.0, std=0.0):
            
        self.name = name
        self.value = value
        self.mean = mean
        self.std = std
     

    def normalize(self):
        """
            There are 2 cases: computing z-score or sin, depending if the column represents the time
        """
        if self.name == "Time":
            return math.sin(math.pi * self.value / (24 * 3600))
        else:
            nvalue = (self.value - self.mean) / self.std
            return nvalue

    @staticmethod
    def denormalize(nvalue, mean, std):
        
        dvalue =  nvalue * std + mean
        return dvalue



