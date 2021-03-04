import torch
import math

def test(a):
    a.append("adsda")

x = "xyz"
test(x)
print(x)

"""
class BasicNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn())
"""