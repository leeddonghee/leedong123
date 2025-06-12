# normal_model.py
from dezero import Model
import dezero.layers as L
import dezero.functions as F

class SimpleMLP(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)
        self.l2 = L.Linear(100)
        self.l3 = L.Linear(10)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
