# residual_model.py
from dezero import Model
from models.residual import ResidualBlock, ResidualSequential
import dezero.layers as L

class ResidualMLP(Model):
    def __init__(self):
        super().__init__()
        self.res = ResidualSequential(
            ResidualBlock(L.Linear(100), in_features=100, out_features=100),
            ResidualBlock(L.Linear(100), in_features=100, out_features=100),
        )
        self.out = L.Linear(10)

    def forward(self, x):
        x = self.res(x)
        x = self.out(x)
        return x
    #입력 2개 출력 1개
