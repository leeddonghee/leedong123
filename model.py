import numpy as np
import dezero.layers as L
import dezero.functions as F
from dezero import Layer, Model, Variable
import dezero.utils as utils
from dezero.models import Sequential

class ResidualBlock(Layer):
    def __init__(self, layer, in_channels=None, out_channels=None, in_features=None, out_features=None):
        super().__init__()
        if isinstance(layer, list):
            self.layer = Sequential(*layer)
            # Conv2d 계열
            if hasattr(layer[-1], 'out_channels'):
                out_ch = out_channels if out_channels is not None else layer[-1].out_channels
                self.is_conv = True
            # Linear 계열
            elif hasattr(layer[-1], 'out_size'):
                out_ch = out_features if out_features is not None else layer[-1].out_size
                self.is_conv = False
            else:
                out_ch = None
                self.is_conv = None
        else:
            self.layer = layer
            if hasattr(layer, 'out_channels'):
                out_ch = out_channels if out_channels is not None else layer.out_channels
                self.is_conv = True
            elif hasattr(layer, 'out_size'):
                out_ch = out_features if out_features is not None else layer.out_size
                self.is_conv = False
            else:
                out_ch = None
                self.is_conv = None

        self.projection = None
        # Conv2d 계열
        if self.is_conv and in_channels is not None and in_channels != out_ch:
            self.projection = L.Conv2d(out_ch, 1, 1, 0)
        # Linear 계열
        elif self.is_conv is False and in_features is not None and in_features != out_ch:
            self.projection = L.Linear(out_ch)

    def forward(self, x):
        y = self.layer(x)
        shortcut = x
        # projection 적용
        if self.projection is not None:
            shortcut = self.projection(x)
        # shape이 다르면 spatial/feature 차원 crop
        if shortcut.shape != y.shape:
            # Conv2d 계열: (N, C, H, W)
            if self.is_conv and len(y.shape) == 4:
                N, C, H, W = y.shape
                shortcut = shortcut[:, :C, :H, :W]
                y = y[:, :C, :H, :W]
            # Linear 계열: (N, D)
            elif not self.is_conv and len(y.shape) == 2:
                N, D = y.shape
                shortcut = shortcut[:, :D]
                y = y[:, :D]
            else:
                raise ValueError(f"ResidualBlock: shape mismatch {shortcut.shape} vs {y.shape}")
        return shortcut + y
        
class Modelee(Model):
    def __init__(self):
        super().__init__()
        self.embedding = L.Linear(10) # out_channels=10
        self.block1 = ResidualBlock([
            L.Linear(10),
            L.Linear(10)
        ], in_channels=10, out_channels=20)

    def forward(self, x):
        x = self.embedding(x)
        x = self.block1(x)
        return x


class ResidualMLP(Model):  # Model 상속 → plot 가능
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)
        self.resblock = ResidualBlock(L.Linear(100), in_features=100, out_features=100)
        self.l2 = L.Linear(10)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.resblock(x)
        x = self.l2(x)
        return x


# ===== 실행 예시 및 그래프 시각화 =====

# 모델 생성
model = Modelee()

# 입력 데이터 준비 (5 samples, 3채널, 24x24)
x = np.random.rand(5, 100).astype(np.float32)

# 계산 그래프 그리기
model.plot(x, to_file='residual_mlp_graph.png')
