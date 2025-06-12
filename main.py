import numpy as np
from dezero import Variable
from models.residual import ResidualBlock, ResidualSequential
import dezero.layers as L

# 샘플 입력
x = Variable(np.random.randn(1, 100))

# ResidualBlock 테스트
linear = L.Linear(100)
res_block = ResidualBlock(linear)
y1 = res_block(x)
print("ResidualBlock 출력 shape:", y1.shape)

# ResidualSequential 테스트 (여러 레이어)
seq = ResidualSequential(
    L.Linear(100),
    L.Linear(100),
    L.Linear(100)
)
y2 = seq(x)
print("ResidualSequential 출력 shape:", y2.shape)

# --- ResidualBlock 다양한 케이스 테스트 ---
from dezero import Layer

# 1. shape이 같은 경우
x_same = Variable(np.random.randn(1, 100))
block_same = ResidualBlock(L.Linear(100), in_features=100, out_features=100)
y_same = block_same(x_same)
print("[TEST] shape 같을 때:", y_same.shape)

# 2. shape이 다른 경우 (projection 적용)
block_proj = ResidualBlock(L.Linear(50), in_features=100, out_features=50)
y_proj = block_proj(x_same)
print("[TEST] shape 다를 때(projection):", y_proj.shape)

# 3. 레이어가 없는 경우 (Identity)
class Identity(Layer):
    def forward(self, x):
        return x
block_identity = ResidualBlock(Identity(), in_features=100, out_features=100)
y_identity = block_identity(x_same)
print("[TEST] 레이어가 없을 때:", y_identity.shape)

