import numpy as np
from dezero import Dataset
from typing import Any, Tuple

class MyDataset(Dataset):
    """
    간단한 예제용 데이터셋 클래스.
    data: (N, D) ndarray 또는 list of ndarray
    labels: (N,) ndarray 또는 list
    """
    def __init__(self):
        # 실제 데이터 예시 (여기선 2D feature, 3개 샘플)
        self.data = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ], dtype=np.float32)
        self.labels = np.array([0, 1, 0], dtype=np.int64)
        assert len(self.data) == len(self.labels), "data와 labels의 길이가 다릅니다."

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"인덱스 {idx}가 데이터셋 범위를 벗어났습니다.")
        return self.data[idx], self.labels[idx]

    def get_batch(self, indices):
        """배치 인덱스 리스트로 (data, labels) 반환"""
        return self.data[indices], self.labels[indices]

train_set = MyDataset()

