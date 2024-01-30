import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, num_samples, num_features, num_classes, imbalance_alpha=None):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_channels = 1

        # 入力データの生成
        self.data = np.random.normal(0, 1, (num_samples, num_features))

        # クラスの偏りを指定するかどうか
        if imbalance_alpha is not None:
            # ディリクレ分布を使用してクラスの偏りを作成
            self.targets = np.random.choice(num_classes, num_samples, p=np.random.dirichlet(np.repeat(imbalance_alpha, num_classes)))
        else:
            # クラスのサンプル数を均等に分割
            self.targets = np.random.randint(0, num_classes, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # NumPy配列からPyTorchのテンソルに変換
        data_tensor = torch.from_numpy(self.data[idx]).float()
        target_tensor = torch.tensor(self.targets[idx], dtype=torch.long)
        return data_tensor, target_tensor
