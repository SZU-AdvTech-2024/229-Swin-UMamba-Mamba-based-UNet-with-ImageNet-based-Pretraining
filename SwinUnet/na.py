# import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 生成不均衡数据集
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2,
                           weights=[0.9, 0.1], flip_y=0, random_state=42,
                           n_informative=2,n_repeated=0, n_redundant=0)

# 可视化数据
plt.figure(figsize=(8,6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0 (Normal)', alpha=0.6)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1 (Fault)', alpha=0.6)
plt.title('Imbalanced Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
