import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)

# ----------------- 1. 准备简单的数据和模型 -----------------
# 模拟 100 个样本，每个样本 5 个特征，共 3 个分类（0, 1, 2）
X = torch.randn(100, 5)
y = torch.randint(0, 3, (100,))

test_loader = DataLoader(TensorDataset(X, y), batch_size=16)

# 定义极简分类模型
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 3)  # 输入5特征，输出3类别logits
    def forward(self, x):
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleClassifier().to(device)

# ----------------- 2. 测试集推理与收集 -----------------
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播与预测
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)  # 取概率最大的类别索引
        
        # 收集每个 batch 的结果
        all_preds.append(preds)
        all_targets.append(targets)

# ----------------- 3. 拼接并转为 NumPy 数组 -----------------
# 把多 batch 的 Tensor 列表拼接成单个一维 Tensor，再转 NumPy
y_pred = torch.cat(all_preds).cpu().numpy()
y_true = torch.cat(all_targets).cpu().numpy()

# ----------------- 4. sklearn 评估计算 -----------------
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_true, y_pred, average='weighted'):.4f}\n")

print("=== 分类评估报告 ===")
print(classification_report(y_true, y_pred, target_names=['类别0', '类别1', '类别2']))

print("=== 混淆矩阵 ===")
print(confusion_matrix(y_true, y_pred))
