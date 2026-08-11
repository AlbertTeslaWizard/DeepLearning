import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# ==================== 1. 配置标准日志 (Logging) ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log", encoding="utf-8"),  # 保存到本地文件
        logging.StreamHandler()                                 # 同时输出到控制台
    ]
)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(42)
    model = MyModel()

    # ==================== 2. 初始化 TensorBoard ====================
    # 日志会存放在 runs/exp_tanh 文件夹下
    writer = SummaryWriter(log_dir="runs/exp_tanh")

    epochs = 1000
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)

    X = torch.randn(1000, 1)
    Y = X * X + 1

    # 把模型结构画到 TensorBoard 里
    writer.add_graph(model, X)

    logging.info("开始模型训练...")

    for epoch in range(epochs):
        y_pred = model(X)
        loss = loss_fn(y_pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 写入 TensorBoard（类别/指标名, Y轴数值, X轴数值）
        writer.add_scalar("Loss/train", loss.item(), epoch)

        # 每 100 轮打一次日志
        if (epoch + 1) % 100 == 0:
            logging.info(f"Epoch: {epoch + 1:4d} | MSE Loss: {loss.item():.6f}")

    # 训练结束，关闭 Writer
    writer.close()
    logging.info("训练完成，日志与 TensorBoard 数据已保存！")

    # ==================== 测试评估 ====================
    test_x_data = torch.randn(10, 1)
    test_y_true = test_x_data ** 2 + 1

    model.eval()
    with torch.no_grad():
        for x, y in zip(test_x_data, test_y_true):
            test_y_pred = model(x)
            logging.info(f"Test -> x: {x.item(): .4f} | y_pred: {test_y_pred.item(): .4f} | y_true: {y.item(): .4f}")
