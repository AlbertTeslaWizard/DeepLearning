import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==================== 1. 配置标准日志 (Logging) ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log", encoding="utf-8"),  # 保存到本地文件
        logging.StreamHandler()                                 # 同时输出到控制台
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device is {device}.")

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

class QuadricDataSet(Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.x = torch.randn(num_samples, 1)
        self.y = self.x * self.x + 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == '__main__':
    torch.manual_seed(42)
    model = MyModel().to(device)

    epochs = 100
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)

    train_dataset = QuadricDataSet(num_samples=2000)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = 64,
        shuffle = True
    )

    logging.info("开始模型训练...")
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            batch_y_pred = model(batch_x)
            loss = loss_fn(batch_y_pred, batch_y)
            total_loss += loss.item() * len(batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataset)

        # 每 10 轮打一次日志
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch: {epoch + 1:4d} | MSE Loss: {avg_loss:.6f}")

    logging.info("训练完成！")

    # ==================== 测试评估 ====================
    test_x_data = torch.randn(10, 1)
    test_y_true = test_x_data ** 2 + 1
    test_x_data = test_x_data.to(device)
    test_y_true = test_y_true.to(device)

    model.eval()
    with torch.no_grad():
        for x, y in zip(test_x_data, test_y_true):
            test_y_pred = model(x)
            logging.info(f"Test -> x: {x.item(): .4f} | y_pred: {test_y_pred.item(): .4f} | y_true: {y.item(): .4f}")
