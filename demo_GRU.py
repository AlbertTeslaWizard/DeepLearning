import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size = 1,
            hidden_size = 32,
            batch_first = True 
        )

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, h_n = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out

if __name__ == '__main__':
    t = torch.linspace(0, 100, 2000)
    data = torch.sin(t)

    seq_len = 20
    X = []
    y = []

    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])

    X = torch.stack(X).unsqueeze(-1)
    y = torch.stack(y).unsqueeze(-1)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    train_dataset = TensorDataset(X, y)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = 64,
        shuffle = True
    )

    model = SimpleGRU().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr = 0.001)
    epochs = 20

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X)

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch: {epoch + 1: 2d} | loss: {avg_loss: .4f}")

    model.eval()
    with torch.no_grad():
        x = X[-1].unsqueeze(0).to(device)
        pred = model(x)

        print("预测值:", pred.item())
        print("真实值:", y[-1].item())
