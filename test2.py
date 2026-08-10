import torch 
import torch.nn as nn 
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(42)
    
    x = torch.randn(1000, 1)
    y = 2 * x + 1
    
    model = MyModel()


    epochs = 50
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if((epoch + 1) % 10 == 0):
            print(f"epoch: {epoch + 1: 3d} | loss: {loss: .4f}")
    
    model.eval()
    with torch.no_grad():
        test_data = torch.randn(10, 1)
        test_output_predict = model(test_data)

        print(f"{'x':>8} | {'y_pred':>8} | {'y_true':>8}")
        for x_i, y_pred_i in zip(test_data, test_output_predict):
            x_val = x_i.item()
            y_true = 2 * x_val + 1
            print(f"{x_val:8.4f} | {y_pred_i.item():8.4f} | {y_true:8.4f}")

        w = model.fc.weight.item()
        b = model.fc.bias.item()

        print(f"w: {w:.4f}, b: {b:.4f}")
    
    
    
