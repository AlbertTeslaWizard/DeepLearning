import math
import yaml 
import torch 
from models import RegularizedModel

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegularizedModel().to(device)
    model.load_state_dict(torch.load(cfg["save_path"], map_location=device, weights_only=True))
    model.eval()
    
    test_x = torch.rand(10, 1)
    test_y_true = torch.sin(2 * math.pi * test_x) + 1

    with torch.no_grad():
        for x, y in zip(test_x, test_y_true):
            x = x.unsqueeze(0).to(device)
            y_pred = model(x)

            print(f"x: {x.item():.4f} | y_pred: {y_pred.item():.4f} | y_true: {y.item():.4f}")

if __name__ == '__main__':
    main()
