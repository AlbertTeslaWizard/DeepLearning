import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models import RegularizedModel
from datasets import SineDataset
from utils import EarlyStop
from torch.nn.utils import clip_grad_norm_

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])

    full_dataset = SineDataset(num_samples=cfg["num_samples"])
    train_size = int(len(full_dataset) * cfg["train_ratio"])
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = cfg["batch_size"],
        shuffle = True
    )

    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = cfg["batch_size"],
        shuffle = False 
    )
    
    model = RegularizedModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["T_max"], eta_min=cfg["eta_min"])
    earlystop = EarlyStop(patience=cfg["patience"], save_path=cfg["save_path"])

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0

        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()

            if cfg["max_grad_norm"]:
                clip_grad_norm_(
                    model.parameters(),
                    max_norm = cfg["max_grad_norm"],
                    norm_type = 2.0
                )
                
            optimizer.step()

            train_loss += loss.item() * len(x)
        avg_train_loss = train_loss / train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)

                val_loss += loss.item() * len(x)
            avg_val_loss = val_loss / val_size

        scheduler.step()
        earlystop(avg_val_loss, model)

        if earlystop.is_earlystop:
            print(f"\n连续 {earlystop.patience} 轮 val_loss 无改善，触发早停")
            break
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | lr: {current_lr:.6f}")

if __name__ == '__main__':
    main()






