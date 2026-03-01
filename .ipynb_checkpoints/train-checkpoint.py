import os 
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import build_model

# function: build_optim
# uses SGD, Adam, and AdamW as the optimisers
def build_optim(model, optim_name="sgd", lr=0.1, wd=1e-4):
    if optim_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif optim_name == "adam":
        return optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    elif optim_name == "adamw":
        return optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)

# function: build_scheduler
# uses either cosine or step scheduling to control how learning rate changes over training
def build_scheduler(optim, scheduler="cosine", num_epochs=30):
    if scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)
    elif scheduler == "step":
        # drops by 10x at epoch 10 and 20
        return optim.lr_scheduler.MultStepLR(optim, milestones=[10, 20], gamma=0.1)

# function: train_epoch
# training loop that iterates over number of epochs
def train_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        # clear gradients from previous batch
        optim.zero_grad()
        # produce raw images for forward pass
        outputs = model(images)
        # compute cross-entropy loss
        loss = criterion(outputs, labels)
        # backpropagation step
        loss.backward()
        # update each parameter
        optim.step()
        # compute average loss per sample
        total_loss += loss.item() * images[0]
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images[0]
    return total_loss / total, correct / total

# function: eval_epoch
# evaluates on validation data
# disable gradient computation (reduces computational load)
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="Val ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()
        total_loss += loss.item() * images[0]
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images[0]
    return total_loss / total, correct / total

# function: run_training
def run_training(trainer_loader, val_loader, architecture="resnet18", optim="sgd", schedule="cosine",
                 norm="batch", dropout=0.0, num_epochs=30, save_name="model", device="cuda"):
    os.makedirs('./checkpoints', exist_ok=True)
    model = build_model(architecture, norm, dropout).to(device)
    optimiser = build_optim(model, optim)
    scheduler = build_scheduler(optimiser, schedule, num_epochs)
    criterion = nn.CrossEntropyLoss()

    history  = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_epoch(model, trainer_loader, optimiser, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        # scheduler called once per epoch
        if scheduler:
            scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f'Epoch [{epoch:02d}/{num_epochs}]  '
              f'Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  '
              f'Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            # saves model weights when validation accuracy is good
            torch.save(model.state_dict(), f'./checkpoints/best_{save_name}.pth')

    print(f'\nBest Val Accuracy: {best_acc:.4f}')
    # return both model and history so graphs can be plot without reloading from disk
    return model, history