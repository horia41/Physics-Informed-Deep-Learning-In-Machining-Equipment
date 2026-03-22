import torch
import torch.nn as nn
import numpy as np
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import LEARNING_RATE, N_EPOCHS, PATIENCE, MODELS_DIR

def train(model, train_dataloader, val_dataloader, device, optimizer_name='Adam'):
    
    model = model.to(device)
    
    # loss function - MAE for regression
    loss_function = nn.L1Loss()
    
    # optimizer
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # tracking
    loss_train, loss_val = [], []
    mae_train, mae_val   = [], []
    
    best_val_loss = float('inf')
    patience = 0
    best_model = None

    for epoch in range(N_EPOCHS):
        
        # ----------------------------
        # TRAINING
        # ----------------------------
        model.train()
        epoch_train_loss = 0.0
        epoch_train_mae  = 0.0

        for images, wear in train_dataloader:
            images, wear = images.to(device), wear.to(device)

            # forward pass
            predictions = model(images)
            loss = loss_function(predictions, wear)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track metrics
            epoch_train_loss += loss.item()
            epoch_train_mae  += torch.mean(torch.abs(predictions - wear)).item()

        # average over all batches
        loss_train.append(epoch_train_loss / len(train_dataloader))
        mae_train.append(epoch_train_mae  / len(train_dataloader))

        # ----------------------------
        # VALIDATION
        # ----------------------------
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mae  = 0.0

        with torch.no_grad():
            for images, wear in val_dataloader:
                images, wear = images.to(device), wear.to(device)

                predictions = model(images)
                loss = loss_function(predictions, wear)

                epoch_val_loss += loss.item()
                epoch_val_mae  += torch.mean(torch.abs(predictions - wear)).item()

        loss_val.append(epoch_val_loss / len(val_dataloader))
        mae_val.append(epoch_val_mae  / len(val_dataloader))

        # print progress every 5 epochs
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {loss_train[-1]:.2f} | Train MAE: {mae_train[-1]:.2f} µm | "
                  f"Val Loss: {loss_val[-1]:.2f} | Val MAE: {mae_val[-1]:.2f} µm")

        # ----------------------------
        # EARLY STOPPING
        # ----------------------------
        if loss_val[-1] < best_val_loss:
            best_val_loss = loss_val[-1]
            best_model = copy.deepcopy(model)
            patience = 0
        else:
            patience += 1

        if patience >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

        scheduler.step(loss_val[-1])

    # save best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "best_model.pth"
    torch.save(best_model, save_path)
    print(f"Best model saved to {save_path}")

    return {
        'loss_train': loss_train,
        'loss_val':   loss_val,
        'mae_train':  mae_train,
        'mae_val':    mae_val,
        'best_model': best_model
    }