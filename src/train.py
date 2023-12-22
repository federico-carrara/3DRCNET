import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from tqdm import tqdm
from typing import Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: Optional[int] = 50,
    learning_rate: Optional[float] = 0.001,
    log_dir: Optional[str] = None,
    log_interval: Optional[int] = 5
) -> None:
    
    training_history = defaultdict(list)
    best_val_accuracy = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0

        for inputs, labels in tqdm(train_loader, desc=f"Training step"):
            inputs, labels = inputs.to(device).to(torch.float32), labels.to(device).to(torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predicted_labels = (outputs > 0.5).float().squeeze()
            correct_predictions += (predicted_labels == labels).sum().item()

        accuracy = correct_predictions / len(train_loader.dataset)
        avg_loss = total_loss / len(train_loader)
        training_history["train_accuracy"].append(accuracy)
        training_history["train_loss"].append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0

        with torch.no_grad():
            for val_inputs, val_labels in tqdm(val_loader, desc=f"Validation step"):
                val_inputs, val_labels = val_inputs.to(device).to(torch.float32), val_labels.to(device).to(torch.float32)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs.squeeze(), val_labels.float()).item()
                val_predicted_labels = (val_outputs > 0.5).float().squeeze()
                val_correct_predictions += (val_predicted_labels == val_labels).sum().item()

        val_accuracy = val_correct_predictions / len(val_loader.dataset)
        avg_val_loss = val_loss / len(val_loader)
        training_history["val_accuracy"].append(val_accuracy)
        training_history["val_loss"].append(avg_val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Overwrite training history
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "training_history.json"), 'w') as json_file:
            json.dump(training_history, json_file)

        # Save model checkpoint
        if log_dir is not None and (epoch + 1) % log_interval == 0 and val_accuracy > best_val_accuracy:
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model_ckpt.pt"))
            print(f"Saved checkpoint at epoch {epoch + 1}")
        best_val_accuracy = val_accuracy if val_accuracy > best_val_accuracy else best_val_accuracy
