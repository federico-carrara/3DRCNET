import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from typing import Dict

def evaluate_model(
    model: torch.nn.Module, 
    test_loader: torch.utils.data.DataLoader
) -> Dict[str, float]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test step"):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(torch.float32)
            outputs = model(inputs)
            predictions = (outputs > 0.5).float().squeeze()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return {
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
