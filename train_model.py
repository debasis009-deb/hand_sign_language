"""
Train Sign Language CNN on Sign MNIST Dataset
==============================================
Uses sign_mnist_train.csv and sign_mnist_test.csv to train a CNN model.

Usage:
  python train_model.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import time
from model import build_model

TRAIN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sign_mnist_train.csv')
TEST_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sign_mnist_test.csv')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sign_model.pth')

# Sign MNIST labels: 0-24 (9=J and 25=Z are excluded as they need motion)
# We map these to consecutive indices 0-23 for training
LABEL_MAP = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8,
             10:9, 11:10, 12:11, 13:12, 14:13, 15:14, 16:15,
             17:16, 18:17, 19:18, 20:19, 21:20, 22:21, 23:22, 24:23}


def load_data():
    """Load Sign MNIST data from CSV files."""
    print("[INFO] Loading training data...")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"[INFO] Training samples: {len(train_df)}")
    
    print("[INFO] Loading test data...")
    test_df = pd.read_csv(TEST_CSV)
    print(f"[INFO] Test samples: {len(test_df)}")
    
    # Extract labels and pixels
    y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values.astype(np.float32)
    
    y_test = test_df['label'].values
    X_test = test_df.drop('label', axis=1).values.astype(np.float32)
    
    # Map labels to consecutive indices (0-23)
    y_train = np.array([LABEL_MAP[l] for l in y_train], dtype=np.int64)
    y_test = np.array([LABEL_MAP[l] for l in y_test], dtype=np.int64)
    
    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Reshape to (N, 1, 28, 28) for CNN
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    
    print(f"[INFO] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[INFO] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"[INFO] Number of classes: {len(set(y_train.tolist()))}")
    
    return X_train, y_train, X_test, y_test


def train():
    """Train the CNN model."""
    X_train, y_train, X_test, y_test = load_data()
    
    # Create DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")
    
    model = build_model(num_classes=24)  # 24 classes (0-23 mapped)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    epochs = 30
    best_test_acc = 0.0
    
    print(f"\n{'='*65}")
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Train Acc':>10} | {'Test Acc':>10} | {'Time':>6}")
    print(f"{'='*65}")
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # --- Training ---
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        train_loss = total_loss / total
        train_acc = correct / total
        
        # --- Testing ---
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == batch_y).sum().item()
                test_total += batch_y.size(0)
        
        test_acc = test_correct / test_total
        elapsed = time.time() - start_time
        
        scheduler.step(test_acc)
        
        print(f"{epoch:>6} | {train_loss:>11.4f} | {train_acc:>9.1%} | {test_acc:>9.1%} | {elapsed:>5.1f}s")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"       -> New best model saved! ({test_acc:.1%})")
    
    print(f"\n{'='*65}")
    print(f"[DONE] Best test accuracy: {best_test_acc:.1%}")
    print(f"[DONE] Model saved to: {MODEL_PATH}")
    print(f"\nRun 'python app.py' to start the web app with the trained model!")


if __name__ == '__main__':
    train()
