import torch
import torch.nn as nn
import os

class SignLanguageCNN(nn.Module):
    """CNN model trained on Sign MNIST (28x28 grayscale hand images)."""
    def __init__(self, num_classes=25):
        super(SignLanguageCNN, self).__init__()
        # Labels 0-24 (no J=9, Z=25 as they need motion)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Label mapping: Sign MNIST labels 0-24 → letters A-Z (skipping J and Z)
LABEL_TO_LETTER = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W',
    22: 'X', 23: 'Y', 24: 'Z'
}


def build_model(num_classes=25):
    return SignLanguageCNN(num_classes=num_classes)


def load_model():
    num_classes = 24  # Sign MNIST: 24 labels (mapped to 0-23)
    model = build_model(num_classes)
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sign_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        print(f"[INFO] Loaded trained model from {model_path}")
    else:
        print(f"[WARNING] No trained model found at {model_path}")
        print(f"[WARNING] Run 'python train_model.py' to train the model first.")
    
    return model