import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=10):
        super(SimpleClassifier, self).__init__()
        # Lapisan Linear sederhana: Input Embedding -> Probabilitas Kelas
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 128)
        return self.fc(x)

# Helper untuk save/load
def save_head(model, path="local_head.pth"):
    torch.save(model.state_dict(), path)

def load_head(path="local_head.pth", num_classes=10):
    model = SimpleClassifier(num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(path))
        print(f"[MODEL] Loaded head from {path}")
    except FileNotFoundError:
        print("[MODEL] No existing head found, creating new one initialized randomly.")
    return model