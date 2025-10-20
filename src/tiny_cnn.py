import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 32, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

def extract_embedding(gray_256, device='cpu'):
    model = TinyCNN(128).to(device).eval()
    # Optional: try to load weights if present
    try:
        state = torch.load('cnn_weights.pth', map_location=device)
        model.load_state_dict(state)
    except Exception:
        pass

    x = torch.from_numpy(gray_256.astype('float32')/255.0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        emb = model(x).cpu().numpy().reshape(-1)
    return emb
