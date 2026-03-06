import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# Device Setup
# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print("Using device:", device)

# ------------------------------
# Load Dataset
# ------------------------------

print("Loading dataset...")

with open("dataset/network_input.pkl", "rb") as f:
    X = pickle.load(f)

with open("dataset/network_output.pkl", "rb") as f:
    y = pickle.load(f)

# Reduce dataset for faster training
X = X[:150000]
y = y[:150000]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

vocab_size = int(torch.max(y).item()) + 1

print("Training samples:", X.shape[0])
print("Vocabulary size:", vocab_size)

# ------------------------------
# Dataset Class
# ------------------------------

class MusicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = MusicDataset(X, y)

# Optimized DataLoader
loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# ------------------------------
# LSTM Model
# ------------------------------

class MusicLSTM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):

        out, _ = self.lstm(x)

        # take last time step
        out = out[:, -1, :]

        out = self.fc(out)

        return out


model = MusicLSTM(vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# Training Loop
# ------------------------------

epochs = 20

print("Starting training...")

for epoch in range(epochs):

    total_loss = 0

    for batch_X, batch_y in loader:

        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

# ------------------------------
# Save Model
# ------------------------------

torch.save(model.state_dict(), "models/music_model.pth")

print("Model saved to models/music_model.pth")