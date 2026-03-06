import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LENGTH = 100
BATCH_SIZE = 128
EPOCHS = 40
EMBED_SIZE = 256
HEADS = 8
LAYERS = 6

print("Using device:", device)

# ---------------------------
# Load dataset
# ---------------------------

with open("dataset/network_input.pkl", "rb") as f:
    X = pickle.load(f)

with open("dataset/network_output.pkl", "rb") as f:
    y = pickle.load(f)

X = X[:200000]
y = y[:200000]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

vocab_size = int(torch.max(y).item()) + 1

print("Training samples:", X.shape[0])
print("Vocabulary size:", vocab_size)

# ---------------------------
# Dataset
# ---------------------------

class MusicDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = MusicDataset(X, y)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------
# Transformer Model
# ---------------------------

class MusicTransformer(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBED_SIZE)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_SIZE,
            nhead=HEADS
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=LAYERS
        )

        self.fc = nn.Linear(EMBED_SIZE, vocab_size)

    def forward(self, x):

        x = x.squeeze(-1).long()

        x = self.embedding(x)

        x = x.permute(1,0,2)

        out = self.transformer(x)

        out = out[-1]

        out = self.fc(out)

        return out


model = MusicTransformer(vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# ---------------------------
# Training
# ---------------------------

for epoch in range(EPOCHS):

    total_loss = 0

    for batch_X, batch_y in loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.2f}")

torch.save(model.state_dict(), "models/music_transformer.pth")

print("Transformer model saved!")