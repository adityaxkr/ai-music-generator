import pickle
import torch
import torch.nn.functional as F
import numpy as np
import music21
from train_transformer import MusicTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMPERATURE = 0.8
GENERATE_LENGTH = 1000

with open("dataset/notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitchnames = sorted(set(notes))

int_to_note = {i: n for i,n in enumerate(pitchnames)}

vocab_size = len(pitchnames)

model = MusicTransformer(vocab_size).to(device)

model.load_state_dict(torch.load("models/music_transformer.pth"))

model.eval()

seed = np.random.randint(0, vocab_size, 100)

pattern = torch.tensor(seed).unsqueeze(0).unsqueeze(-1).to(device)

generated = []

for i in range(GENERATE_LENGTH):

    prediction = model(pattern)

    prediction = prediction / TEMPERATURE

    probs = F.softmax(prediction, dim=1)

    index = torch.multinomial(probs,1).item()

    generated.append(int_to_note[index])

    new_note = torch.tensor([[[index]]], dtype=torch.float32).to(device)

    pattern = torch.cat((pattern[:,1:,:], new_note),1)

offset = 0
output_notes = []

for note in generated:

    new_note = music21.note.Note(note)
    new_note.offset = offset
    output_notes.append(new_note)

    offset += 0.5

midi_stream = music21.stream.Stream(output_notes)

midi_stream.write("midi", fp="transformer_music.mid")

print("Generated transformer_music.mid")