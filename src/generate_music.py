import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import music21

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQUENCE_LENGTH = 100
TEMPERATURE = 0.8
GENERATE_LENGTH = 1000

# -----------------------------
# Load notes
# -----------------------------

with open("dataset/notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitchnames = sorted(set(notes))
n_vocab = len(pitchnames)

note_to_int = {note: number for number, note in enumerate(pitchnames)}
int_to_note = {number: note for number, note in enumerate(pitchnames)}

# -----------------------------
# Load input sequences
# -----------------------------

with open("dataset/network_input.pkl", "rb") as f:
    network_input = pickle.load(f)

network_input = torch.tensor(network_input, dtype=torch.float32)

# -----------------------------
# Model Definition
# -----------------------------

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

        out = out[:, -1, :]

        out = self.fc(out)

        return out


model = MusicLSTM(n_vocab).to(device)

model.load_state_dict(torch.load("models/music_model.pth", map_location=device))

model.eval()

# -----------------------------
# Choose seed pattern
# -----------------------------

start = np.random.randint(0, 1000)

pattern = network_input[start]

pattern = pattern.unsqueeze(0).to(device)

generated_notes = []

print("Generating music...")

# -----------------------------
# Generate notes
# -----------------------------

for i in range(GENERATE_LENGTH):

    prediction = model(pattern)

    prediction = prediction / TEMPERATURE

    prediction = F.softmax(prediction, dim=1)

    index = torch.multinomial(prediction, 1).item()

    result = int_to_note[index]

    generated_notes.append(result)

    index = index / float(n_vocab)

    new_input = torch.tensor([[[index]]], dtype=torch.float32).to(device)

    pattern = torch.cat((pattern[:, 1:, :], new_input), dim=1)

# -----------------------------
# Convert to MIDI
# -----------------------------

offset = 0
output_notes = []

for pattern in generated_notes:

    if '.' in pattern or pattern.isdigit():

        notes_in_chord = pattern.split('.')

        notes = []

        for current_note in notes_in_chord:

            new_note = music21.note.Note(int(current_note))

            new_note.storedInstrument = music21.instrument.Piano()

            notes.append(new_note)

        new_chord = music21.chord.Chord(notes)

        new_chord.offset = offset

        output_notes.append(new_chord)

    else:

        new_note = music21.note.Note(pattern)

        new_note.offset = offset

        new_note.storedInstrument = music21.instrument.Piano()

        output_notes.append(new_note)

    offset += 0.5


midi_stream = music21.stream.Stream(output_notes)

midi_stream.write("midi", fp="generated_music.mid")

print("Music generated successfully!")
print("File saved as: generated_music1.mid")