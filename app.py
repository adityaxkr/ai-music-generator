import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import music21
import os

# ---------------------------------------------------
# Page title
# ---------------------------------------------------

st.title("🎵 AI Music Generator")
st.write("Generate AI music using a trained LSTM model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------
# User Controls
# ---------------------------------------------------

mood = st.selectbox(
    "Choose Mood",
    ["Happy", "Sad", "Relaxed", "Energetic"]
)

instrument_name = st.selectbox(
    "Choose Instrument",
    ["Piano", "Guitar", "Violin", "Flute"]
)

length = st.slider(
    "Music Length (notes)",
    200,
    1000,
    500
)

# ---------------------------------------------------
# Mood temperature settings
# ---------------------------------------------------

mood_temp = {
    "Happy": 0.9,
    "Sad": 0.6,
    "Relaxed": 0.7,
    "Energetic": 1.0
}

TEMPERATURE = mood_temp[mood]

# ---------------------------------------------------
# Instrument mapping
# ---------------------------------------------------

instrument_map = {
    "Piano": music21.instrument.Piano(),
    "Guitar": music21.instrument.AcousticGuitar(),
    "Violin": music21.instrument.Violin(),
    "Flute": music21.instrument.Flute()
}

instrument = instrument_map[instrument_name]

# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------

@st.cache_resource
def load_data():

    with open("dataset/notes.pkl", "rb") as f:
        notes = pickle.load(f)

    pitchnames = sorted(set(notes))

    n_vocab = len(pitchnames)

    int_to_note = {i: n for i, n in enumerate(pitchnames)}

    with open("dataset/network_input.pkl", "rb") as f:
        network_input = pickle.load(f)

    network_input = torch.tensor(network_input, dtype=torch.float32)

    return network_input, int_to_note, n_vocab


network_input, int_to_note, n_vocab = load_data()

# ---------------------------------------------------
# Model
# ---------------------------------------------------

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


@st.cache_resource
def load_model():

    model = MusicLSTM(n_vocab).to(device)

    model.load_state_dict(
        torch.load("models/music_model.pth", map_location=device)
    )

    model.eval()

    return model


model = load_model()

# ---------------------------------------------------
# Generate Music
# ---------------------------------------------------

if st.button("Generate Music 🎶"):

    start = np.random.randint(0, len(network_input) - 1)

    pattern = network_input[start].unsqueeze(0).to(device)

    generated_notes = []

    st.write("Generating music...")

    for i in range(length):

        prediction = model(pattern)

        prediction = prediction / TEMPERATURE

        probs = F.softmax(prediction, dim=1)

        index = torch.multinomial(probs, 1).item()

        generated_notes.append(int_to_note[index])

        index = index / float(n_vocab)

        new_input = torch.tensor(
            [[[index]]],
            dtype=torch.float32
        ).to(device)

        pattern = torch.cat((pattern[:, 1:, :], new_input), dim=1)

    # ---------------------------------------------------
    # Convert to MIDI
    # ---------------------------------------------------

    offset = 0
    output_notes = []

    for pattern in generated_notes:

        if '.' in pattern or pattern.isdigit():

            notes_in_chord = pattern.split('.')

            notes = []

            for current_note in notes_in_chord:

                new_note = music21.note.Note(int(current_note))

                new_note.storedInstrument = instrument

                notes.append(new_note)

            new_chord = music21.chord.Chord(notes)

            new_chord.offset = offset

            new_chord.storedInstrument = instrument

            output_notes.append(new_chord)

        else:

            new_note = music21.note.Note(pattern)

            new_note.offset = offset

            new_note.storedInstrument = instrument

            output_notes.append(new_note)

        offset += 0.5

    midi_stream = music21.stream.Stream(output_notes)

    os.makedirs("outputs", exist_ok=True)

    filepath = "outputs/generated_music.mid"

    midi_stream.write("midi", fp=filepath)

    st.success("Music Generated!")

    with open(filepath, "rb") as f:

        st.download_button(
            "Download MIDI 🎵",
            f,
            file_name="ai_music.mid"
        )