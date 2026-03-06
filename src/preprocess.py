import music21
import glob
import os
import pickle
import numpy as np

SEQUENCE_LENGTH = 100

# ---------------------------------------------------
# Extract notes
# ---------------------------------------------------

def extract_notes(dataset_path="dataset/maestro-v3.0.0"):

    notes = []

    midi_files = glob.glob(os.path.join(dataset_path, "**/*.midi"), recursive=True)

    # limit dataset for faster processing
    midi_files = midi_files[:100]

    print(f"Found {len(midi_files)} MIDI files")

    for i, file in enumerate(midi_files):

        if i % 10 == 0:
            print(f"Processing file {i}/{len(midi_files)}")

        try:
            midi = music21.converter.parse(file)

            for element in midi.flatten().notes:

                if isinstance(element, music21.note.Note):
                    notes.append(str(element.pitch))

                elif isinstance(element, music21.chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        except Exception as e:
            print("Skipping problematic file:", file)

    print("Total notes extracted:", len(notes))
    print("Example notes:", notes[:20])

    return notes


# ---------------------------------------------------
# Convert notes into sequences
# ---------------------------------------------------

def prepare_sequences(notes):

    pitchnames = sorted(set(notes))

    note_to_int = {note: number for number, note in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(len(notes) - SEQUENCE_LENGTH):

        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]

        network_input.append([note_to_int[n] for n in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    n_vocab = len(pitchnames)

    print("Total patterns:", n_patterns)
    print("Vocabulary size:", n_vocab)

    network_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    network_input = network_input / float(n_vocab)

    return network_input, network_output


# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------

if __name__ == "__main__":

    os.makedirs("dataset", exist_ok=True)

    # Extract notes
    notes = extract_notes()

    # Save notes
    with open("dataset/notes.pkl", "wb") as f:
        pickle.dump(notes, f)

    print("Notes saved")

    # Prepare sequences
    network_input, network_output = prepare_sequences(notes)

    with open("dataset/network_input.pkl", "wb") as f:
        pickle.dump(network_input, f)

    with open("dataset/network_output.pkl", "wb") as f:
        pickle.dump(network_output, f)

    print("Training sequences saved")
    print("Input shape:", network_input.shape)