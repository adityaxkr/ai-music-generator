# 🎵 AI Music Generator

An AI-powered music generation system built using **PyTorch LSTM** and **Streamlit** that generates musical melodies based on user-selected mood and instrument.

---

## 🚀 Features

- 🎶 AI-generated music using deep learning
- 🎭 Mood-based music generation
- 🎻 Multiple instrument support
- 🌐 Interactive Streamlit web interface
- 🎼 MIDI music output

---

## 🧠 Model

The project uses a **Long Short-Term Memory (LSTM)** neural network trained on MIDI datasets to learn musical patterns and generate new melodies.

Model Architecture:

- 2 Layer LSTM
- Hidden Size: 512
- Softmax Output Layer
- Temperature Sampling for diversity

---

## 🛠 Tech Stack

- Python
- PyTorch
- Streamlit
- music21
- NumPy

---
ai-music-generator
│
├── app.py # Streamlit web app
├── src
│ ├── preprocess.py
│ ├── train_model.py
│ ├── generate_music.py
│
├── requirements.txt
└── README.md


---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ai-music-generator.git
cd ai-music-generator

Install dependencies
pip install -r requirements.txt

Start the streamlit application
streamlit run app.py

🎹 How It Works

The LSTM model is trained on MIDI files.

It learns musical sequences and note patterns.

During generation, it predicts the next note based on previous notes.

Generated notes are converted into MIDI format.

Example Output
The model generates melodies that can be played in any MIDI player or digital audio workstation.

## 📂 Project Structure
