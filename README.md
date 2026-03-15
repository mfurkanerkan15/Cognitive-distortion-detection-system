# Cognitive Distortion Analysis System (Turkish NLP)

This project features a two-stage hierarchical deep learning system designed to detect and classify **cognitive distortions** in Turkish text. It aims to support mental health awareness by identifying biased thinking patterns.

## 🚀 System Architecture
The system is divided into two specialized stages to ensure high precision:

1. **Stage 1: The Detector (Gatekeeper)** A binary classifier that determines if a sentence contains any cognitive distortion.
   - **Model:** [Hugging Face - Stage 1 Detector](https://huggingface.co/mfurkanerkan15/cognitive-distortion-detector-tr)

2. **Stage 2: The Classifier (Specialist)** If the Detector identifies a distortion, this multi-class model categorizes it into specific types.
   - **Model:** [Hugging Face - Stage 2 Classifier](https://huggingface.co/mfurkanerkan15/cognitive-distortion-classifier-tr)


## 🛠 Features
- **Turkish Support:** Specifically trained on Turkish datasets for high-accuracy NLP tasks.
- **Web App:** A real-time inference interface built with **Streamlit**.
