![Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/Uses-OpenAI%20Whisper-FF6F00?style=for-the-badge&logo=openai&logoColor=white)
![Transformers](https://img.shields.io/badge/NLP-Transformers-9cf?style=for-the-badge&logo=huggingface&logoColor=black)
![Git](https://img.shields.io/badge/Version%20Control-Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![Status](https://img.shields.io/badge/Project-Open%20Source-orange?style=for-the-badge)

# 🎙️ AI-Powered Lecture Transcription & Summarization

A powerful Python-based tool that leverages OpenAI’s Whisper model for automatic transcription of lectures or any spoken content, followed by both extractive (TextRank) and abstractive (Transformer) summarization. It evaluates transcription quality using Word Error Rate (WER) and summary quality using ROUGE and BERTScore metrics. This is a one-stop pipeline for turning long audio lectures into concise, meaningful insights.

## 🚀 Features
- 🎧 Converts audio/video files (MP3/WAV/MP4/MKV) to text using Whisper
- 🔉 Normalizes and cleans audio using PyDub and FFmpeg
- 🧠 Generates both extractive and abstractive summaries
- 📊 Evaluates using WER, ROUGE (1 & L), and BERTScore (F1)
- ⚙️ All-in-one Python script (Project.py) with modular functions

## 📊 Model Performance Scores
| Metric              | Score   |
|---------------------|---------|
| Word Error Rate     | 0.182   |
| ROUGE-1             | 0.57    |
| ROUGE-L             | 0.53    |
| BERTScore (F1)      | 0.72    |

## 🔧 Prerequisites
Ensure you have Python 3.8+ installed. Install all required Python libraries using:
```
pip install openai-whisper transformers torch pydub rouge-score bert-score sumy
```
Also make sure ffmpeg is installed and available in your system PATH.

Install FFmpeg:

Windows: Download from ffmpeg.org and add to PATH

Linux: sudo apt install ffmpeg

macOS: brew install ffmpeg

📁 Project Structure
```
lecture-transcription-summary/
├── Project.py               # Main script for full processing
├── .venv/                   # (Optional) Virtual environment
├── README.md                # This file
```
▶️ How to Run
Clone the repository:
```
git clone https://github.com/yourusername/lecture-transcription-summary.git
cd lecture-transcription-summary
```
Place your audio file (MP3/WAV) in the same directory.

Run the main script:
```
python Project.py
```
You’ll receive:

📝 Full transcript in console

📄 Extractive and abstractive summaries

📈 Evaluation scores printed and/or saved

### 🛠️ Libraries Used:
- openai-whisper  
- transformers  
- torch  
- pydub  
- rouge-score  
- bert-score  
- sumy



🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

📫 Contact
Created with ❤️ by [Harshit Dhiman]. Connect on LinkedIn(https://www.linkedin.com/in/harshit3209) or visit GitHub.
