import os
import subprocess
import whisper
from transformers import pipeline
import torch
import difflib
import re
from pydub import AudioSegment, effects
from rouge_score import rouge_scorer
import bert_score
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Convert to WAV (mono, 16kHz)
def convert_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + ".wav"
    command = ['ffmpeg', '-i', input_path, '-ac', '1', '-ar', '16000', output_path, '-y']
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Converted {input_path} to {output_path}")
        return output_path
    except subprocess.CalledProcessError:
        print(f"Failed to convert {input_path} to WAV format.")
        return ""

# Audio normalization using PyDub
def preprocess_audio(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_clean.wav"
    audio = AudioSegment.from_file(input_path)
    audio = effects.normalize(audio)
    audio.export(output_path, format="wav")
    print(f"Preprocessed (normalized) audio saved to {output_path}")
    return output_path

# Whisper transcription
def transcribe_audio(audio_path: str) -> str:
    model = whisper.load_model("small")
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path)
    return result["text"]

# Extractive summarization using TextRank
def extractive_summarization(text: str, max_sentences=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, max_sentences)
    return ' '.join(str(sentence) for sentence in summary)

# Abstractive summarization using BART
def abstractive_summarization(text: str, chunk_size: int = 1000) -> str:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    summaries = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return ' '.join(summaries)

# Save transcription and summaries
def save_to_file(transcription: str, summary: str, output_path: str) -> bool:
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Transcription:\n")
            f.write(transcription)
            f.write("\n\nSummary:\n")
            f.write(summary)
        print(f"Results saved to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save results: {e}")
        return False

# Clean text preprocessing
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# WER calculation
def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    d = [[0] * (len(hyp_words)+1) for _ in range(len(ref_words)+1)]
    for i in range(len(ref_words)+1):
        d[i][0] = i
    for j in range(len(hyp_words)+1):
        d[0][j] = j
    for i in range(1, len(ref_words)+1):
        for j in range(1, len(hyp_words)+1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer

# Character-level accuracy, WER, ROUGE, BERTScore
def calculate_accuracy(transcription: str, reference_file: str) -> float:
    try:
        with open(reference_file, 'r', encoding='utf-8') as file:
            reference_text = file.read()
        transcription_clean = preprocess_text(transcription)
        reference_clean = preprocess_text(reference_text)

        sequence_matcher = difflib.SequenceMatcher(None, transcription_clean, reference_clean)
        accuracy = sequence_matcher.ratio() * 100
        wer = calculate_wer(reference_clean, transcription_clean) * 100

        print(f"Character-level Accuracy: {accuracy:.2f}%")
        print(f"Word Error Rate (WER): {wer:.2f}%")

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_text, transcription)
        for key in scores:
            print(f"{key}: {scores[key].fmeasure:.4f}")

        P, R, F1 = bert_score.score([transcription], [reference_text], lang="en", rescale_with_baseline=True)
        print(f"BERTScore F1: {F1.mean():.4f}")

        return accuracy
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return 0.0

# Process a single file end-to-end
def process_file(file_path: str, output_dir: str, reference_folder: str) -> bool:
    file_path = os.path.abspath(file_path)
    file_name = os.path.basename(file_path)
    file_extension = os.path.splitext(file_path)[1].lower()

    print(f"\nProcessing: {file_name}")

    supported_audio_formats = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    supported_formats = supported_audio_formats + supported_video_formats

    if file_extension not in supported_formats:
        print(f"Unsupported file format: {file_extension}.")
        return False

    if file_extension == '.wav':
        audio_path = file_path
    else:
        print(f"Converting {file_extension} file to WAV format...")
        audio_path = convert_to_wav(file_path)
        if not audio_path:
            return False

    preprocessed_audio = preprocess_audio(audio_path)

    transcription = transcribe_audio(preprocessed_audio)
    if not transcription:
        return False
    print(f"Transcription completed: {len(transcription)} characters")

    extractive_summary = extractive_summarization(transcription)
    abstractive_summary = abstractive_summarization(transcription)
    combined_summary = f"Extractive Summary:\n{extractive_summary}\n\nAbstractive Summary:\n{abstractive_summary}"

    output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_results.txt")
    if not save_to_file(transcription, combined_summary, output_file):
        return False

    reference_file = os.path.join(reference_folder, f"{os.path.splitext(file_name)[0]}.txt")
    if os.path.exists(reference_file):
        calculate_accuracy(transcription, reference_file)
    else:
        print(f"Reference file not found for {file_name}")

    return True

# Process all files in a folder
def process_folder(folder_path: str, reference_folder: str) -> None:
    print(f">>\nScanning folder: {folder_path}")
    media_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print(f"Found {len(media_files)} media file(s):")
    for f in media_files:
        print(f"- {f}")

    processed_count = 0
    for file_name in media_files:
        file_path = os.path.join(folder_path, file_name)
        if process_file(file_path, folder_path, reference_folder):
            processed_count += 1

    print(f"\nProcessing complete: {processed_count}/{len(media_files)} files successfully processed.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = script_dir
    reference_folder = script_dir 

    print(f"Using folder: {folder_path}")
    process_folder(folder_path, reference_folder)

