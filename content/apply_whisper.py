import os
import av
import torch
import numpy as np
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

def load_and_process_audio(file_path):
    """Load and process an MP3 file using pyav."""
    container = av.open(file_path)
    stream = container.streams.audio[0]
    audio_frames = []
    for frame in container.decode(stream):
        audio_frames.append(frame.to_ndarray())

    audio = np.concatenate(audio_frames, axis=1).flatten()
    sampling_rate = stream.sample_rate

    # Resample to 16 kHz if necessary
    if sampling_rate != 16000:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), orig_freq=sampling_rate, new_freq=16000).numpy()
        sampling_rate = 16000

    return audio, sampling_rate

def transcribe_audio_to_text(file_path):
    """Transcribe audio file to text."""
    audio, sampling_rate = load_and_process_audio(file_path)
    input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

def save_transcription_to_file(transcription, file_path):
    """Save the transcription to a text file."""
    text_file_path = os.path.splitext(file_path)[0] + '.txt'
    with open(text_file_path, 'w', encoding='utf-8') as f:
        f.write(transcription)

def process_directory(directory):
    """Process all MP3 files in the given directory and its subdirectories."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.mp3'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                transcription = transcribe_audio_to_text(file_path)
                save_transcription_to_file(transcription, file_path)
                print(f"Saved transcription to {os.path.splitext(file_path)[0]}.txt")

                
if __name__ == '__main__':
    directory_path = "./resources/oli_french/ds918_problem_content_2024_0521_210556"
    process_directory(directory_path)