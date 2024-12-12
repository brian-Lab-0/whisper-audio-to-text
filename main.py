import os
import torch
import whisper

# Ensure you're in the current working directory
current_dir = os.getcwd()

# Load Whisper model locally, downloading it if not already cached
model_path = os.path.join(current_dir, "whisper-small.pt")

# Download and save the model if not present
if not os.path.exists(model_path):
    print("Downloading model...")
    model = whisper.load_model("small")
    torch.save(model, model_path)
else:
    print("Loading model from disk...")
    model = torch.load(model_path)

# Enable GPU or NPU acceleration if available
if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU for processing.")
elif torch.backends.mps.is_available():  # Apple Silicon or MPS-enabled devices
    device = "mps"
    print("Using Apple Silicon MPS for processing.")
else:
    print("Using CPU for processing.")
    device = "cpu"

# Transcribe audio file from current directory
audio_file = os.path.join(current_dir, "count.mp3")
result = model.transcribe(audio_file, fp16=False)  # fp16 disabled if using CPU

# Print the transcription
print(result['text'])
