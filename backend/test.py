from audio_utils import preprocess_audio
# import os

# file_path = os.path.join("test_audio", "typing_audio.wav")
# print(file_path)
import os

file_path = r"test_audio\typing_audio.wav"
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    print(f"File found: {file_path}")

# file_path = "./test_audio/typing_audio.wav"

input_tensor = preprocess_audio(file_path)

print(input_tensor.shape)
# Output: [1, 1, n_mels, time_frames] → ready for CNN
