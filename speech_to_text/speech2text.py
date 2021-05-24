import librosa
import sounddevice as sd
import torch
from scipy.io.wavfile import write
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

fs = 16000

seconds = int(input("Enter you duration of recording.  "))

print("Your recording started to ", seconds, " seconds")

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()

write('output.wav', fs, myrecording)

print("It is stopped....")

audio, rate = librosa.load("output.wav", sr=fs)

input_values = tokenizer(audio, return_tensors="pt").input_values

logits = model(input_values).logits

prediction = torch.argmax(logits, dim=-1)

transcription = tokenizer.batch_decode(prediction)[0]
print(transcription)
