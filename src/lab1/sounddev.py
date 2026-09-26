import sounddevice as sd

fs = 48000
sd.default.samplerate = 48000
sd.default.channels = 2

myrecording = sd.rec(int(5 * fs))
sd.wait()
sd.play(myrecording, fs)