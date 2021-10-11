import win32com.client as comclt
import pyaudio
import numpy as np
frame_length = 8000
threshold = 0.02
sr = 44100

wsh= comclt.Dispatch("WScript.Shell")
wsh.AppActivate("Notepad") # select another application

p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paFloat32,channels=1,rate=sr,input=True,
              frames_per_buffer=frame_length)
print('start')
while True : 
        sig = np.frombuffer(stream.read(frame_length),dtype=np.float32)
        b = np.abs(sig)
        m = np.mean(b) 
        if(m >= threshold):
            wsh.SendKeys("l")
            print('press') # send the keys you want
        # print(m)
