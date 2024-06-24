import time

import numpy as np

class CNaivePulseCalculator:
    
    def __init__(self, fps) -> None:
        
        self.data =[]
        self.buffer_size = int(fps*10)
        self.fps = fps 
        
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0
    
    def set_fps(self, fps):
        self.fr = fps 
    
    def area_mean_value(self, frame, bbox):
        forehead = frame[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        means = np.mean(forehead, axis=(0,1))
        value = np.sum(means)/3
        return value 
    
    
    def plot_bpm(self):
        
        return self.freqs, self.fft
    
    
    def approximate_pulse(self):
        
        L = len(self.data)
        processed = np.array(self.data)
        self.samples = processed
        if L > 15:
            self.output_dim = processed.shape[0]

            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > 50) & (freqs < 180))

            pruned = self.fft[idx]
            phase = phase[idx]

            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned
            idx2 = np.argmax(pruned)

            t = (np.sin(phase[idx2]) + 1.) / 2.
            t = 0.9 * t + 0.1
            
            self.bpm = self.freqs[idx2]
            
        return self.bpm
        
    
    def process(self, timestamp, frame, face_features):
        
        bbox = face_features["forehead_location"]
        
        self.times.append(timestamp - self.t0)
        value = self.area_mean_value(frame, bbox)
        self.data.append(value)
        
        L = len(self.data)
        
        if L > self.buffer_size:
            self.data = self.data[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
        
        return
    
    