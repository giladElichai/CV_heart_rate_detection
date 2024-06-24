import time

import numpy as np

from FaceFeatures import CFaceFeatures
from NaivePulseCalculator import CNaivePulseCalculator
# from MagPulseCalculator import CMagPulseCalculator

class CHeartRateExtractor:
    
    def __init__(self) -> None:
        self.fps = 0
        self.ff = CFaceFeatures()
        self.pc= CNaivePulseCalculator(self.fps)
        # self.pc = CMagPulseCalculator(self.fps)
        
        self.bpm = 0
    
    def set_fps(self, fps):
        self.fps = fps 
        self.pc.set_fps(fps)
         
    
    def plot_bpm( self ):
        
        freq, fft = self.pc.plot_bpm()
        
        return freq, fft
    
    
    def process(self, iFrame, frame):
        
        timestamp = time.time()
        
        face_result = self.ff.process(frame)

        if len(face_result) == 0:
           return frame 
             
        self.pc.process(timestamp, frame, face_result[0])
        
        if iFrame % self.fps == 0:
            self.bpm = self.pc.approximate_pulse()
        
        # print(bpm)
        frame = self.ff.draw_result(frame, face_result)

        return frame, int(self.bpm)
    