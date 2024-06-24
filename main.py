import os, sys 

import cv2 
import numpy as np


from heartRateExtractor import CHeartRateExtractor


def main():
    
    cv2.namedWindow("preview")

    vc = cv2.VideoCapture(0)
    fps = vc.get(cv2.CAP_PROP_FPS)
    
    hre = CHeartRateExtractor(fps)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
        
    iFrame = 0

    cv2.imshow("preview", frame)
    while rval:
        
        rval, frame = vc.read()
        frame = frame.copy()
        
        frame = hre.process(frame, iFrame)
        cv2.imshow("preview", frame)
        key = cv2.waitKey(20)
        
        iFrame += 0
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()
    
    
    pass 

if __name__ == "__main__":
    main()
