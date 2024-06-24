import time

import cv2
import numpy as np
import scipy.signal as signal

## ======================================================================================
## Helper functions

## Color spaces
def rgb2yiq(rgb):
    """ Converts an RGB image to YIQ using FCC NTSC format.
        This is a numpy version of the colorsys implementation
        https://github.com/python/cpython/blob/main/Lib/colorsys.py
        Inputs:
            rgb - (N,M,3) rgb image
        Outputs
            yiq - (N,M,3) YIQ image
        """
    # compute Luma Channel
    y = rgb @ np.array([[0.30], [0.59], [0.11]])

    # subtract y channel from red and blue channels
    rby = rgb[:, :, (0,2)] - y

    i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
    q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)

    yiq = np.dstack((y.squeeze(), i, q))
    
    return yiq


def bgr2yiq(bgr):
    """ Coverts a BGR image to float32 YIQ """
    # get normalized YIQ frame
    rgb = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    yiq = rgb2yiq(rgb)

    return yiq


def yiq2rgb(yiq):
    """ Converts a YIQ image to RGB.
        Inputs:
            yiq - (N,M,3) YIQ image
        Outputs:
            rgb - (N,M,3) rgb image
        """
    r = yiq @ np.array([1.0, 0.9468822170900693, 0.6235565819861433])
    g = yiq @ np.array([1.0, -0.27478764629897834, -0.6356910791873801])
    b = yiq @ np.array([1.0, -1.1085450346420322, 1.7090069284064666])
    rgb = np.clip(np.dstack((r, g, b)), 0, 1)
    return rgb


inv_colorspace = lambda x: cv2.normalize(
    yiq2rgb(x), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)

## Gaussian Pyramid
def gaussian_pyramid(image, level):
    """ Obtains single band of a Gaussian Pyramid Decomposition
        Inputs: 
            image - single channel input image
            num_levels - number of pyramid levels
        Outputs:
            pyramid - Pyramid decomposition tensor
        """ 
    rows, cols, colors = image.shape
    scale = 2**level # downscale factor
    pyramid = np.zeros((colors, rows//scale, cols//scale))

    for i in range(0, level):

        image = cv2.pyrDown(image, dstsize=(cols//2, rows//2))
        rows, cols, _ = image.shape

        if i == (level - 1):
            for c in range(colors):
                pyramid[c, :, :] = image[:, :, c]

    return pyramid


def box_center(box):
    cy = (box[0] + box[2]) // 2
    cx = (box[1] + box[3]) // 2
    return cy, cx


def mag_colors(rgb_frames, fs, freq_lo, freq_hi, level, alpha):
    """ Function to obtain Amplified Colors in a given list of RGB frames 
        Inputs:
            rgb_frames - list of RGB uint8 video frames 
            fs - sample frequency
            freq_lo - lower frequency bound
            freq_hi - upper frequency bound
            level - level of Gaussian Pyramid
            alpha - magnification factor
        Outputs:
            magnified_frames - COlor magnified RGB video frames
    """
    rows, cols, colors = rgb_frames[0].shape
    num_frames = len(rgb_frames)

    # convert frames to YIQ colorspace
    frames = [rgb2yiq(frame/255) for frame in rgb_frames]

    ## Get Temporal Filter
    bandpass = signal.firwin(numtaps=num_frames,
                             cutoff=(freq_lo, freq_hi),
                             fs=fs,
                             pass_zero=False)
    
    transfer_function = np.fft.fft(np.fft.ifftshift(bandpass))
    transfer_function = transfer_function[:, None, None, None].astype(np.complex64)

    ## Get Gaussian Pyramid Stack
    scale = 2**level
    pyramid_stack = np.zeros((num_frames, colors, rows//scale, cols//scale))
    for i, frame in enumerate(frames):
        pyramid = gaussian_pyramid(frame, level)
        pyramid_stack[i, :, :, :] = pyramid

    ## Apply Temporal Filtering
    pyr_stack_fft = np.fft.fft(pyramid_stack, axis=0).astype(np.complex64)
    _filtered_pyramid = pyr_stack_fft * transfer_function
    filtered_pyramid = np.fft.ifft(_filtered_pyramid, axis=0).real

    ## Apply magnification to video
    magnified_pyramid = filtered_pyramid * alpha

    ## Collapse Pyramid and reconstruct video
    magnified = []

    for i in range(num_frames):
        y_chan = frames[i][:, :, 0] 
        i_chan = frames[i][:, :, 1] 
        q_chan = frames[i][:, :, 2] 
        
        fy_chan = cv2.resize(magnified_pyramid[i, 0, :, :], (cols, rows))
        fi_chan = cv2.resize(magnified_pyramid[i, 1, :, :], (cols, rows))
        fq_chan = cv2.resize(magnified_pyramid[i, 2, :, :], (cols, rows))

        # apply magnification
        mag = np.dstack((
            y_chan + fy_chan,
            i_chan + fi_chan,
            q_chan + fq_chan,
        ))

        # convert to RGB and normalize
        mag = inv_colorspace(mag)

        # store magnified frames
        magnified.append(mag)
        
    return magnified



## ======================================================================================

class CMagPulseCalculator:
    
    def __init__(self, fps) -> None:
        
        self.crop_size = -1
        self.crop_locations = []
        self.face_crops = []
        self.yiq_face_crops = []
        self.gaussian_pyramids = []
            
        self.level = 4
        self.alpha = 50.0
        self.freq_lo = 50/60
        self.freq_hi = 60/60
        
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
        self.fps = fps 
    
    
    def plot_bpm(self):
        
        return self.freqs, self.fft
    
    
    def magnified_crops(self):
    
        
        num_frames = len(self.gaussian_pyramids)
        bandpass = signal.firwin(numtaps=num_frames,
                                cutoff=(self.freq_lo, self.freq_hi),
                                fs=self.fps,
                                pass_zero=False)
        
        transfer_function = np.fft.fft(np.fft.ifftshift(bandpass))
        transfer_function = transfer_function[:, None, None, None].astype(np.complex64)

        ## Get Gaussian Pyramid Stack
        scale = 2**self.level
        # pyramid_stack = np.zeros((num_frames, 3, rows//scale, cols//scale))
        pyramid_stack = np.array(self.gaussian_pyramids)
        ## Apply Temporal Filtering
        pyr_stack_fft = np.fft.fft(pyramid_stack, axis=0).astype(np.complex64)
        _filtered_pyramid = pyr_stack_fft * transfer_function
        filtered_pyramid = np.fft.ifft(_filtered_pyramid, axis=0).real

        ## Apply magnification to video
        magnified_pyramid = filtered_pyramid * self.alpha
        ## Collapse Pyramid and reconstruct video
        magnified = []
        for i in range(num_frames):
            y_chan = self.yiq_face_crops[i][:, :, 0] 
            i_chan = self.yiq_face_crops[i][:, :, 1] 
            q_chan = self.yiq_face_crops[i][:, :, 2] 
            
            fy_chan = cv2.resize(magnified_pyramid[i, 0, :, :], (self.crop_size, self.crop_size))
            fi_chan = cv2.resize(magnified_pyramid[i, 1, :, :], (self.crop_size, self.crop_size))
            fq_chan = cv2.resize(magnified_pyramid[i, 2, :, :], (self.crop_size, self.crop_size))

            # apply magnification
            mag = np.dstack((
                y_chan + fy_chan,
                i_chan + fi_chan,
                q_chan + fq_chan,
            ))

            # convert to RGB and normalize
            mag = inv_colorspace(mag)

            # store magnified frames
            magnified.append(mag)
            
        return magnified
    
    def bbox_color_values(self, face_crop, bbox):
        forehead = face_crop[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        means = np.sum(forehead, axis=(0,1))
        return means
    
    
    
    def approximate_pulse(self):
        
        NUM_FRAMES = len(self.gaussian_pyramids)
        if NUM_FRAMES < 2*self.fps:
            return self.bpm
        
        # mag = mag_colors(self.face_crops, self.fps, self.freq_lo, self.freq_hi, self.level, self.alpha)
         
        values = []
        magnified = self.magnified_crops()
        for i in range(len(magnified)):
            forehead_location = self.crop_locations[i]["forehead_location"]
            value = self.bbox_color_values(magnified[i], forehead_location)
            values.append(value)
        values = np.array(values)
        
        reds = values[:, 0]
        greens = values[:, 1]
        blues = values[:, 2]
    
        freqs = np.fft.rfftfreq(NUM_FRAMES) * self.fps
        rates = np.abs(np.fft.rfft(reds))/NUM_FRAMES
        peak_idx, _ = signal.find_peaks(rates, height=1000)
        self.bpm = freqs[peak_idx].min() * 60
            
        return self.bpm
        
    
    def calculate_crop_region(self, face_bbox):
        
        cy, cx = box_center(face_bbox)
        w = face_bbox[2] - face_bbox[0]
        h = face_bbox[3] - face_bbox[1]
        
        if self.crop_size < 0:
            self.crop_size = int(max(w, h) * 1.) 
            
        top_x = max(0, cx - self.crop_size//2)
        top_y = max(0, cy - self.crop_size//2)
        
        return int(top_x), int(top_y)
    
    
    def readjust_box(self, box, top_x, top_y):
        
        new_box = [box[0] - top_x, 
                    box[1] - top_y, 
                    box[2] - top_x, 
                    box[3] - top_y]
        return new_box
        
    
    def process(self, timestamp, frame, face_features):
        
        self.times.append(timestamp - self.t0)
        
        rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_bbox = face_features["face_bbox"]
        landmarks_as_tuples = face_features["landmarks_as_tuples"]
        forehead_location = face_features["forehead_location"]
        
        top_y, top_x  = self.calculate_crop_region(face_bbox)
        cropped_face_location = self.readjust_box(face_bbox, top_x, top_y)
        cropped_forehead_location = self.readjust_box(forehead_location, top_x, top_y)
        face_crop = rbg[top_y:top_y+self.crop_size, top_x:top_x+self.crop_size]
        
        self.face_crops.append(face_crop)
        self.crop_locations.append({"face_bbox":cropped_face_location, 
                                    "forehead_location":cropped_forehead_location})
        
        yiq_face = rgb2yiq(face_crop/255.)
        self.yiq_face_crops.append(yiq_face)
        pyramid = gaussian_pyramid(yiq_face, self.level)
        self.gaussian_pyramids.append(pyramid)
         
        L = len(self.face_crops)
        if L > self.buffer_size:
            self.times = self.times[-self.buffer_size:]
            self.face_crops = self.face_crops[-self.buffer_size:]
            self.crop_locations = self.crop_locations[-self.buffer_size:]
            self.yiq_face_crops = self.yiq_face_crops[-self.buffer_size:]
            self.gaussian_pyramids = self.face_crops[-self.buffer_size:]
            
        
        return
    
    