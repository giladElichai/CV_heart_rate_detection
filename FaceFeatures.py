import os, sys

import cv2 
import numpy as np 
import matplotlib.pyplot as plt

import dlib

_DEBUG_ = True

def draw_bbox(frame, bboxes, color=(0, 255, 0)):
    
    image = frame.copy()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    for (x1,y1,x2,y2) in bboxes:
        cv2.rectangle(image,(x1, y1),(x2, y2),color,int(round(frameHeight / 150)),4,)

    return image


def draw_dlib_bbox(frame, bboxes):
    
    image = frame.copy()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    for b in bboxes:
        t = _rect_to_css(b)
        cv2.rectangle(image,(t[3], t[0]),(t[1], t[2]),(0, 255, 0),int(round(frameHeight / 150)),8,)

    return image


def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def bboxes_to_dlib(bboxes):
    
    new_boxes = []
    for b in bboxes:
        t = dlib.rectangle(b[0], b[1], b[2], b[3])
        new_boxes.append(t)
        
    return new_boxes

class CFaceFeatures:
     
    def __init__(self) -> None:
        
        modelFile = "models/opencv_face_detector_uint8.pb"
        configFile = "models/opencv_face_detector.pbtxt"
        self.face_detector = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        
        # predictor_68_point_model = r"models\shape_predictor_68_face_landmarks.dat"
        # self.pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

        predictor_5_point_model = r"models\shape_predictor_5_face_landmarks.dat"
        self.pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)
        
        # self.dlib_face_detector = dlib.get_frontal_face_detector()
        
        pass
    
    def locate_face(self, frame, conf_threshold=0.7):
        
        
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
 
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < conf_threshold:
                continue
                
            if any(detections[0, 0, i, 3:7] > 1.):
                continue
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            
            bboxes.append([x1, y1, x2, y2])
            
        
        # tmp = self.dlib_face_detector(frame, 1)
        # tmp_image = draw_dlib_bbox(frame, tmp)
                
        return bboxes
        
    
    def raw_face_landmarks(self, frame, face_locations=None):
        
        if face_locations is None:
            face_locations = self.locate_face(frame)

        # pose_predictor = self.pose_predictor_68_point
        # if model == "small":
        pose_predictor = self.pose_predictor_5_point
            
        face_locations = bboxes_to_dlib(face_locations)
        
        # dlib_face_locations = self.dlib_face_detector(frame, 1)
        # # tmp_image = draw_dlib_bbox(frame, tmp)
        
        faces_landmarks = [pose_predictor(frame, face_location) for face_location in face_locations]
        landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in faces_landmarks]
        
        return landmarks_as_tuples
    
    
    def face_landmarks(self, face_image, face_locations=None):

        landmarks_as_tuples = self.raw_face_landmarks(face_image, face_locations)
        
        landmarks = [{
            "nose_tip": [points[4]],
            "left_eye": points[2:4],
            "right_eye": points[0:2],
        } for points in landmarks_as_tuples]
             
        return landmarks, landmarks_as_tuples
    
    def locate_foreheads(self, face_bboxes, face_landmarks):
        
        foreheads = []
        for bbox, landmarks in zip(face_bboxes, face_landmarks):
            
            # left_eye = (np.array(landmarks["left_eye"][0]) + np.array(landmarks["left_eye"][1])) / 2
            # right_eye = (np.array(landmarks["right_eye"][0]) + np.array(landmarks["right_eye"][1])) / 2
            left_eye = np.array(landmarks["left_eye"][0])
            right_eye = np.array(landmarks["right_eye"][0])
            bottom = min(left_eye[1], right_eye[1])
            forehead = [int(left_eye[0]), int(bbox[1]), int(right_eye[0]), int(bottom)]
            w, h = forehead[2]-forehead[0], forehead[3]-forehead[1]
            w *= 0.1
            h *= 0.1
            forehead = [forehead[0]+2*int(w), forehead[1]+2*int(h), forehead[2]-3*int(w), forehead[3]-3*int(h)]
            foreheads.append(forehead)
            
        
        return foreheads 
    
    def process(self, frame):
        
        face_bboxes = self.locate_face(frame)
        if len(face_bboxes) == 0:
            return []
        
        face_landmarks, landmarks_as_tuples = self.face_landmarks(frame, face_locations=face_bboxes)
        foreheads_locations = self.locate_foreheads(face_bboxes, face_landmarks)
    
        results = []
        for i in range(len(face_bboxes)):
            result = {"face_bbox":face_bboxes[i], 
                    "face_landmarks":face_landmarks[i], 
                    "landmarks_as_tuples": landmarks_as_tuples[i],
                    "forehead_location":foreheads_locations[i]
                    }
            results.append(result)
            
        if _DEBUG_:
            self.draw_result(frame, results)
        
        return results 
    
    def draw_result(self, frame, results):
        
        if len(results) == 0:
            return frame
        
        image = frame.copy()
        for result in results:
            face_bbox = result["face_bbox"]
            landmarks_as_tuples = result["landmarks_as_tuples"]
            forehead_location = result["forehead_location"]
            
            image = draw_bbox(image, [face_bbox])
            image = draw_bbox(image, [forehead_location], (255,0,0))
            for (x, y) in landmarks_as_tuples:
                cv2.circle(image, (x, y), 1, (0, 0, 255), 3)
            
        return image
    
    
def main():
    
    fa = CFaceFeatures()
    frame = cv2.imread("devimages/1.jpg")
    landmarks = fa.process(frame)
    
    pass 
    
if __name__ == "__main__":
    main()