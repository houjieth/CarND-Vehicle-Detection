import cv2
import os
import pickle
from classifier import SVMClassifier
from detector import Detector
from moviepy.editor import VideoFileClip


if __name__ == '__main__':
    classifier = SVMClassifier('svc.model')
    saved_data = pickle.load(open('saved_data.p', 'rb'))
    X_scaler = saved_data['X_scaler']
    detector = Detector(classifier, X_scaler)

    # img = cv2.imread('test_images/test5.jpg')
    # img = detector.find_cars(img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    video = VideoFileClip('project_video.mp4')
    result = video.fl_image(detector.find_cars)
    result.write_videofile('output_video.mp4', audio=False)
