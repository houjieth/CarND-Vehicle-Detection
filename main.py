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

    # img = cv2.imread('debug_images/shot0081.png')
    # img = detector.find_cars(img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    for file in sorted(os.listdir('debug_images_0')):
        img = cv2.imread(os.path.join('debug_images_0', file))
        img = detector.find_cars(img)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join('debug_output_images_0', file), img)


    # video = VideoFileClip('project_video.mp4')
    # result = video.fl_image(detector.find_cars)
    # result.write_videofile('output_project_video.mp4', audio=False)
