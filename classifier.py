# Linear SVM classifier (car vs non-car) using HOG features

import cv2
import numpy as np
import os
import time
import pickle
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# TODO: go back and try other than GRAY

class SVMClassifier(object):
    def __init__(self, model_path=None):
        self.orientation = 8
        self.pixel_per_cell = 8
        self.cell_per_block = 2
        self.orientation = 9
        self.cspace = 'GRAY'
        self.hog_channel = 'ALL'

        if model_path:  # Load existing model if there is
            self.model = joblib.load(model_path)

    def calc_hog_features(self, img, orientation, pixel_per_cell, cell_per_block, vis=False, feature_vec=True):
        if vis is True:
            hog_features, hog_image = hog(img, orientations=orientation,
                                          pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                                          cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                          visualise=True, feature_vector=feature_vec)
            return hog_features, hog_image
        else:
            hog_features = hog(img, orientations=orientation, pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                               cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                               visualise=False, feature_vector=feature_vec)
            return hog_features

    def extract_features(self, image_files):
        features = []
        for file in image_files:
            img = cv2.imread(file)

            # Convert to specified channels
            if self.cspace == 'GRAY':
                feature_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif self.cspace == 'RGB':
                feature_img = img
            elif self.cspace == 'HSV':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.cspace == 'LUV':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.cspace == 'HLS':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.cspace == 'YUV':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.cspace == 'YCrCb':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

            # Select specified channel(s)
            if self.cspace == 'GRAY':  # Gray only has 1 channel, no need to select
                hog_features = self.calc_hog_features(feature_img, self.orientation,
                                                      self.pixel_per_cell, self.cell_per_block, vis=False, feature_vec=True)
            else:
                if self.hog_channel == 'ALL':  # Concatenate all channels together
                    hog_features = []
                    for channel in range(feature_img.shape[2]):
                        hog_features.append(self.calc_hog_features(feature_img[:, :, channel],
                                                                   self.orientation, self.pixel_per_cell,
                                                                   self.cell_per_block,
                                                                   vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)
                else:  # Select only one channel
                    hog_features = self.calc_hog_features(feature_img[:, :, self.hog_channel], self.orientation,
                                                          self.pixel_per_cell, self.cell_per_block, vis=False, feature_vec=True)

            features.append(hog_features)
            print("Extracted HOG features for {}".format(file))

        return features

    def get_features_for_files_in_folder(self, folder):
        filepaths = []
        for root, _, files in os.walk(folder):
            for filename in files:
                if filename.endswith('png'):
                    filepaths.append(os.path.join(root, filename))
        return self.extract_features(filepaths)

# Train a SVM classifier
if __name__ == '__main__':
    classifier = SVMClassifier()
    # X_train = classifier.extract_features(['hog_test.png'])

    # Prepare training data
    car_features = classifier.get_features_for_files_in_folder('vehicles')
    non_car_features = classifier.get_features_for_files_in_folder('non-vehicles')

    X = np.vstack((car_features, non_car_features)).astype(np.float64)

    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    # Ready to train SVM classifier

    svc = LinearSVC()

    print('Training SVC...')
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    print(round(t2 - t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVM = ', round(svc.score(X_test, y_test), 4))

    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    # Save model to file
    joblib.dump(svc, 'svc.model')

    # Also save other data that's needed for doing prediction later
    saved_data = {
        'X_scaler': X_scaler
    }
    pickle.dump(saved_data, open('saved_data.p', 'wb'))
