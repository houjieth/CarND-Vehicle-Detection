import cv2
import numpy as np
import pickle
from classifier import SVMClassifier
from scipy.ndimage.measurements import label


class Detector(object):
    def __init__(self, classifier, X_scaler):
        self.classifier = classifier
        self.X_scaler = X_scaler
        self.bboxes = []
        self.heat_map = None

    def find_car_windows(self, img, window_size):
        """
        Slow version. Will need to calculate HOG feature for each window scanned
        """
        draw_img = np.copy(img)
        feature_img = self.classifier.get_feature_image(img)

        x_range = feature_img.shape[1]
        y_range = self.y_stop - self.y_start

        n_x_steps = x_range // window_size
        n_y_steps = y_range // window_size

        x_step = 0.0
        while x_step < n_x_steps:
            y_step = 0.0
            while y_step < n_y_steps:
                y_top = int(self.y_start + y_step * window_size)
                x_left = int(x_step * window_size)

                # Pick up the sub area from whole HOG result by specifying block index ranges on X and Y
                window_img = cv2.resize(feature_img[y_top:y_top + window_size, x_left:x_left + window_size], (64, 64))
                hog_features, _ = self.classifier.get_multi_channel_hog_features(window_img)
                # Scale features and make a prediction
                scaled_features = self.X_scaler.transform(hog_features)
                prediction = self.classifier.model.predict(scaled_features)

                if prediction == 1:
                    bbox = ((x_left, y_top), (x_left + window_size, y_top + window_size))
                    self.bboxes.append(bbox)
                    cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 2)

                y_step += 0.25
            x_step += 0.25

        return draw_img

    def find_car_windows_fast(self, img, scale, color=(0, 0, 255)):
        """
        Fast version. Will only need to calculate the HOG for the whole image once
        """
        draw_img = np.copy(img)

        img_to_search = img[self.y_start:self.y_stop, :, :]
        feature_img = self.classifier.get_feature_image(img_to_search)
        if scale != 1:
            imshape = feature_img.shape
            feature_img = cv2.resize(feature_img, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        n_x_blocks = (feature_img.shape[1] // self.classifier.pixel_per_cell) - self.classifier.cell_per_block + 1
        n_y_blocks = (feature_img.shape[0] // self.classifier.pixel_per_cell) - self.classifier.cell_per_block + 1

        # Always use window_size = 64 when searching on feature image:
        #   if scale is 1.0, this means a 64x64 window search on original image
        #   if scale is 1.5, this means a 96x96 window search on original image
        #   if scale is 2.0, this means a 128x128 window search on original image
        window_size = 64
        n_blocks_per_window = (window_size // self.classifier.pixel_per_cell) - self.classifier.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        n_x_steps = (n_x_blocks - n_blocks_per_window) // cells_per_step  # x overlap percentage: (1 - cells_per_step / (window_size / pixel_per_cell)) * 100%
        n_y_steps = (n_y_blocks - n_blocks_per_window) // cells_per_step  # y overlap percentage: (1 - cells_per_step / (window_size / pixel_per_cell)) * 100%

        # Compute individual channel HOG features for the entire image
        hog, n_channel = self.classifier.get_multi_channel_hog_features(feature_img, feature_vec=False, ravel=False)  # Use feature_vec=False to keep original shape (such as MxNx7x7x9)

        for x_block in range(n_x_steps + 1):
            for y_block in range(n_y_steps + 1):
                y_pos = y_block * cells_per_step
                x_pos = x_block * cells_per_step

                # Pick up the sub area from whole HOG result by specifying block index ranges on X and Y
                if n_channel == 1:
                    hog_features = hog[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
                else:
                    hog_features = []
                    for i in range(n_channel):
                        hog_features.append(hog[i][y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window])
                    hog_features = np.ravel(hog_features)

                x_left = x_pos * self.classifier.pixel_per_cell
                y_top = y_pos * self.classifier.pixel_per_cell

                # Scale features and make a prediction
                scaled_features = self.X_scaler.transform(hog_features)
                prediction = self.classifier.model.predict(scaled_features)

                if prediction == 1:
                    x_left_origin_scale = np.int(x_left * scale)
                    y_top_origin_scale = np.int(y_top * scale)
                    window_size_origin_scale = np.int(window_size * scale)
                    if x_left_origin_scale < 700:
                        continue
                    bbox = ((x_left_origin_scale, y_top_origin_scale + self.y_start),
                            (x_left_origin_scale + window_size_origin_scale,
                             y_top_origin_scale + self.y_start + window_size_origin_scale))
                    self.bboxes.append(bbox)
                    cv2.rectangle(draw_img, bbox[0], bbox[1], color, 2)

        return draw_img

    @staticmethod
    def draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img

    @staticmethod
    def apply_threshold(heat_map, threshold):
        heat_map[heat_map <= threshold] = 0

    def find_cars(self, img):
        self.y_start = 400
        self.y_stop = 470
        self.find_car_windows_fast(img, 1.0)

        self.y_start = 400
        self.y_stop = 500
        self.find_car_windows_fast(img, 1.25)

        self.y_start = 400
        self.y_stop = 600
        self.find_car_windows_fast(img, 1.5)

        self.y_start = 400
        self.y_stop = 650
        self.find_car_windows_fast(img, 2.0)

        self.y_start = 400
        self.y_stop = 700
        self.find_car_windows_fast(img, 3.0)

        if len(self.bboxes) > 15:
            self.bboxes = self.bboxes[len(self.bboxes) - 15:]

        self.heat_map = np.zeros((img.shape[0], img.shape[1]))
        for bbox in self.bboxes:
            self.heat_map[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

        self.apply_threshold(self.heat_map, 3)

        # cv2.imshow('img', self.heat_map)
        # cv2.waitKey(0)

        labels = label(self.heat_map)
        output_img = self.draw_labeled_bboxes(img, labels)
        # cv2.imshow('img', output_img)
        # cv2.waitKey(0)
        return output_img
