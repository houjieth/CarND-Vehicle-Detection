import cv2
import numpy as np
import pickle
from classifier import SVMClassifier


class Detector(object):
    def __init__(self, classifier, X_scaler):
        self.classifier = classifier
        self.y_start = 400
        self.y_stop = 656
        self.X_scaler = X_scaler

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, scale):

        draw_img = np.copy(img)
        img = img.astype(np.float32)

        img_to_search = img[self.y_start:self.y_stop, :, :]
        img_to_search = self.classifier.get_feature_image(img_to_search)
        if scale != 1:
            imshape = img_to_search.shape
            img_to_search = cv2.resize(img_to_search, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        img = img_to_search

        n_x_blocks = (img.shape[1] // self.classifier.pixel_per_cell) - self.classifier.cell_per_block + 1
        n_y_blocks = (img.shape[0] // self.classifier.pixel_per_cell) - self.classifier.cell_per_block + 1

        # NOT USED
        n_feature_per_block = self.classifier.orientation * self.classifier.cell_per_block ** 2

        # Always use window_size = 64 when searching on feature image:
        #   if scale is 1.0, this means a 64x64 window search on original image
        #   if scale is 1.5, this means a 96x96 window search on original image
        #   if scale is 2.0, this means a 128x128 window search on original image
        window_size = 64
        n_blocks_per_window = (window_size // self.classifier.pixel_per_cell) - self.classifier.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        n_x_steps = (
                    n_x_blocks - n_blocks_per_window) // cells_per_step  # x overlap percentage: (1 - cells_per_step / (window_size / pixel_per_cell)) * 100%
        n_y_steps = (
                    n_y_blocks - n_blocks_per_window) // cells_per_step  # y overlap percentage: (1 - cells_per_step / (window_size / pixel_per_cell)) * 100%

        # Compute individual channel HOG features for the entire image
        hog = self.classifier.calc_hog_features(img, self.classifier.orientation, self.classifier.pixel_per_cell,
                                                self.classifier.cell_per_block,
                                                feature_vec=False)  # Use feature_vec=False to keep original shape (such as MxNx7x7x9)

        for x_block in range(n_x_steps):
            for y_block in range(n_y_steps):
                y_pos = y_block * cells_per_step
                x_pos = x_block * cells_per_step

                # Pick up the sub area from whole HOG result by specifying block index ranges on X and Y
                hog_features = hog[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()

                x_left = x_pos * self.classifier.pixel_per_cell
                y_top = y_pos * self.classifier.pixel_per_cell

                # Extract the image patch
                # NOT USED
                sub_img = cv2.resize(img[y_top:y_top + window_size, x_left:x_left + window_size], (64, 64))

                # Scale features and make a prediction
                scaled_features = self.X_scaler.transform(hog_features)
                prediction = self.classifier.model.predict(scaled_features)

                if prediction == 1:
                    x_left_origin_scale = np.int(x_left * scale)
                    y_top_origin_scale = np.int(y_top * scale)
                    window_size_origin_scale = np.int(window_size * scale)
                    cv2.rectangle(draw_img, (x_left_origin_scale, y_top_origin_scale + self.y_start),
                                  (x_left_origin_scale + window_size_origin_scale,
                                   y_top_origin_scale + self.y_start + window_size_origin_scale), (0, 0, 255), 6)

        return draw_img


if __name__ == '__main__':
    classifier = SVMClassifier('svc.model')
    saved_data = pickle.load(open('saved_data.p', 'rb'))
    X_scaler = saved_data['X_scaler']
    detector = Detector(classifier, X_scaler)

    img = cv2.imread('test_images/test1.jpg')

    out_img = detector.find_cars(img, 2.0)
    cv2.imshow('img', out_img)
    cv2.waitKey(0)


