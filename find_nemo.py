from KNN import KNN

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class FindingNemo:
    def __init__(self, train_image):
        self.knn = KNN(k=3)
        X_train, Y_train = self.convert_image_to_dataset(train_image)
        self.knn.fit(X_train, Y_train)
    
    def convert_image_to_dataset(self, image):
        # Load and preprocess the image
        nemo = cv2.imread(image)
        nemo = cv2.resize(nemo, (0, 0), fx=0.25, fy=0.25)
        nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
        nemo_hsv = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)

        # Reshape the image to a 2D array of pixels (rows, columns, channels)
        pixels_list_hsv = nemo_hsv.reshape(-1, 3)
        X_train = pixels_list_hsv / 255

        # Create binary mask for Nemo's color range
        light_orange = (1, 190, 200)
        dark_orange = (18, 255, 255)

        # Create a binary mask within the specified orange color range
        mask = cv2.inRange(nemo_hsv, light_orange, dark_orange)

        # Define the white color range in HSV
        light_white = (0, 0, 200)
        dark_white = (145, 60, 255)

        # Create a binary mask within the specified white color range
        mask_white = cv2.inRange(nemo_hsv, light_white, dark_white)

        # Combine the masks for orange and white regions using logical OR
        final_mask = mask + mask_white
    
        Y_train = final_mask.reshape(-1,) // 255

        return X_train, Y_train

    def remove_background(self, test_image):
        # Load and preprocess the test image
        test_nemo = cv2.imread(test_image)
        test_nemo = cv2.resize(test_nemo, (0, 0), fx=0.25, fy=0.25)
        test_nemo = cv2.cvtColor(test_nemo, cv2.COLOR_BGR2RGB)
        test_nemo_hsv = cv2.cvtColor(test_nemo, cv2.COLOR_RGB2HSV)
        X_test = test_nemo_hsv.reshape(-1, 3) / 255

        # Predict the mask for the test image
        Y_pred = self.knn.predict(X_test)
        output = Y_pred.reshape(test_nemo.shape[:2])

        # Show the results
        plt.imshow(output, cmap='gray')
        plt.show()

# Example usage:
train_image = 'image2.jpeg'
test_image = 'image2.jpeg'
finder = FindingNemo(train_image)
finder.remove_background(test_image)

