import cv2
import os
import imghdr
import json
import numpy as np
import pandas as pd
from types import FunctionType


class Feature_extractor:

    def __init__(self, DATA_PATH, is_test=False):
        self.DATA_PATH = DATA_PATH
        self.IMAGE_PATH = os.path.join(DATA_PATH, "images")
        self.TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
        self.TEST_PATH = os.path.join(DATA_PATH, "test.csv")
        self.pre_extracted_feature_panda_frame = pd.read_csv(self.TRAIN_PATH)
        if not is_test:
            self.features = self.read_csv(self.TRAIN_PATH)
        else:
            self.features = self.read_csv(self.TEST_PATH, is_test = True)

    def read_csv(self, file_name = "train.csv",is_test = False):
        features = {}
        with open(file_name, "r") as f:
            lines = f.readlines()[1:]
        for line in lines:
            row = line.strip().split(",")
            file_name = str(row[0])+".jpg"
            if not is_test:
                class_name = row[1]
                row_feature = map(float, row[2:])
            else:
                row_feature = map(float, row[1:])
            img_fe = Image_feature_extractor(os.path.join(self.IMAGE_PATH, file_name))
            print("Processing " + file_name + ".......")
            if is_test == True:
                features[file_name] = {"pre_feature" : row_feature, "cv_feature" : img_fe.get_all_features()}
            else:
                features[file_name] = {'class_name' : class_name, "pre_feature" : row_feature, "cv_feature" : img_fe.get_all_features()}
        return features
    
    def get_feature(self, file_name):
        return self.features[file_name]['feature']

    def get_class_name(self, file_name):
        return self.features[file_name]['class_name']

    def write_to_json(self, file_name):
        json_str = json.dumps(self.features, indent=4, sort_keys=True)
        with open(file_name, "w") as f:
            f.write(json_str)


class Image_feature_extractor:
    
    def __init__(self, file_name):
        self.img = cv2.imread(file_name)
        self.img = self.img[:,:,0]
        self.contour = self.get_contour()

    def get_contour(self):
        _, thresh = cv2.threshold(self.img,127,255,cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return sorted(contours, key=cv2.contourArea)[-1]

    def feature_width(self):
        return float(self.img.shape[0])

    def feature_height(self):
        return float(self.img.shape[1])

    def feature_width_height_ratio(self):
        return self.feature_width()/self.feature_height()

    def feature_black_white_ratio(self):
        return np.sum(self.img)/(self.feature_width()*self.feature_height())

    def feature_counter_width(self):
        x,y,w,h = cv2.boundingRect(self.contour)
        return w

    def feature_counter_heigth(self):
        x,y,w,h = cv2.boundingRect(self.contour)
        return h

    def feature_is_convex(self):
        convex_hull = cv2.convexHull(self.contour)
        is_convex = cv2.isContourConvex(convex_hull)
        if is_convex:
            return 1
        else:
            return 0

    def feature_convex_contour_ratio(self):
        convex_hull = cv2.convexHull(self.contour)
        area_contour = cv2.contourArea(self.contour)
        area_convex_hull = cv2.contourArea(convex_hull)
        return float(area_contour)/float(area_convex_hull)

    def feature_contour_box_rotation(self):
        rect = cv2.minAreaRect(self.contour)
        return rect[-1]

    def feature_contour_approx_circle_area_ratio(self):
        (x, y), radius = cv2.minEnclosingCircle(self.contour)
        area_circle = radius * radius * np.pi
        area_contour = cv2.contourArea(self.contour)
        return area_contour/area_circle

    def feature_contour_approx_ellipse_area_ratio(self):
        (x, y), (max_axis, min_axis), angle = cv2.fitEllipse(self.contour)
        area_ellipse = np.pi * max_axis * min_axis
        area_contour = cv2.contourArea(self.contour)
        return area_contour/area_ellipse

    def feature_approx_poly_vertex(self):
        epsilon = 0.1*cv2.arcLength(self.contour,True)
        approx = cv2.approxPolyDP(self.contour,epsilon,True)
        return len(approx)

    def feature_convex_hull_vertex(self):
        convex_hull = cv2.convexHull(self.contour)
        epsilon = 0.1*cv2.arcLength(convex_hull,True)
        approx = cv2.approxPolyDP(convex_hull,epsilon,True)
        return len(approx)

    def feature_contour_approx_convex_hull_vertex_ratio(self):
        convex_hull = cv2.convexHull(self.contour)
        epsilon_convex = 0.1*cv2.arcLength(convex_hull,True)
        approx_convex = cv2.approxPolyDP(convex_hull,epsilon_convex,True)
        epsilon_contour = 0.1*cv2.arcLength(self.contour,True)
        approx_contour = cv2.approxPolyDP(self.contour,epsilon_contour,True)
        return float(len(approx_convex))/float(len(approx_contour))

    def get_all_features(self):
        features = []
        for i in dir(self):
           if i.startswith('feature'):
               features.append(getattr(self,i)())
        return features


def main(DATA_DIR):
    # feature_extractor = Feature_extractor('./input')
    # feature_extractor.write_to_json('train_features.json')
    feature_extractor = Feature_extractor('./input',is_test=True)
    feature_extractor.write_to_json('test_features.json')

if __name__ == "__main__":
    main('./input')