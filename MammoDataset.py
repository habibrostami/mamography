import os
import sys
import numpy as np
import skimage.io
import keras.preprocessing.image as Kimage
from xml.etree import ElementTree
import re
import cv2
# Import Mask RCNN
#sys.path.append(os.path.abspath("./Mask_RCNN"))  # To find local version of the library
from MaskRCNN import utils

# class that defines and loads the kangaroo dataset
class MammoDataset(utils.Dataset):
    def load_image(self, image_id):
        info = self.image_info[image_id]
        filepath = info['path']
        #image = imageio.imread(fp)
        image = Kimage.img_to_array(Kimage.load_img(filepath), dtype='uint8')
        #ds = pydicom.read_file(fp)
        #image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_dataset(self, dataset_dir, subset):
        # define one class
        self.add_class("dataset", 1, "Mass")
        self.add_class("dataset", 2, "Calc")
        i = 0
        # define data locations
        images_dir = os.path.join(dataset_dir, subset, 'mammo')
        # find all images
        for filename in os.listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            img_path = os.path.join(images_dir, filename)
            # add to dataset
            self.add_image('dataset', image_id=i, path=img_path, annotation='')
            i += 1

    def extract_boxes(self, filename):
        roi_mask = Kimage.img_to_array(Kimage.load_img(filename), dtype='uint8')
        bboxes = self.__find_roi(roi_mask, 5)
        return bboxes, roi_mask.shape[0], roi_mask.shape[1]

    def __find_roi(self, roi_mask, min_roi_area=5):
        roi_mask_8u = roi_mask.astype('uint8')
        roi_mask_8u = cv2.cvtColor(roi_mask_8u, cv2.COLOR_RGB2GRAY)
        low_th = int(roi_mask_8u.max() * .05)
        _, img_bin = cv2.threshold(roi_mask_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)
        #Kimage.save_img('test.png', roi_mask_8u)
        _, contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        bbox = []
        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M['m00'] < 1:
                continue
            area = cv2.contourArea(cnt)
            if area < min_roi_area:
                continue
            bbox.append([rx, ry, rw, rh, M, cnt])
        return bbox

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        #image_id = info['image_id']
        #path = info['annotation']
        path = info['path']
        path_roiMass= path.replace('mammo', 'roiMass')
        path_roiCalc = path.replace('mammo', 'roiCalc')

        masks, class_ids, masks1, class_ids1, masks2, class_ids2 = None, None, None, None, None, None
        if os.path.isfile(path_roiMass):
            masks1, class_ids1 = self.extract_masks(path_roiMass, 'Mass')
        if os.path.isfile(path_roiCalc):
            masks2, class_ids2 = self.extract_masks(path_roiCalc, 'Calc')

        if class_ids1 is not None and class_ids2 is not None:
            masks = np.concatenate((masks1, masks2), axis=2)
            class_ids = np.concatenate((class_ids1, class_ids2))
        elif class_ids1 is not None:
            masks = masks1
            class_ids = class_ids1
        elif class_ids2 is not None:
            masks = masks2
            class_ids = class_ids2

        return masks, class_ids

    def extract_masks(self, path_mask, class_name):
        boxes, h, w = self.extract_boxes(path_mask)
        masks = np.zeros([h, w, len(boxes)], dtype=np.bool)
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            (rx, ry, rw, rh, M, r_contours) = box
            masks[:, :, i] = cv2.drawContours(np.zeros([h, w]), [r_contours], -1, 1, -1)
            class_ids.append(self.class_names.index(class_name))
        return masks, np.asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
