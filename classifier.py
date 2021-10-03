import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from keras import models
import matplotlib.pyplot as plt
from MaskRCNN.config import Config
from MaskRCNN import utils
from MaskRCNN import model as modellib
from MaskRCNN import visualize
from MammoDataset import *
from config_mrcnn import *
from sklearn import metrics
import itertools
import keras.preprocessing.image as Kimage
import fnmatch
import cv2
from imgaug import augmenters as iaa
import imgaug as ia
import csv
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle
from config import *
#from keras.models import load_model
from keras import models
if sys.platform == 'win32':
    import winsound


try:
    os.makedirs(REPORTS_DIR, 775, True)
    os.makedirs(REPORTS_DIR, 775, True)
    os.makedirs(os.path.join(REPORTS_DIR, 'classifier'), 775, True)
except:
    print('can not create directory')



def plot_micro_averaged_precision(y_true_binary, y_pred_score):
    y_true_binary = np.array(y_true_binary)
    y_pred_score = np.array(y_pred_score)
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(3):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binary[:, i],
                                                            y_pred_score[:, i])
        average_precision[i] = average_precision_score(y_true_binary[:, i], y_pred_score[:, i])
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_binary.ravel(),
                                                                    y_pred_score.ravel())
    average_precision["micro"] = average_precision_score(y_true_binary, y_pred_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    #plt.figure()
    #plt.step(recall['micro'], precision['micro'], where='post')
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.ylim([0.0, 1.05])
    #plt.xlim([0.0, 1.0])
    #plt.title(
    #    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    #    .format(average_precision["micro"]))

    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 6.5))
    #plt.figure()
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('f1')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('mAP (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(3), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('{0} (area = {1:0.2f})'
                      ''.format(class_names[i], average_precision[i]))

    #fig = plt.gcf()
    #fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve (mAP = {0:0.2f})'.format(average_precision["micro"]))
    plt.legend(lines, labels, loc="lower left", prop=dict(size=14))
    return plt
    #plt.savefig(os.path.join(target_path, MODEL_NAME + '-map.png'))
    #plt.show()


def plot_roc_chart(y_true_binary, y_pred_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true_binary = np.array(y_true_binary)
    y_pred_score = np.array(y_pred_score)
    for i in range(3):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(y_true_binary[:, i], y_pred_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    auc_score = float(sum(roc_auc.values())) / len(roc_auc)
    print(auc_score)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(3):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='{0} (area = {1:0.3f})'
                       ''.format(class_names[i], roc_auc[i]))

    plt.title('ROC curve (AUC = {:.2f})'.format(auc_score, 4))
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc="lower right")
    return plt
    #plt.savefig(os.path.join(target_path, MODEL_NAME + '-roc.png'))
    #plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)
    #plot_confusion_matrix(cm, np.array(class_names))

    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(3)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.suptitle(figure_title)
    plt.title('Confusion Matrix')
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=10)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.tight_layout()
    return plt

    #plt.savefig(os.path.join(target_path, MODEL_NAME)+'-mat.png')
    #plt.show()


def get_report_seq(sufix):
    matches = fnmatch.filter(os.listdir(REPORTS_DIR), '*-'+sufix+'.png')
    seq = "{0:04d}".format(len(matches)+1)
    filepath = os.path.join(REPORTS_DIR, seq + '-'+sufix+'.png')
    while os.path.isfile(filepath):
        seq = "{0:04d}".format(int(seq) + 1)
        filepath = os.path.join(REPORTS_DIR, seq + '-'+sufix+'.png')
    return seq

############################################################
#  Detection
############################################################
def extract_bbox(roi_mask, min_roi_area=5):
    roi_mask_8u = roi_mask.astype('uint8')
    if roi_mask_8u.shape[-1] == 3:
        roi_mask_8u = cv2.cvtColor(roi_mask_8u, cv2.COLOR_RGB2GRAY)
    low_th = int(roi_mask_8u.max() * .05)
    _, img_bin = cv2.threshold(roi_mask_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)
    if cv2.__version__.split('.')[0] == "3":
        _, contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bbox = []
    for cnt in contours:
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M['m00'] < 1:
            continue
        area = cv2.contourArea(cnt)
        if area < min_roi_area:
            continue
        bbox.append([ry, rx, ry + rh, rx + rw])
    return bbox


def find_roi(roi_mask, min_roi_area=5):
    roi_mask_8u = roi_mask.astype('uint8')
    if roi_mask_8u.shape[-1] == 3:
        roi_mask_8u = cv2.cvtColor(roi_mask_8u, cv2.COLOR_RGB2GRAY)
    low_th = int(roi_mask_8u.max() * .05)
    _, img_bin = cv2.threshold(roi_mask_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)
    if cv2.__version__.split('.')[0] == "3":
        _, contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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


def extract_masks(mask, class_id=None):
    boxes = find_roi(mask, 5)
    h, w = mask.shape[:2]
    masks = np.zeros([h, w, len(boxes)], dtype=np.bool)
    class_ids = list()
    bbox = []
    for i in range(len(boxes)):
        box = boxes[i]
        (rx, ry, rw, rh, M, r_contours) = box
        masks[:, :, i] = cv2.drawContours(np.zeros([h, w]), [r_contours], -1, 1, -1)
        if class_id is not None:
            class_ids.append(class_id)
        bbox.append([ry, rx, ry + rh, rx + rw])
    if class_id is not None:
        return masks, np.array(bbox), np.asarray(class_ids, dtype='int32')
    return masks, np.array(bbox)


def box_overlap(box1, box2):
    y11, x11, y12, x12 = box1
    y21, x21, y22, x22 = box2
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)
    if x_right < x_left or y_bottom < y_top:
        return False
    return True


def load_image_gt(target_path, filename, fliplr=False, flipud=False):
    mask_mass = None
    mask_calc = None
    gt_bbox = None
    gt_class_ids = None
    gt_masks = None
    bbox_calc = None
    bbox_mass = None
    class_ids_mass = None
    class_ids_calc = None

    image = Kimage.img_to_array(Kimage.load_img(os.path.join(target_path, 'mammo', filename)),
                                dtype=np.uint8)

    if fliplr:
        image = np.fliplr(image).astype(np.uint8)
    if flipud:
        image = np.flipud(image).astype(np.uint8)
    if os.path.isfile(os.path.join(target_path, 'roiMass', filename)):
        roi_mask = Kimage.img_to_array(Kimage.load_img(os.path.join(target_path, 'roiMass', filename)),
                                       dtype=np.uint8)
        if fliplr:
            roi_mask = np.fliplr(roi_mask).astype(np.uint8)
        if flipud:
            roi_mask = np.flipud(roi_mask).astype(np.uint8)
        mask_mass, bbox_mass, class_ids_mass = extract_masks(roi_mask, 1)

    if os.path.isfile(os.path.join(target_path, 'roiCalc', filename)):
        roi_mask = Kimage.img_to_array(Kimage.load_img(os.path.join(target_path, 'roiCalc', filename)),
                                       dtype=np.uint8)
        if fliplr:
            roi_mask = np.fliplr(roi_mask).astype(np.uint8)
        if flipud:
            roi_mask = np.flipud(roi_mask).astype(np.uint8)
        mask_calc, bbox_calc, class_ids_calc = extract_masks(roi_mask[:, :, 0], 2)

    if mask_mass is not None and mask_calc is not None:
        #gt_masks = np.stack((mask_mass, mask_calc))
        #gt_class_ids = np.stack((class_ids_mass, class_ids_calc))
        #gt_bbox = np.stack((bbox_mass, bbox_calc))
        gt_masks = mask_mass
        gt_class_ids = class_ids_mass
        gt_bbox = bbox_mass
    elif mask_mass is not None:
        gt_masks = mask_mass
        gt_class_ids = class_ids_mass
        gt_bbox = bbox_mass
    elif mask_calc is not None:
        gt_masks = mask_calc
        gt_class_ids = class_ids_calc
        gt_bbox = bbox_calc

    return image, gt_class_ids, gt_bbox, gt_masks

def find_real_and_augmented_overlaps(r, r_aug):
    overlap_indexes = []
    if r_aug is not None and r_aug['masks'].any():
        l = r_aug['masks'].shape[-1]
        for i in range(l):
            rois_aug = extract_bbox(np.fliplr(r_aug['masks'][:, :, i]).astype(np.uint8))
            for k, box in enumerate(r['rois']):
                for box_aug in rois_aug:
                    if box_overlap(box, box_aug):
                        overlap_indexes.append(k)
    return overlap_indexes


def is_in_augmented_image(box, r_aug):
    if r_aug is not None and r_aug['masks'].any():
        l = r_aug['masks'].shape[-1]
        for i in range(l):
            rois_aug = extract_bbox(np.fliplr(r_aug['masks'][:, :, i]).astype(np.uint8))
            for k, box_aug in enumerate(rois_aug):
                if box_overlap(box, box_aug):
                    return k
    return -1

def generate_and_predict_patch(image, rois):
    for i in range(len(rois)):
        y1, x1, y2, x2 = rois[i]
        if x1 - 112 < 0:
            x1_loc = 0
            x2_loc = 224
        elif x1 + 112 > image.shape[1]:
            x1_loc = image.shape[1] - 224
            x2_loc = image.shape[1]
        else:
            x1_loc = x1 - 112
            x2_loc = x1 + 112

        if y1 - 112 < 0:
            y1_loc = 0
            y2_loc = 224
        elif y1 + 112 > image.shape[0]:
            y1_loc = image.shape[0] - 224
            y2_loc = image.shape[0]
        else:
            y1_loc = y1 - 112
            y2_loc = y1 + 112
        patch = image[y1_loc:y2_loc, x1_loc:x2_loc]
        patchX = np.expand_dims(patch, axis=0)

    predict_cnn = model_cnn.predict(patchX / 255.0)
    predic_class_cnn = int(np.argmax(predict_cnn))
    predict_score_cnn = predict_cnn[0][predic_class_cnn]
    return predic_class_cnn, predict_score_cnn


def compare_pooling(image, r, r_aug):

    if r is not None and r['rois'] is not None:
        #rois = r['rois']
    #elif r is not None and r_aug['rois'] is not None:
        #rois = r_aug['rois']
    #else:
        #return 0, 1.0

        best_patch_score = -1
        best_patch_class = -1
        patch_score = 1.0
        patch_class = 0

        for i in range(len(r['rois'])):
            y1, x1, y2, x2 = r['rois'][i]
            if x1 - 112 < 0:
                x1_loc = 0
                x2_loc = 224
            elif x1 + 112 > image.shape[1]:
                x1_loc = image.shape[1] - 224
                x2_loc = image.shape[1]
            else:
                x1_loc = x1 - 112
                x2_loc = x1 + 112

            if y1 - 112 < 0:
                y1_loc = 0
                y2_loc = 224
            elif y1 + 112 > image.shape[0]:
                y1_loc = image.shape[0] - 224
                y2_loc = image.shape[0]
            else:
                y1_loc = y1 - 112
                y2_loc = y1 + 112
            patch = image[y1_loc:y2_loc, x1_loc:x2_loc]
            patchX = np.expand_dims(patch, axis=0)

            predict_cnn = model_cnn.predict(patchX / 255.0)
            predic_class_cnn = int(np.argmax(predict_cnn))
            predict_score_cnn = predict_cnn[0][predic_class_cnn]
            if predict_score_cnn < 0.86:
                predict_score_cnn = 1.0
                predic_class_cnn = 2 #Normal

            if r_aug is not None:
                k = is_in_augmented_image(r['rois'][i], r_aug)
            else:
                k = -1

            if k >= 0:
                if class_names_mrcnn[r['class_ids'][i]] == class_names_cnn[predic_class_cnn] == class_names_mrcnn[r_aug['class_ids'][k]]:
                    patch_score = max(r['scores'][i], predict_score_cnn, r_aug['scores'][k])
                    patch_class = r['class_ids'][i]
                elif class_names_mrcnn[r['class_ids'][i]] == class_names_mrcnn[r_aug['class_ids'][k]]:
                    patch_score = max(r['scores'][i], r_aug['scores'][k])
                    patch_class = r['class_ids'][i]
                elif class_names_mrcnn[r['class_ids'][i]] == class_names_cnn[predic_class_cnn]:
                    patch_score = max(r['scores'][i], predict_score_cnn)
                    patch_class = r['class_ids'][i]
                elif class_names_cnn[predic_class_cnn] == class_names_mrcnn[r_aug['class_ids'][k]]:
                    patch_score = max(predict_score_cnn, r_aug['scores'][k])
                    patch_class = r_aug['class_ids'][k]
                else:
                    patch_score = 1.0
                    patch_class = 0 #Normal
            elif k < 0:
                if class_names_mrcnn[r['class_ids'][i]] == class_names_cnn[predic_class_cnn]:
                    patch_score = max(r['scores'][i], predict_score_cnn)
                    patch_class = r['class_ids'][i]
                else:
                    patch_score = 1.0
                    patch_class = 0 #Normal


            if patch_class != 0 and best_patch_score < patch_score:
                best_patch_score = patch_score
                best_patch_class = patch_class


            print('+MRCNN:')
            print('class: {}'.format(class_names_mrcnn[r['class_ids'][i]]))
            print('scores: {}'.format(r['scores'][i]))
            print('+AUG MRCNN:')
            if k > -1:
                print('class: {}'.format(class_names_mrcnn[r_aug['class_ids'][k]]))
                print('scores: {}'.format(r_aug['scores'][k]))
            print('+xception')
            print('percent: {}'.format(predict_score_cnn))
            print('class: {}'.format(class_names_cnn[predic_class_cnn]))

        if best_patch_class < 0:
            best_patch_class = 0
            best_patch_score = 1
        print('=>result')
        print('percent: {}'.format(best_patch_score))
        print('class: {}'.format(class_names_mrcnn[best_patch_class]))
        print()

    return best_patch_class, best_patch_score


def classifier(model_mrcnn, model_cnn, target_path, class_names, advanced_detection=True):
    y_pred = []
    y_true = []
    y_true_binary = []
    y_true_class = []
    y_pred_score = []
    with open(os.path.join(REPORTS_DIR, 'report.csv'), 'w', newline='', encoding='utf-8') as rpt:
        report = csv.writer(rpt)
        report.writerow(['filename', 'predicted class', 'score', 'true class'])
        rpt.flush()

        for filename in os.listdir(os.path.join(target_path, 'mammo')):
            if not os.path.isfile(os.path.join(target_path, 'mammo', filename)):
                continue
            if filename[-4:] !='.png':
                continue
            r_aug = None
            if advanced_detection:
                image_lr, gt_class_ids_lr, gt_bbox_lr, gt_masks_lr = load_image_gt(target_path, filename, fliplr=True)
                augmentation = iaa.Sequential([
                    iaa.GaussianBlur(sigma=(0.5, 0.9)),
                    #iaa.GaussianBlur(sigma=(0.2, 0.7)),
                    iaa.ContrastNormalization((0.75, 1.5)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                ])
                image_lr = augmentation.augment_image(image_lr).astype(np.uint8)
                #Kimage.array_to_img(image_lr).save(os.path.join(RESULTS_DIR, 'mrcnn', filename[:-4]+'_aug'+filename[-4:]))

                r_aug = model_mrcnn.detect([image_lr], verbose=0)[0]

            image, gt_class_ids, gt_bbox, gt_masks = load_image_gt(target_path, filename)
            if gt_masks is not None and not gt_masks.any():
                continue

            #Kimage.array_to_img(image).save(os.path.join(RESULTS_DIR, 'mrcnn', filename))
            r = model_mrcnn.detect([image], verbose=0)[0]

            if advanced_detection:
                pred_mammo_class_id, pred_mammo_score = compare_pooling(image, r, r_aug)
            else:
                if r['class_ids'] is not None and r['class_ids'].any():
                    pred_mammo_class_id = r['class_ids'][0]
                    pred_mammo_score = r['scores'][0]
                else:
                    pred_mammo_class_id = 0
                    pred_mammo_score = 1.0

            gt_mammo_class_id = gt_class_ids[0] if gt_class_ids is not None else 0
            y_pred.append(pred_mammo_class_id)
            if gt_mammo_class_id == 0:
                y_true_class.append([1, 0, 0])
            elif gt_mammo_class_id == 1:
                y_true_class.append([0, 1, 0])
            elif gt_mammo_class_id == 2:
                y_true_class.append([0, 0, 1])

            if pred_mammo_class_id == 0:
                y_pred_score.append([pred_mammo_score, 0, 0])
            elif pred_mammo_class_id == 1:
                y_pred_score.append([0, pred_mammo_score, 0])
            elif pred_mammo_class_id == 2:
                y_pred_score.append([0, 0, pred_mammo_score])



            image_res = image.copy()
            image_org = image.copy()
            gt_match = None
            overlaps = None
            pred_match = None
            if gt_masks is not None:
                gt_match, pred_match, overlaps = utils.compute_matches(
                    gt_bbox, gt_class_ids, gt_masks,
                    r['rois'], r['class_ids'], r['scores'], r['masks'],
                    iou_threshold=0.1, score_threshold=0.1)
                for box in gt_bbox:
                    y1, x1, y2, x2 = box
                    cv2.rectangle(image_org, (x1, y1), (x2, y2), (0, 255, 0), 1)

            #Kimage.array_to_img(image_org).save(os.path.join(RESULTS_DIR, 'mrcnn', filename[:-4] + '_org' + filename[-4:]))

            #gt_mammo_class_id = gt_class_ids[0] if gt_class_ids is not None else 0
            y_true.append(gt_mammo_class_id)
            if gt_mammo_class_id == 0:
                y_true_binary.append([1, 0, 0])
            elif gt_mammo_class_id == 1:
                y_true_binary.append([0, 1, 0])
            elif gt_mammo_class_id == 2:
                y_true_binary.append([0, 0, 1])

            if advanced_detection:
                for j in range(r['rois'].shape[0]):
                    y1, x1, y2, x2 = r['rois'][j]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

                for j in range(r_aug['rois'].shape[0]):
                    y1, x1, y2, x2 = r_aug['rois'][j]
                    cv2.rectangle(image_lr, (x1, y1), (x2, y2), (0, 255, 0), 1)

                #for i in overlap_indexes:
                #    y1, x1, y2, x2 = r['rois'][i]
                #    cv2.rectangle(image_res, (x1, y1), (x2, y2), (0, 0, 255), 1)

                #Kimage.array_to_img(image_lr).save(
                #        os.path.join(RESULTS_DIR, 'classifier', filename[:-4] + '_aug_bbox' + filename[-4:]))

                #Kimage.array_to_img(image_res).save(
                #        os.path.join(RESULTS_DIR, 'classifier', filename[:-4] + '_res_bbox' + filename[-4:]))

            else:
                for roi in r['rois']:
                    y1, x1, y2, x2 = roi
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

            #Kimage.array_to_img(image).save(
            #        os.path.join(RESULTS_DIR, 'classifier', filename[:-4] + '_bbox' + filename[-4:]))

            if gt_masks is None:
                gt_mammo_class_id = 0

            print("{} Class: {} => {} Score: {:.6f}".format(filename, gt_mammo_class_id, pred_mammo_class_id, pred_mammo_score))
            report.writerow([filename, class_names[pred_mammo_class_id], pred_mammo_score, class_names[gt_mammo_class_id]])

    print()
    print()
    #y_true_binary = y_true
    plt = plot_confusion_matrix(y_true, y_pred, class_names)
    plt.savefig(os.path.join(REPORTS_DIR, str(get_report_seq('mat'))+'-mat.png'))
    plt.show()

    plt = plot_roc_chart(y_true_binary, y_pred_score)
    plt.savefig(os.path.join(REPORTS_DIR, str(get_report_seq('roc'))+'-roc.png'))
    plt.show()

    plt = plot_micro_averaged_precision(y_true_binary, y_pred_score)
    plt.savefig(os.path.join(REPORTS_DIR, str(get_report_seq('map'))+'-map.png'))
    plt.show()

    '''
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)
    plot_confusion_matrix(cm, np.array(class_names))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true_class = np.array(y_true_class)
    y_pred_score = np.array(y_pred_score)
    for i in range(3):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(y_true_class[:, i], y_pred_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    auc_score = float(sum(roc_auc.values())) / len(roc_auc)
    print(auc_score)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(3):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='{0} (area = {1:0.3f})'
                       ''.format(class_names[i], roc_auc[i]))

    plt.title('ROC curve (AUC = {:.2f})'.format(auc_score, 4))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(REPORTS_DIR, str(get_report_seq('roc')) + '-roc.png'))
    plt.show()
    '''


def compute_batch_ap(model, dataset, image_ids, verbose=1):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        #results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
        # Compute AP over range 0.5 to 0.95
        #r = results[0]
        # Detect objects
        r = model.detect([image], verbose=0)[0]

        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            verbose=0)
        APs.append(ap)
        if verbose:
            info = dataset.image_info[image_id]
            meta = modellib.parse_image_meta(image_meta[np.newaxis,...])
            print("{:3} {}   AP: {:.2f}".format(
                meta["image_id"][0], meta["original_image_shape"][0], ap))
    return APs


############################################################
#  Command Line
############################################################


if __name__ == '__main__':

    mrcnn_weights_path = os.path.join(MODELS_DIR, 'mrcnn', MRCNN_WEIGHTS+'.h5')
    cnn_weights_path = os.path.join(MODELS_DIR, 'cnn', CNN_MODEL+'-weights.h5')
    cnn_model_path = os.path.join(MODELS_DIR, 'cnn', CNN_MODEL+'.h5')

    print()
    print()
    print("=> Dataset Path:")
    print(DATASET_PATH)
    print('\n')
    config = MammoInferenceConfig(DATASET_PATH)
    model_mrcnn = modellib.MaskRCNN(mode="inference", config=config, model_dir='./logs')

    # Load weights
    print("=> Loading Mask RCNN Weights: ")
    print(mrcnn_weights_path)
    print('\n')
    model_mrcnn.load_weights(mrcnn_weights_path, by_name=True)

    print('-> Loading CNN Model...')
    print(cnn_model_path)
    print('\n')
    model_cnn = models.load_model(cnn_model_path)

    print('-> Loading CNN Weights...')
    print(cnn_weights_path)
    print('\n')
    model_cnn.load_weights(cnn_weights_path)

    # Read dataset
    dataset = MammoDataset()
    dataset.load_dataset(DATASET_PATH, "Test")
    dataset.prepare()
    class_names = dataset.class_names.copy()
    class_names_cnn = sorted(['Calc', 'Mass', 'Normal'])
    class_names_mrcnn = dataset.class_names.copy()
    class_names_mrcnn[0] = 'Normal'
    #class_names[2] = 'Normal'

    classifier(model_mrcnn, model_cnn, CLASSIFY_DATASET_PATH, class_names_mrcnn, advanced_detection=True)

    APs = compute_batch_ap(model_mrcnn, dataset, dataset.image_ids)
    print("Mean AP overa {} images: {:.4f}".format(len(APs), np.mean(APs)))


    if sys.platform == 'win32':
        winsound.Beep(2500, 1000)
