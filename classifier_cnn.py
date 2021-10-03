#
#Siavash Salemi
#
import numpy as np
import os
from keras.models import load_model
import argparse
import cv2
import matplotlib.pyplot as plt
import sys
from sklearn import metrics
import itertools
import keras.preprocessing.image as Kimage
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle

if sys.platform == 'win32':
    import winsound


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


def mask_overlap(mask, cutoff=.6):
    add_val = 1000
    roi_area = (mask > 0).sum()
    roi_patch_added = mask.copy().astype('float32')
    roi_patch_added += add_val
    patch_area = (roi_patch_added >= add_val).sum()
    inter_area = (roi_patch_added > add_val).sum().astype('float32')
    return (inter_area/roi_area > cutoff or inter_area/patch_area > cutoff)


def mammography_visualizer(dataset_path, img_filename, model, patch_size=224, stride=5, sensitivity=0.84):

    img_original = Kimage.img_to_array(Kimage.load_img(os.path.join(dataset_path, 'mammo', img_filename)), dtype=np.uint8)
    img_width = img_original.shape[1]
    img_height = img_original.shape[0]
    img_mask_calc = np.zeros(img_original.shape, dtype='uint8')
    img_mask_mass = np.zeros(img_original.shape, dtype='uint8')

    if os.path.isfile(os.path.join(dataset_path, 'roiCalc', img_filename)):
        img_mask_calc = Kimage.img_to_array(Kimage.load_img(os.path.join(dataset_path, 'roiCalc', img_filename)), dtype=np.uint8)
    if os.path.isfile(os.path.join(dataset_path, 'roiMass', img_filename)):
        img_mask_mass = Kimage.img_to_array(Kimage.load_img(os.path.join(dataset_path, 'roiMass', img_filename)), dtype=np.uint8)

    ture_mammo_class = 0
    if (img_mask_mass == 0).all() and (img_mask_calc == 0).all():
        print('[normal]'.format(img_filename))
        y_true_binary.append([0, 0, 1])
        ture_mammo_class = 2
        y_true.append(ture_mammo_class)
    elif (img_mask_mass == 0).all():
        print('[Calc]'.format(img_filename))
        y_true_binary.append([1, 0, 0])
        ture_mammo_class = 0
        y_true.append(ture_mammo_class)
    else:
    #elif (img_mask_calc == 0).all():
        print('[Mass]'.format(img_filename))
        y_true_binary.append([0, 1, 0])
        ture_mammo_class = 1
        y_true.append(ture_mammo_class)


    img_result = np.copy(img_original)
    p = 0
    pred_mammo_score = 1
    pred_mammo_class = 2
    patch_list = []
    for row in range(0, img_height - patch_size, stride):
        col = 0
        while col <= img_original.shape[1] - patch_size:

            patch = img_original[row:row + patch_size, col:col + patch_size, 0]
            roi_mass = img_mask_mass[row:row + patch_size, col:col + patch_size, 0]
            roi_calc = img_mask_calc[row:row + patch_size, col:col + patch_size, 0]
            patchX = np.zeros((1, patch.shape[0], patch.shape[1], 3), dtype='uint8')
            patchX[:, :, :, 0] = patchX[:, :, :, 1] = patchX[:, :, :, 2] = patch
            patch_rgb = np.zeros((patch.shape[0], patch.shape[1], 3), dtype='uint8')
            patch_rgb[:, :, 0] = patch_rgb[:, :, 1] = patch_rgb[:, :, 2] = patch

            #predict = model.predict(np.expand_dims(np.expand_dims(patchX / 255.0, axis=0), axis=3))

            predict = model.predict(patchX/255.0)
            class_idx = int(np.argmax(predict))
            predict_percent = predict[0][class_idx]

            if class_names[class_idx] == 'Mass' and predict_percent > 0.84:
                #print('Mass {:.2f}'.format(predict_percent))
                p += 1
                if (roi_mass == 0).all() and (roi_calc == 0).all():
                    Kimage.save_img(os.path.join(temp_path, 'Normal', '{0}-f{1}.png'.format(img_filename, p)),
                                    Kimage.array_to_img(patch_rgb))
                elif mask_overlap(roi_mass, 0.8):
                    Kimage.save_img(os.path.join(temp_path, 'Mass', '{0}-f{1}.png'.format(img_filename, p)),
                                    Kimage.array_to_img(patch_rgb))
                    patch_list.append({'row': row, 'col': col, 'class_idx': class_idx,
                                       'class_names': class_names[class_idx], 'predict_percent': predict_percent})
                    print('({0}, {1}) - {2}, {3}, {4}'.format(str(col), str(row), class_idx, class_names[class_idx],
                                                              predict_percent))

                else:
                    class_idx = 2
                if predict_percent > pred_mammo_score or pred_mammo_class == 2:
                    pred_mammo_score = predict_percent
                    pred_mammo_class = class_idx

                #img_result[row:row + patch_size, col:col + patch_size, 1] = int(2.55*(int(predict_percent*100)))

            elif class_names[class_idx] == 'Calc' and predict_percent > 0.84:
                #print('Calc {:.2f}'.format(predict_percent))
                p += 1
                if (roi_calc == 0).all() and (roi_mass == 0).all():
                    Kimage.save_img(os.path.join(temp_path, 'Normal', '{0}-f{1}.png'.format(img_filename, p)),
                                    Kimage.array_to_img(patch_rgb))
                elif mask_overlap(roi_calc, 0.8):
                    Kimage.save_img(os.path.join(temp_path, 'Calc', '{0}-f{1}.png'.format(img_filename, p)),
                                    Kimage.array_to_img(patch_rgb))
                    patch_list.append({'row': row, 'col': col, 'class_idx': class_idx,
                                       'class_names': class_names[class_idx], 'predict_percent': predict_percent})
                    print('({0}, {1}) - {2}, {3}, {4}'.format(str(col), str(row), class_idx, class_names[class_idx],
                                                              predict_percent))
                else:
                    class_idx = 2
                if predict_percent > pred_mammo_score or pred_mammo_class == 2:
                    pred_mammo_score = predict_percent
                    pred_mammo_class = class_idx

            elif class_names[class_idx] == 'Normal' and predict_percent > sensitivity:
                col += patch_size // 2
                print('.', end='')
            else:
                print('!', end='')

            col += stride


    y_pred.append(pred_mammo_class)
    if pred_mammo_class == 0:
        y_pred_score.append([pred_mammo_score, 0, 0])
        y_pred_class.append([1, 0, 0])
    elif pred_mammo_class == 1:
        y_pred_score.append([0, pred_mammo_score, 0])
        y_pred_class.append([0, 1, 0])
    elif pred_mammo_class == 2:
        y_pred_score.append([0, 0, pred_mammo_score])
        y_pred_class.append([0, 0, 1])
    else:
        print('*')



    print('Found= ', str(len(patch_list)))
    print()
    print('-> Analyze')

    for i in range(0, len(patch_list) - 1):
        for j in range(i + 1, len(patch_list)):
            if patch_list[i]['col'] <= patch_list[j]['col'] <= (patch_list[i]['col'] + patch_size-10) and \
                    patch_list[i]['row'] <= patch_list[j]['row'] <= (patch_list[i]['row'] + patch_size-10) and \
                    patch_list[i]['class_idx'] == patch_list[j]['class_idx']:
                if patch_list[i]['predict_percent'] <= patch_list[j]['predict_percent']:
                    patch_list[i]['predict_percent'] = 0
                elif patch_list[i]['predict_percent'] > patch_list[j]['predict_percent']:
                    patch_list[j]['predict_percent'] = 0
            elif patch_list[i]['col'] <= patch_list[j]['col'] + patch_size-10 <= (patch_list[i]['col'] + patch_size-10) and \
                    patch_list[i]['row'] <= patch_list[j]['row'] <= (patch_list[i]['row'] + patch_size-10) and \
                    patch_list[i]['class_idx'] == patch_list[j]['class_idx']:
                if patch_list[i]['predict_percent'] <= patch_list[j]['predict_percent']:
                    patch_list[i]['predict_percent'] = 0
                elif patch_list[i]['predict_percent'] > patch_list[j]['predict_percent']:
                    patch_list[j]['predict_percent'] = 0

    patch_list2 = []
    for p in patch_list:
        if p['predict_percent'] >= sensitivity:
            if p['class_names'] == 'Mass':
                color = (200, 0, 0)
            else:
                color = (0, 200, 0)

            cv2.rectangle(img_result, (p['col'], p['row']), (p['col'] + patch_size, p['row'] + patch_size), color,
                          1)

            cv2.rectangle(img_result, (p['col'], p['row']), (p['col'] + 40, p['row'] + 13), color, -1)
            cv2.putText(img_result, p['class_names'], (p['col'], p['row'] + 12), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)

            #detected_patch = Image.fromarray(img_result).convert(mode='RGB')

            print('({0}, {1}) - {2}, {3}, {4}'.format(
                str(p['col']), str(p['row']), p['class_idx'], p['class_names'], p['predict_percent']))

            patch_list2.append(p)

    #regex = re.search('(\w+)\.png', input_img, flags=re.U)

    #detected_patch.save(os.path.join(target_path, 'Calc', '{0}.png'.format(img_filename)))
    print('Detected=', len(patch_list2))
    return img_result


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
MODEL_NAME = 'Xception'
#dataset_name = 'Patch-300rnd-v200716'
dataset_name = 'Trainable-1024'
model_filename = MODEL_NAME + '-Patch-1024.h5'
weight_filename = MODEL_NAME + '-Patch-1024-best-weights.h5'

base_dir = os.path.dirname(os.path.realpath(__file__))
FOLDER_TRAIN_SET = 'Test'
dataset_dir = os.path.join(r'D:\Master of Science\Datasets\INBreast\PNG-Dataset-v5', dataset_name, FOLDER_TRAIN_SET)

#classes = {'Normal': 0, 'Mass': 1, 'Calc': 2}
#classes_reversed = {value: key for key, value in classes.items()}

class_names = sorted(['Calc', 'Mass', 'Normal'])

report_path = os.path.join(base_dir, 'report', 'classifier', MODEL_NAME)
os.makedirs(report_path, 755, 1)

temp_path = os.path.join(base_dir, 'temp', 'classifier', MODEL_NAME)
os.makedirs(temp_path, 755, 1)
for cls in class_names:
    os.makedirs(os.path.join(temp_path, cls), 755, 1)
#os.makedirs(os.path.join(report_path, 'Mass'), 755, 1)
#os.makedirs(os.path.join(report_path, 'Normal'), 755, 1)

model_file = os.path.join(base_dir, 'models', model_filename)
weight_file = os.path.join(base_dir, 'weights', weight_filename)


print('-> Loading Model...\n\n')
model = load_model(model_file)
print('-> Loading Weights...\n\n')
model.load_weights(weight_file)

#source_path = os.path.join(dataset_dir, 'mammo')

if not os.path.isdir(os.path.join(dataset_dir, 'mammo')):
    exit()

y_pred = []
y_true = []
y_true_binary = []
y_pred_score = []
y_pred_class = []
#for png_file in glob.glob(os.path.join(dataset_dir, 'mammo') + '/*.png', recursive=False):
for filename in os.listdir(os.path.join(dataset_dir, 'mammo')):
    if not os.path.isfile(os.path.join(dataset_dir, 'mammo', filename)):
        continue
    if filename[-4:] != '.png':
        continue
    #if not os.path.isfile(png_file):
    #    continue
    #png_filename, _ = os.path.splitext(os.path.basename(png_file))
    print()
    print('=> {0}'.format(filename))
    img_result = mammography_visualizer(dataset_dir, filename, model, patch_size=224, stride=76, sensitivity=0.86)
    #img = Kimage.array_to_img(img_result)
    Kimage.save_img(os.path.join(temp_path, '{0}'.format(filename)), Kimage.array_to_img(img_result))
    #img.save(os.path.join(report_path, '{0}.png'.format(png_filename)))



plt = plot_confusion_matrix(y_true, y_pred, class_names)
plt.savefig(os.path.join(report_path, MODEL_NAME + '-mat.png'))
plt.show()

plt = plot_roc_chart(y_true_binary, y_pred_score)
plt.savefig(os.path.join(report_path, MODEL_NAME + '-roc.png'))
plt.show()

plt = plot_micro_averaged_precision(y_true_binary, y_pred_score)
plt.savefig(os.path.join(report_path, MODEL_NAME + '-map.png'))
plt.show()

if sys.platform == 'win32':
    winsound.Beep(2500, 1000)
