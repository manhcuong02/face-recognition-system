
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_box_and_label(image, lw = 1, box = None, label= None, color=(128, 128, 128), txt_color=(255, 255, 255), landmark = None):
    # Add one xyxy box to image with label
    if box is not None:
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        
    if landmark is not None:
        for p in landmark:
            p = p.astype(np.int16)
            cv2.circle(image, p, 2, (0, 0, 255), -1)
    
    if label is not None:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)
    return image

def plot_confusion_matrix(y_true, y_pred, label, title):
    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Hiển thị confusion matrix bằng seaborn và matplotlib
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label, yticklabels=label)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)