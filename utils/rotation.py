import numpy as np
import cv2

def calculate_rotation_matrix(angle, image_width, image_height):
    matrix = cv2.getRotationMatrix2D((image_width//2, image_height//2), angle, 1)
    return matrix

def rotate_box(x1, y1, x2, y2, matrix, image_width, image_height):
    # Chuyển thành tâm và kích thước
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    size = (x2 - x1, y2 - y1)

    # Áp dụng ma trận xoay cho tâm và kích thước
    
    rotated_center = np.dot(matrix, np.array([center_x, center_y, 1]))

    # Chuyển trở lại từ tâm và kích thước thành (x1, y1, x2, y2)
    rotated_x1 = rotated_center[0] - size[0] // 2
    rotated_y1 = rotated_center[1] - size[1] // 2
    rotated_x2 = rotated_center[0] + size[0] / 2
    rotated_y2 = rotated_center[1] + size[1] / 2

    # Kiểm tra xem có vượt quá biên hình ảnh không
    rotated_x1 = max(0, rotated_x1)
    rotated_y1 = max(0, rotated_y1)
    rotated_x2 = min(image_width, rotated_x2)
    rotated_y2 = min(image_height, rotated_y2)

    return rotated_x1, rotated_y1, rotated_x2, rotated_y2

def rotate_point(x, y, matrix):
    new_point = np.dot(matrix, np.array([x, y, 1]))
    return new_point

def rotate_landmark(landmark, matrix):
    new_landmark = []
    
    for x, y in landmark:
        x, y = rotate_point(x, y, matrix)
        new_landmark.append([x, y])

    return np.array(new_landmark, dtype = np.int16)

def rotate_image(image, rotation_matrix):
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

