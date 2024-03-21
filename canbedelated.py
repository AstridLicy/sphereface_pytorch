import os
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from matlab_cp2tform import get_similarity_transform_for_cv2

def alignment(src_img, src_pts, crop_size=(96, 112), padding=2):
    ref_pts = [[30.2946+padding, 51.6963+padding],
               [65.5318+padding, 51.5014+padding],
               [48.0252+padding, 71.7366+padding],
               [33.5493+padding, 92.3655+padding],
               [62.7299+padding, 92.2041+padding]]
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def process_dataset(dataset_path, output_path):
    detector = MTCNN()
    # print(sorted(os.listdir(dataset_path)))
    for label in sorted(os.listdir(dataset_path)):
        if label[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
            continue
        print('Processing:', label)
        label_path = os.path.join(dataset_path, label)
        output_label_path = os.path.join(output_path, label)
        if not os.path.exists(output_label_path):
            os.makedirs(output_label_path)

        # if not os.listdir(output_label_path):
        #     os.rmdir(output_label_path)
        if os.path.isdir(label_path):
            for image_name in sorted(os.listdir(label_path)):
                image_path = os.path.join(label_path, image_name)
                output_image_path = os.path.join(output_label_path, image_name)
                if os.path.exists(output_image_path):
                    print('Skip:', output_image_path)
                    continue
                img = cv2.imread(image_path)
                result = detector.detect_faces(img)

                if result:
                    keypoints = result[0]['keypoints']
                    src_pts = [
                        keypoints['left_eye'],
                        keypoints['right_eye'],
                        keypoints['nose'],
                        keypoints['mouth_left'],
                        keypoints['mouth_right']
                    ]
                    aligned_img = alignment(img, src_pts)
                    cv2.imwrite(os.path.join(output_label_path, image_name), aligned_img)
                    print('Processed:', output_image_path)

# 调用函数
dataset_path = '/home/chiyunli/dataset/casia'  # 替换为您的数据集路径
output_path = '/home/chiyunli/dataset/casia_align_crop'  # 输出处理后图像的路径
process_dataset(dataset_path, output_path)


