# Import the required libraries
import cv2
import pandas as pd
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", dest='id', type=str)
    args = parser.parse_args()
    return args


def check_result(**kwargs):
    # read input image
    image_id = kwargs.get('id', '0000')
    image_path = '/opt/ml/dataset/test/'
    annotation_path = '/opt/ml/yolov8/yolo_submission.csv'
    result_path = '/opt/ml/temp/check/'

    image = cv2.imread(image_path + image_id + '.jpg')
    annotations_raw = pd.read_csv(annotation_path).iloc[int(image_id) + 1]["PredictionString"].split()
    annotations = [annotations_raw[i:i + 6] for i in range(0, len(annotations_raw), 6)]
    # print(annotations)

    # bounding box in (xmin, ymin, xmax, ymax) format
    # top-left point=(xmin, ymin), bottom-right point = (xmax, ymax)
    for annotation in annotations:
        category = int(annotation[0])
        confidence = float(annotation[1])
        bbox = [int(float(s)) for s in annotation[2:]]
        # print(annotation, bbox_raw)
        # draw bounding box on the input image
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # transform it to PIL image and display
    # cv2.imshow("Image with Bounding Boxes", image)
    # cv2.waitKey(0)  # Wait for a key press to close the window

    # Alternatively, save the image to a file
    cv2.imwrite(result_path + image_id + '.jpg', image)


if __name__ == '__main__':
    check_result(id=get_args().id)