from model.models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def configuration():
    config_path='../config/yolov3.cfg'
    weights_path='../config/yolov3.weights'
    class_path='../config/coco.names'
    img_size=416
    conf_thres=0.8
    nms_thres=0.4

    # Load model and weights
    model = Darknet(config_path, img_size=img_size)
    model.load_weights(weights_path)

    # define the device to run the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # model.cpu()
    model.eval()
    classes = utils.load_classes(class_path)

    # Tensor = torch.cuda.FloatTensor
    Tensor = torch.FloatTensor

    return img_size, Tensor, model, classes


def detect_image(img, img_size, Tensor, model):
    # scale and pad image
    conf_thres = 0.8
    nms_thres = 0.4

    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


def draw_bounding_box(img_path):
    # load image and get detections
    # img_path = "images/blueangels.jpg"

    img_size, Tensor, model, classes = configuration()

    prev_time = time.time()
    img = Image.open(img_path)
    detections = detect_image(img, img_size, Tensor, model)

    # testers
    print('detections: ', detections)

    inference_time = datetime.timedelta(seconds=time.time() - prev_time)
    print('Inference Time: %s' % (inference_time))

    # Get bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    img = np.array(img)
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        # browse detections and draw bounding boxes

        i = 0

        detected_objects = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if i >= 0 and i <= 5:
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(bbox)
                plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                         bbox={'color': color, 'pad': 0})

                y2 = ((y2 - pad_y // 2) / unpad_h) * img.shape[0]
                x2 = ((x2 - pad_x // 2) / unpad_w) * img.shape[1]

                plt.scatter(x1, y1, color='r')
                plt.scatter(x2, y2, color='g')

                obj = {}
                obj['name'] = classes[int(cls_pred)]
                # x,y points in the actual image, whatever resolution it has.
                obj['tlx'] = x1
                obj['tly'] = y1
                obj['brx'] = x2
                obj['bry'] = y2

                detected_objects.append(obj)

            i += 1
    plt.axis('off')
    # save image
    plt.savefig(img_path.replace(".jpg", "-det.jpg"), bbox_inches='tight', pad_inches=0.0)
    plt.show()

    return detected_objects


# call this function with the image path.. it will do rest of the things.
# runs with CPU and GPU both.
# once sample example is shown below.
# the below function return the detection, where each row is <x1, y1, x2, y2, conf, cls_conf, cls_pred>
# the first four elements are the x1, y1, x2, y2 coordinates.

detected_objs = draw_bounding_box('../images/blueangels.jpg')
print(detected_objs)

