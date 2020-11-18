import cv2
import numpy as np
import scipy as sp
import pdb
import numpy as np

def segment_image_dnn(image_, threshold):

    labelsPath = "mask_rcnn/mscoco_labels.names"
    colorsPath = "mask_rcnn/colors.txt"
    LABELS = open(labelsPath).read().strip().split("\n")

    # derive the paths to the Mask R-CNN weights and model configuration
    weightsPath = "mask_rcnn/frozen_inference_graph.pb"
    configPath = "mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28 .pbtxt"

    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
    # Load the colors
    image= image_.copy()

    ## initilize new mask
    mask_with_segm_num =  np.zeros(image.shape[:2], dtype=np.float32)
    with open(colorsPath, 'rt') as f:
        colorsStr = f.read().rstrip('\n').split('\n')
    COLORS = []
    for i in range(len(colorsStr)):
        rgb = colorsStr[i].split(' ')
        color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
        COLORS.append(color)

    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final",
                                  "detection_masks"])

    idxs = np.argsort(boxes[0, 0, :, 2])[::-1]
    (H, W) = image.shape[:2]
    # initialize the mask, ROI, and coordinates of the person for the
    # current frame
    mask = None
    roi = None
    coords = None
    # loop over the indexes
    for i in idxs:
        # extract the class ID of the detection along with the
        # confidence (i.e., probability) associated with the
        # prediction
        classID = int(boxes[0, 0, i, 1])
        if not LABELS[classID] == "person":
            continue
        confidence = boxes[0, 0, i, 2]
        # if the detection is not the 'person' class, ignore it
        if  threshold < confidence:

            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),interpolation=cv2.INTER_LINEAR_EXACT)
            mask = (mask > threshold)
            # extract the ROI of the image but *only* extracted the
            # masked region of the ROI
            roi = image[startY:endY, startX:endX][mask]
            color = COLORS[classID %6]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
            # store the blended ROI in the original frame
            image[startY:endY, startX:endX][mask] = blended
            mask_with_segm_num[startY:endY, startX:endX][mask] =255
            #draw the bounding box of the instance on the frame
            #color = [int(c) for c in color]

    cv2.imwrite("mask_with_segm_num.png",mask_with_segm_num)



