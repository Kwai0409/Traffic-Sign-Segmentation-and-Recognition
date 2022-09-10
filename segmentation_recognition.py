import argparse
import time
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime

######### helper function #############
def largestContour(mask):
    """
    input: mask in gray scale
    output: largest contour
    """
    
    mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) 
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    #if no contour
    if len(contours)==0:
        return None
    
    #find largest contour
    length = []
    for i in contours:
        length.append(len(i))
    Idx_max = np.argmax(length)
    cnt = contours[Idx_max]
    
    return cnt
    
def drawBbox(img, xywh, color = (0,255,0), thickness=3, copy=True):
    """
    input: img in BGR
    output: img with bounding box
    """
    
    #if no contour
    if xywh is None:
        return img
    
    if copy == True:
        img_c = img.copy()
    else:
        img_c = img
        
    x, y, w, h = xywh
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    bbox = cv.rectangle(img_c, (x,y), (x+w, y+h), color, thickness)
    return bbox   
    
def trueBBox(cnt1, cnt2):
    """this function is to select bounding box based on aspect ratio"""
    x1, y1, w1, h1 = cv.boundingRect(cnt1)
    x2, y2, w2, h2 = cv.boundingRect(cnt2)
    img_size = 200
    
    #if size of bounding box too small, select the other one
    if w1 < img_size*0.4 or h1 < img_size*0.4: #if width or height smaller than 40%
        return cnt2
    if w2 < img_size*0.4 or h2 < img_size*0.4: #if width or height smaller than 40%
        return cnt1
    
    #calculate aspect ratio
    if h1 > w1:
        aspect_ratio1 = w1/h1
    else:
        aspect_ratio1 = h1/w1
    
    if h2 > w2:
        aspect_ratio2 = w2/h2
    else:
        aspect_ratio2 = h2/w2
    
    #compare aspect ratio, choose higher one
    if aspect_ratio1 > aspect_ratio2:
        return cnt1
    else:
        return cnt2   
   
def readGroundtruths(path):
    """this function is to read annotations from the given path"""
    annotations = []
    with open(path) as file:
           for line in file:
                row = line.strip().split(";")
                annotations.append(row)
    return annotations

def getOriSizeBBox(cnt, ori_height, ori_width, resized_size=(200, 200)):
    """this function is to convert bounding box for resized image to bounding box for original size image
    input: contour, groundtruth, size resized
    output: bounding box for original size image
    """
    
    #information resized image
    x, y, w, h = cv.boundingRect(cnt) #bounding box for resized size 
    new_width, new_height = resized_size
    
    #calculate bounding box for original size image
    new_width_ratio = float(ori_width/new_width)
    new_height_ratio = float(ori_height/new_height)
    x *= new_width_ratio
    y *= new_height_ratio
    w *= new_width_ratio
    h *= new_height_ratio
        
    return (x,y,w,h)

def getGroundtruth(imgName, annotations=None):
    annotationFound = False
    """this function will return resized predicted bounding box, actual bounding box, actual label and annotation for the image"""
    for annotation in annotations:
        if annotation[0] == imgName:
            annotationFound = True
            break
    if annotationFound == False:
        print(imgName, " groundtruth not found !!!")

    #get actual bbox
    (x,y,x2,y2) = (int(annotation[3]), int(annotation[4]), int(annotation[5]), int(annotation[6]))
    w = x2-x
    h = y2-y
    xywh_actual = (x,y,w,h)
    
    return xywh_actual, annotation

def accuracy(predictedXYWH, annotation):
    """this function will return IoU, pixel_acc"""
    predicted_x, predicted_y, predicted_w, predicted_h = predictedXYWH
    
    #predicted boundingbox width
    predicted_x2 = predicted_x + predicted_w 
    #predicted boundingbox height
    predicted_y2 = predicted_y + predicted_h 
    
    actual_x, actual_y, actual_x2, actual_y2 = float(annotation[3]), float(annotation[4]), float(annotation[5]), float(annotation[6])
    #actual boundingbox weight & height
    actual_weight, actual_height = (actual_x2 - actual_x, actual_y2 - actual_y) 
    ################# area overlap #############
    #top left point of overlap area
    (x1, y1) = (max(predicted_x, actual_x), max(predicted_y, actual_y))  
    #bottom right point of overlap area
    (x2, y2) = (min(predicted_x2, actual_x2), min(predicted_y2, actual_y2)) 
    
    overlap_weight = x2 - x1
    overlap_height = y2 - y1
    if overlap_weight > 0 and overlap_height > 0:
        overlap_area = overlap_weight*overlap_height
    else:
        overlap_area = 0
    
    ################ area union ##############
    #image area
    img_width, img_height = float(annotation[1]), float(annotation[2])
    area_img = img_width * img_height
    
    #area predicted
    area_pred = predicted_w*predicted_h
    
    #area actual
    area_actual = actual_weight*actual_height
    
    #area union
    area_union = area_pred + area_actual - overlap_area
    
    area_TP = overlap_area
    area_TN = area_img - area_union
    area_FP = area_union - area_actual
    area_FN = area_union - area_pred
    
    ############## IoU = area overlap / area union #########
    IoU = overlap_area / area_union
    IoU = round(IoU, 2)
    ############## pixel accuracy ###############
    pixel_acc = (overlap_area + area_TN) / (overlap_area + area_TN + area_FP + area_FN) # FP and FN = area_union - overlap_area
    pixel_acc = round(pixel_acc, 2)
    
    return IoU, pixel_acc
   
def hsv_segmentation(hsv_img):
    """this function will perform HSV segmentation and return largest contour"""
    # get mask
    mask1 = cv.inRange(hsv_img, lower1, upper1)
    mask2 = cv.inRange(hsv_img, lower2, upper2)
    mask3 = cv.inRange(hsv_img, lower3, upper3)
    # combine 3 mask 
    combined_mask = cv.add(cv.add(mask1, mask2), mask3)
    # opening
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel, iterations=3)
    # get biggest contour
    cnt1 = largestContour(combined_mask) 
    
    return cnt1

def canny_detection(img):
    """this function will perform Canny edge detection and return largest contour"""
    edges = cv.Canny(img, lower, upper)
    cnt2 = largestContour(edges)
    return cnt2

def displayResult(img_ori, xywh_pred, pred_label, prob):
    #predicted bounding box point
    leftX = int(xywh_pred[0])
    topY = int(xywh_pred[1])
    rightX = int(leftX + xywh_pred[2])
    bottomY = int(topY + xywh_pred[3])

    top_left_point = (leftX, topY) 
    bottom_right_point = (rightX, bottomY)

    #background text
    bg_top_left = (leftX, topY -13)
    bg_bottom_right = (rightX, topY)
    cv.rectangle(img_ori, bg_top_left, bg_bottom_right, (0,0,255), -2)

    #text
    text = str(pred_label) + " (" + str(prob) + ")"
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255,255,255)
    thickness = 1
    cv.putText(img_ori, text, top_left_point, font, fontScale, color, thickness, cv.LINE_AA, False)


######### model function ############
# transformer
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict(img, model): #img in PIL
    """this function will perform classification for input image"""
    #preprocess the image
    transformed_img = val_transform(img) #shape:(3,224,224)

    #add a batch dimention
    transformed_img = transformed_img.unsqueeze(0) #shape:(1,3,224,224)

    #get available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #pass to device 
    transformed_img = transformed_img.to(device)
    model = model.to(device)

    #perform inference with the image
    yhat = model(transformed_img)
    pred_class = yhat.argmax().item()
    prob = "{:.2f}".format(yhat[0][yhat.argmax().item()].item())
    return pred_class, prob


######### segmentation and recognition function #########
def segmentation_recognition(img_path, result_path, annotations=None):
    IoU = 0.
    pixel_acc = 0.
    
    img = cv.imread(img_path)
    img_ori = img.copy()
    
    ori_height = img.shape[0]
    ori_width = img.shape[1]

    #image name with extension
    imgName = os.path.basename(img_path)
    
    
    start_time = time.time()
    #image preprocessing
    img_resized = cv.resize(img, (200,200))
    hsv_img = cv.cvtColor(img_resized, cv.COLOR_BGR2HSV)

    ###################### HSV ##################
    cnt1 = hsv_segmentation(hsv_img)

    #################### Canny Edge Detection ###########
    cnt2 = canny_detection(img_resized)

    ################### bounding box selection ##########
    trueCnt = trueBBox(cnt1, cnt2)

    ################### get original size image of bounding box ##############
    xywh_pred = getOriSizeBBox(trueCnt, ori_height, ori_width)
    
    stop_time = time.time()
    print('Segmentation speed: {:.3f}ms'.format((stop_time - start_time) * 1000))
    if annotations is not None:
        ################## calculate IoU and pixel accuracy for original size image #########
        #get actual bounding box and annotation
        xywh_actual, annotation = getGroundtruth(imgName, annotations)
        #calculate IoU, pixel_acc
        IoU, pixel_acc = accuracy(xywh_pred, annotation)
        
    ##################### recognition #################
    start_time = time.time()
    imgRGB = cv.cvtColor(img_ori, cv.COLOR_BGR2RGB)
    imgRGB_PIL = Image.fromarray(imgRGB) #convert to PIL
    pred_label, prob = predict(imgRGB_PIL, model) #classification
    stop_time = time.time()
    print('Recognition speed: {:.3f}ms'.format((stop_time - start_time) * 1000))
    
    ################### draw bounding box ############
    drawBbox(img_ori, xywh_pred, (0,0,255), thickness=2, copy=False) #red = predicted
    if annotations is not None:
        drawBbox(img_ori, xywh_actual, (0,255,0), thickness=2, copy=False) #green = groundtruth

    ################## display classification result (label, probability) ##########
    displayResult(img_ori, xywh_pred, pred_label, prob)
    
    #image name without extension
    img_name = os.path.splitext(imgName)[0]
    if annotations is not None:
        cv.imwrite(os.path.join(result_path, img_name+"_IoU-"+str(IoU)+"_PixelAcc-"+str(pixel_acc)+"_PredictedClass-"+str(pred_label)+".png"), img_ori)
    else:
        cv.imwrite(os.path.join(result_path, img_name+"_PredictedClass-"+str(pred_label)+".png"), img_ori)
        
    return IoU, pixel_acc

def seg_rec_folder(folder_path, result_path, annotations=None):
    all_accuracy = []
    all_Iou = []
    #load all image in the folder
    images = os.listdir(folder_path)
    
    for image in images:
        #get image full path
        img_path = os.path.join(folder_path, image)
        
        ############ segmentation and recognition ###############
        IoU, pixel_acc = segmentation_recognition(img_path, result_path, annotations)
        
        #store iou and pixel accuracy
        all_accuracy.append(pixel_acc)
        all_Iou.append(IoU)
        
    print("Successfully display", len(images), "result to", result_path)
    if annotations is not None:
        #calculate average pixel accuracy and average IoU
        avg_acc = sum(all_accuracy)/len(all_accuracy)
        avg_IoU = sum(all_Iou)/len(all_Iou)
        print("Average accuracy:", avg_acc)
        print("Average IoU:", avg_IoU)

def clearFolder(result_path):
    #remove old result image
    if len(os.listdir(result_path)) != 0:
        remove_images = os.listdir(result_path)
        for imageName in remove_images:
            os.remove(os.path.join(result_path, imageName))
        print("removed", len(remove_images), "old result image in ", result_path)

########### main function ############
if __name__ == '__main__':
    #current path
    current_dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--folder_path',
        default="image_without_annotation",
        help='images folder to be segmented and recognized')
    parser.add_argument(
        '-r',
        '--result_folder_path',
        default=os.path.join(current_dir, "result"),
        help='folder to be segmented and recognized')
    parser.add_argument(
        '-a',
        '--annotation',
        default=False,
        type=bool,
        help='whether the test images is come with annotation') #if with annotation, then only can show groundtruth and accuracy/IoU
    parser.add_argument(
        '-t',
        '--train',
        default=True,
        type=bool, 
        help='the test images with annotation is from train set') #the images with annotation is from train set or test set
    args = parser.parse_args()

    """Setup"""

    #HSV range (need to change the range below to the best range we have found)
    lower1 = np.array([0,100,0])
    upper1 = np.array([180, 255, 255])
    lower2 = np.array([94, 130, 0])
    upper2 = np.array([124, 255, 255])
    lower3 = np.array([0, 210, 0])
    upper3 = np.array([180, 255, 255])

    #Canny edge range (need to change the range below to the best range we have found)
    lower = 31
    upper = 75

    #load model
    model = torch.load("saved_model.pt") 

    #output result path
    result_path = args.result_folder_path

    #source image folder path
    folder_path = os.path.join(current_dir, args.folder_path)
    #create folder if not yet created
    if os.path.exists(result_path) == False:
        os.mkdir(result_path)
        print("created folder => ", result_path)
    clearFolder(result_path)
    
    withAnnotation = args.annotation
    isTrainSet = args.train
    #if the test image is come with annotation
    if withAnnotation == True:
        #the test image is from train dataset
        if isTrainSet == True:
            annotation_path = os.path.join(current_dir, "TsignRecgTrain4170Annotation.txt")
        #if the test image is from test dataset
        else:
            annotation_path = os.path.join(current_dir, "TsignRecgTest1994Annotation.txt")
        #read annotation for train dataset
        annotations = readGroundtruths(annotation_path)
    
    

    if args.annotation == False:
        seg_rec_folder(folder_path, result_path)
    else:
        seg_rec_folder(folder_path, result_path, annotations)

















