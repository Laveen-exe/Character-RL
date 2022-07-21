import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

reader = easyocr.Reader(['en', 'de'])

def preprocessing(path, save):
    ## Provide the path to the image
    IMAGE_PATH = path
    true_image = cv2.imread(IMAGE_PATH)
    image = text_detection_number_character(IMAGE_PATH)
    _, data = character_detection(IMAGE_PATH)
    # Filename
    filename = 'savedImage.png'
    cv2.imwrite(filename, image)

    top_left = 0
    bottom_right = 0
    i = 1
    for detection in data:
        top_left = (detection[0], detection[1])
        bottom_right = (detection[2], detection[3])
        crop_img = true_image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
        filename = save + str(i) + ".png"
    # filename = "croppedsavedImage.png"
        cv2.imwrite(filename, crop_img)
        i += 1
    return data

def character_detection(path):
    moment = [] # list to contain Hu moments of the all the characters
    data = []
    # Defining images
    true_image = cv2.imread(path)
    image = cv2.imread(path)
    img = cv2.imread(path,0)

    #Applying threshold for the EasyOCR
    _,img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    _,image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    result = reader.readtext(img)
    # box_size = 10
    for i in range(len(result)):

        # Coordinates of the detected box by EasyOCR
        top_left = result[i][0][0]
        bottom_right = result[i][0][2]
        top_left[0], top_left[1], bottom_right[0], bottom_right[1] = int(top_left[0]), int(top_left[1]), int(bottom_right[0]), int(bottom_right[1])

        # Cropping the image to word level
        im = img[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
        im_1 = image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
        # cv2_imshow(im)

        inputCopy = im_1.copy()
        inputImage = ~im_1
        # Convert BGR to grayscale:
        grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        # Set the adaptive thresholding (gasussian) parameters:
        windowSize = 31
        windowConstant = -1
        # Apply the threshold:
        binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, windowSize, windowConstant)

        # Perform an area filter on the binary blobs:
        componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(binaryImage, connectivity=4)

        # Set the minimum pixels for the area filter:
        minArea = 20

        # Get the indices/labels of the remaining components based on the area stat
        remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

        # Filter the labeled pixels based on the remaining labels,
        # assign pixel intensity to 255 (uint8) for the remaining pixels
        filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

        # Set kernel (structuring element) size:
        kernelSize = 3

        # Set operation iterations:
        opIterations = 1

        # Get the structuring element:
        maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

        # Perform closing:
        closingImage = cv2.morphologyEx(filteredImage, cv2.MORPH_CLOSE, maxKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

        # Get each bounding box
        # Find the big contours/blobs on the filtered image:
        contours, hierarchy = cv2.findContours(closingImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None] * len(contours)
        # The Bounding Rectangles will be stored here:
        boundRect = []

        # Alright, just look for the outer bounding boxes:
        for i, c in enumerate(contours):

            if hierarchy[0][i][3] == -1:
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                boundRect.append(cv2.boundingRect(contours_poly[i]))


        # Draw the bounding boxes on the (copied) input image:
        boundRect.sort()
        for i in range(len(boundRect)):
            color = (0, 255, 0)
            original_x_1 = boundRect[i][0] + top_left[0]
            original_y_1 = boundRect[i][1] + top_left[1]
            original_x_2 = boundRect[i][0] + boundRect[i][2] + top_left[0]
            original_y_2 = boundRect[i][1] + boundRect[i][3] + top_left[1]
            cv2.rectangle(true_image, (int(original_x_1), int(original_y_1)), \
                      (int(original_x_2), int(original_y_2)), color, 1)
            data_points = [original_x_1,original_y_1,original_x_2,original_y_2]
            data.append(data_points)
            cv2.rectangle(inputCopy, (int(boundRect[i][0]), int(boundRect[i][1])), \
                      (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
            moment_img = im[boundRect[i][1] : (boundRect[i][1] + boundRect[i][3]), boundRect[i][0] : (boundRect[i][0] + boundRect[i][2])]
            moments = cv2.moments(moment_img)
            huMoments = cv2.HuMoments(moments)
            huMoments = np.append(huMoments,[[0]])
            moment.append(huMoments)

    return true_image,data

def text_detection_number_character(IMAGE_PATH):
    img, data = character_detection(IMAGE_PATH)
    count = 0
    for detection in data:
        top_left = (detection[0],detection[1])
        bottom_right = (detection[2],detection[3])
        font = cv2.FONT_HERSHEY_SIMPLEX
        img  = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 1)
        img = cv2.putText(img, str(count), top_left, font, 0.2, (255, 0, 0),1, cv2.LINE_AA)
        count+=1
    plt.figure(figsize=(100,100))
    return img

if __name__ == '__main__':
    data = preprocessing(path = 'yy.png', save = "cropped_images/croppedsavedImage_")
    print(data)
    # print(cropped_img.shape)

