import cv2
import time
import os

wCam, hCam = 640, 480

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))

while True:
    success, img = cap.read()

    h, w, c = overlayList[1].shape
    img[0:h, 0:w] = overlayList[1]

    cv2.imshow("Image", img)
    cv2.waitKey(1)