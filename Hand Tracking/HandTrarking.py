import time
import mediapipe as mp
import cv2

cap =cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils



while True:
    success, img =cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #we need to convert img into RGB beacuse hand class only uses RGB img
    results=hands.process(imgRGB) #now we are process that img

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms)




    cv2.imshow("Image",img)
    cv2.waitKey(1)