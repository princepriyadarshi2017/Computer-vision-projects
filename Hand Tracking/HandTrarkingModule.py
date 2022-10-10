import time
import mediapipe as mp
import cv2


class handDetector():
    def __init__(self,mode=False, maxHands = 2, detectionCon =0.5, trackCon = 0.5):
        self.mode = mode
        self.maxhands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        hands = self.mpHands.Hands(self.mode,self.maxhands,self.detectionCon,
                                    self.trackCon)
        mpDraw = mp.solutions.drawing_utils

cap =cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime=0 #pervious time
cTime=0 #current time


while True:
    success, img =cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #we need to convert img into RGB beacuse hand class only uses RGB img
    results=hands.process(imgRGB) #now we are process that img

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                #print(id,lm) #this is hand landmarks
                h, w, c =img.shape #h=height of img w=widhth c= chnanel
                cx, cy = int(lm.x*w), int(lm.y*h) #for finding postions
                print(id,cx,cy) #this is id of node at x and y 
                #if id==0:
                cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED) #we are tracking by drawing circle

            mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS) #if we just use 
                                                                            #handLms it will only show dot but if we also 
                                                                            # use mpHands.HAND_CONNECTIONS then it show connection between dots

    cTime= time.time()
    fps = 1/(cTime - pTime)

    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,0,255),3)


    cv2.imshow("Image",img)
    cv2.waitKey(1)










