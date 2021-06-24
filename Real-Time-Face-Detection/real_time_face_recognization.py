import cv2
print(cv2.__version__)
import sys
imgpath=sys.argv[1]
facecascade=cv2.CascadeClassifier(imgpath)
video_capture=cv2.VideoCapture(0)

while True:
    ret,frame=video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facecascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow('FACE DETECTOR',frame)


    if cv2.waitKey(1) == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()