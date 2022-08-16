import numpy as np
import cv2

def sharpness(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(img, cv2.CV_16S)
    mean, stddev = cv2.meanStdDev(lap)
    return stddev[0,0]

def main():
    #Video capturing starts
    image = cv2.VideoCapture(0)
    

    while(True):
        ret, frame = image.read()
        
        if sharpness(frame) < 12:
            cv2.putText(frame, "Covered!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 ,255, 255), 1, cv2.LINE_AA) 

        cv2.imshow('Frame', frame)
        #Close camera with "esc"        
        k = cv2.waitKey(2) & 0xff 
        if k == 27:
            break

    #releasing camera and closing all the windows
    image.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()