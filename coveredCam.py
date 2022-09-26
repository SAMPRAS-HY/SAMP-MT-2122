import cv2

vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_16S)
    mean, stddev = cv2.meanStdDev(lap)
    print(stddev)

    if stddev < 4:
        cv2.putText(frame,"Covered!!!",(50,150), cv2.FONT_HERSHEY_SIMPLEX, 2,(136,8,8),4)

    cv2.imshow('frame', frame)

vid.release()
cv2.destroyAllWindows()