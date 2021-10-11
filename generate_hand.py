import cv2

frame_length = 900
sr = 44100

cap=cv2.VideoCapture(0)
i=0

while(1):
    ret ,frame = cap.read()
    frame=cv2.flip(frame,1)
    color = (0, 255, 255)
    cv2.rectangle(frame, (190, 210), (449, 469), color, 2)
    k=cv2.waitKey(1)
    if k==27:
        break
    elif k in [ord('a'),ord('b'),ord('c'),ord('d'),ord('e'),ord('f'),ord('g'),ord('h'),ord('i'),ord('j'), ord('k'), ord('l')]:
        print("s!")
        cv2.imwrite(f'D:/ML/final/data/{chr(k)}/'+'n1_'+str(i)+'.jpg',frame[212:468,192:448])
        i+=1
        cv2.imshow("i", frame[212:468,192:448])
    cv2.imshow("capture", frame)
cap.release()
cv2.destroyAllWindows()