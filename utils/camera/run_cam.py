import cv2
# import time

from Camera import Camera

learning = False # TODO: Improve
def run_cam(pipe):
    cap = cv2.VideoCapture(0)
    cam = Camera()

    while cap.isOpened():

        global learning
        if pipe.poll():
            cmd = pipe.recv()
            if cmd == 'Learning': 
                learning = True
            elif cmd == 'Finished learning':
                learning = False

        if not learning:
            success, frame = cap.read()
            if success:
                # cv2.imshow('Frame', frame)
                # k = cv2.waitKey(20)
                # print("SUCCESS")
                pipe.send(cam.check_obstacle(frame)) # sends bool
            # else:
            #     print("FAILURE!")

    cap.release()
    cv2.destroyAllWindows()