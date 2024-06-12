import cv2
# import time
import os

from Camera import Camera

learning = False # TODO: Improve
def run_cam(pipe):
    cap = cv2.VideoCapture(0)
    cam = Camera()

    crawl_path = 'crawl_img.png'
    walk_path = 'walk_img.png'
    idle_path = 'idle_img.png'
    init_path = 'init_img.png'
    mid_path = 'mid_img.png'

    fnames = ['img_' + str(i) for i in range(10)]
    fname_ctr = 1


    while cap.isOpened():

        # global learning
        # if pipe.poll():
        #     cmd = pipe.recv()
        #     if cmd == 'Learning': 
        #         learning = True
        #     elif cmd == 'Finished learning':
        #         learning = False

        if not learning:
            success, frame = cap.read()
            if success:
                # cv2.imshow('Frame', frame)
                # k = cv2.waitKey(20)
                # print("SUCCESS")
                # pipe.send(cam.check_obstacle(frame)) # sends bool

                check_obs = False
                fname = ''
                save_img = False

                if pipe.poll():
                    cmd = pipe.recv()
                    # if cmd == 'Crawl' and not os.path.isfile(crawl_path):
                    #     fname = crawl_path
                    #     save_img = True
                    #     # cv2.imwrite(crawl_path, frame)
                    # elif cmd == 'Walk' and not os.path.isfile(walk_path):
                    #     fname = walk_path
                    #     save_img = True
                    #     # cv2.imwrite(walk_path, frame)
                    # elif cmd == 'Idle' and not os.path.isfile(idle_path):
                    #     fname = idle_path
                    #     save_img = True
                    #     # cv2.imwrite(idle_path, frame)
                    # elif cmd == 'Init' and not os.path.isfile(init_path):
                    #     fname = init_path
                    #     save_img = True
                    #     # cv2.imwrite(idle_path, frame)
                    # elif cmd == 'Mid' and not os.path.isfile(mid_path):
                    #     fname = mid_path
                    #     save_img = True
                    if cmd == 'save_img':
                        fname = fnames[fname_ctr]
                        save_img = True
                        fname_ctr += 1
                    elif cmd == 'done':
                        break

                pipe.send(cam.check_obstacle(frame, save_img, fname))
            

    cap.release()
    cv2.destroyAllWindows()