"""Simulates the camera module."""
import cv2
import os
import numpy as np
import glob
import time
import matplotlib.pyplot as plt

class Camera:

    def __init__(self) -> None:
        self.reward = 0
        self.good_forward_move = True
        
        # contrast adjustment factor
        self.alpha = 8
        # brightness adjustment factor
        self.beta = 2
        self.num = 0
        
        
        
    def _take_picture_(self):   # deprecated
        image_path = f"./img/img{self.num-2}.png"
        if os.path.exists(image_path):
            os.remove(image_path)
        
        cap = cv2.VideoCapture(0)

        success, img = cap.read()
            
        #img = cv2.convertScaleAbs(img, alpha=self.alpha, beta=self.beta)
        
        if success:
            cv2.imwrite(f"./img/img{self.num}.png",img)
        else:
            return False
        self.num += 1

        #cv2.imshow('Img',img)
        #cv2.waitKey(0)

        # Release and destroy all windows before termination
        cap.release()
        cv2.destroyAllWindows()
        
        return True
    
    # def check_obstacle(self,cam_info) -> bool:
    def check_obstacle(self, frame) -> bool:
        """ Returns if True if obstacle detected. """
        # Define the coordinates of the center 100x100 window
        center_x = 260  
        center_y = 150  
        window_size = 200

        center_window = frame[center_y:center_y+window_size, center_x:center_x+window_size]

        blurred = cv2.GaussianBlur(center_window, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)  
        binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)[1]

        # cv2.imshow('Canny output', binary_edges)
        # k = cv2.waitKey(20)
        
        if cv2.countNonZero(binary_edges) > 0:
            print("OBSTACLE DETECTED")
            return True
        return False
