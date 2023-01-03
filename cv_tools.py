import cv2
import numpy as np
def detect_edges(img): return cv2.Canny(img, 100, 200)
if __name__ == '__main__': print('CV Tools Ready')