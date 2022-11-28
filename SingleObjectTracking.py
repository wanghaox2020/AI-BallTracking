from ObjectTracking import ObjectTracking
from KalmanFilter import KalmanFilter
from PIL import Image
import numpy as np
import cv2


def SingleObjectTracking():
    dt = 1.0 / 60
    # initalize the kalman_filter with the required parameter
    kalman_filter = KalmanFilter(dt, 0, 0, 1, 0.1, 0.1 )
    ini_x, ini_y = kalman_filter.predict()
    ini_draft = cv2.imread('SingleBallRawFolder/SingleBall0.jpg')
    cv2.circle(ini_draft, (ini_x, ini_y), 10, (0, 255, 0), 2)

    for i in range(51):
        img = Image.open('SingleBallRawFolder/SingleBall%d.jpg' % i)
        draft = cv2.imread('SingleBallRawFolder/SingleBall%d.jpg' % i)
        # find the signel object
        center, score = ObjectTracking(img)
        if len(center) > 0:
            index = score.index(max(score))
            update_x, update_y = kalman_filter.update(center[index])
            cv2.circle(draft, (update_x, update_y), 10, (0, 255, 0), 2)

    cv2.imwrite('SingleBallResultFold/SingleBallTracking%d.jpg'% i, draft)