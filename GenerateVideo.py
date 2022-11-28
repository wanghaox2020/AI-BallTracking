import cv2


def SingleBallVideo():
    frameSize = (960, 540)
    out = cv2.VideoWriter('SingleBallDetect.avi', cv2.VideoWriter_fourcc(*'DIVX'), 12, frameSize)

    for i in range(51):
        img = cv2.imread('SingleBallResultFolder/SingleBallResult%d.jpg' % i)
        out.write(img)

    out.release()


def MultipleBallVideo():
    frameSize = (640, 360)
    out = cv2.VideoWriter('MultipleBallDetect.avi', cv2.VideoWriter_fourcc(*'DIVX'), 12, frameSize)

    for i in range(41):
        img = cv2.imread('MultipleBallResultFolder/MultiBallResult%d.jpg' % i)
        out.write(img)

    out.release()