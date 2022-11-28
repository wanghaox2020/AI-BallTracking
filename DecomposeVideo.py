import cv2
import os


def CreateSingleBallRawFolder():
    try:
        # Create target Directory
        dirName1 = "SingleBallRawFolder"
        os.mkdir(dirName1)
        print("Directory ", dirName1, " Created ")
    except FileExistsError:
        print("Directory ", dirName1, " already exists")


def CreateSingleBallResultFolder():
    try:
        dirName3 = "SingleBallResultFolder"
        os.mkdir(dirName3)
        print("Directory ", dirName3, " Created ")
    except FileExistsError:
        print("Directory ", dirName3, " already exists")


def DecomposeSingleBall():
    vidCap = cv2.VideoCapture("ball.mp4")
    success, image = vidCap.read()
    count = 0
    while success:
        cv2.imwrite("SingleBallRawFolder/SingleBall%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidCap.read()
        count += 1

    print("ball.mp4 decompose successful")


def CreateMultipleBallRawFolder():
    try:
        dirName2 = "MultipleBallRawFolder"
        os.mkdir(dirName2)
        print("Directory ", dirName2, " Created ")
    except FileExistsError:
        print("Directory ", dirName2, " already exists")


def CreateMultipleBallResultFolder():
    try:
        dirName4 = "MultipleBallResultFolder"
        os.mkdir(dirName4)
        print("Directory ", dirName4, " Created ")
    except FileExistsError:
        print("Directory ", dirName4, " already exists")


def DecomposeMultiBall():
    vidCap = cv2.VideoCapture("multiBall.mp4")
    success, image = vidCap.read()
    count = 0
    while success:
        cv2.imwrite("MultipleBallRawFolder/MultiBall%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidCap.read()
        count += 1

    print("multiBall.mp4 decompose successful")
