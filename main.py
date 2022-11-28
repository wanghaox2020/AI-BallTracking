import DecomposeVideo
import GenerateVideo
import ObjectDetector
import SingleObjectTracking


# initial setting up for generate folder for the plays

def FolderGenerate():
    DecomposeVideo.CreateSingleBallRawFolder()
    DecomposeVideo.CreateSingleBallResultFolder()

    DecomposeVideo.CreateMultipleBallRawFolder()
    DecomposeVideo.CreateMultipleBallResultFolder()


def Decompose():
    DecomposeVideo.DecomposeSingleBall()
    DecomposeVideo.DecomposeMultiBall()


# Task 2 function, decompose the video and find the bounding box on each frame of the picture
# and then generate the video from the result, including two video, which names is SingleBallDetect.avi and
# MultiBallDetect.avi


def task2():
    ObjectDetector.SingleObjectDetector()
    GenerateVideo.SingleBallVideo()

    ObjectDetector.MultipleObjectDetector()
    GenerateVideo.MultipleBallVideo()


# Task 3 function, used to use kalmen filter to produce the ceter and the prediction of the object

def task3():
    SingleObjectTracking.SingleObjectTracking()


def main():
    FolderGenerate()
    Decompose()
    # task2()
    task3()


if __name__ == "__main__":
    main()
