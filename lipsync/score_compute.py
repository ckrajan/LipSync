import cv2 as cv
import math
import numpy as np
import re
import ast


def ScoreMouth():
    finalscore_add = 0
    counter = 0
    with open('/home/chathushkavi/projects/lipsync/lipsync/source.json', 'r') as in_file:
        data = in_file.read()
    source_res = ast.literal_eval(data)

    with open('/home/chathushkavi/projects/lipsync/lipsync/target.json', 'r') as in_file:
        data = in_file.read()
    target_res = ast.literal_eval(data)

    for i in range(0,len(source_res)):
        if(source_res[i][0] == target_res[i][0]):
            baseMouthCoord = source_res[i][1]
            predMouthCoord = target_res[i][1]
            # predMouthCoord = [[536.9680786132812, 393.8536376953125],[543.4403686523438, 392.0062561035156],[551.5603637695312, 391.424560546875],[560.7589111328125, 390.8169860839844],[570.7390747070312, 389.66326904296875],[580.8385009765625, 388.4952087402344],[587.8954467773438, 387.36077880859375],[592.9296875, 386.2567443847656],[596.129638671875, 385.4277038574219],[598.2338256835938, 384.7377014160156],[600.5786743164062, 385.60113525390625],[598.4376831054688, 388.181640625],[596.8475952148438, 389.5059814453125],[593.4703369140625, 390.7751159667969],[588.3738403320312, 392.3139953613281],[581.1087036132812, 393.646240234375],[571.0308227539062, 394.8466796875],[560.9015502929688, 395.2613525390625],[551.0765991210938, 395.6007385253906],[544.3613891601562, 395.2867736816406]]
            # baseMouthCoord = [[208.9613037109375,374.5960998535156],[214.79347229003906,373.6083679199219],[221.36044311523438,374.5328063964844],[229.1942901611328,375.75616455078125],[238.94931030273438,377.16790771484375],[250.2723388671875,377.9490661621094],[262.5882568359375,376.80609130859375],[273.5610656738281,375.12908935546875],[282.87139892578125,373.62762451171875],[290.90692138671875,372.4633483886719],[297.68865966796875,373.29888916015625],[290.4766540527344,376.79876708984375],[284.1789855957031,378.7607727050781],[274.47662353515625,381.11895751953125],[263.1787414550781,383.18804931640625],[250.48892211914062,384.22076416015625],[238.55491638183594,383.4581604003906],[228.40078735351562,381.7041931152344],[220.17918395996094,379.6187744140625],[215.1010284423828,377.8349304199219]]
            baseMouthCoord_list = ast.literal_eval(baseMouthCoord)
            predMouthCoord_list = ast.literal_eval(predMouthCoord)

            predBin = CreateBinImage(predMouthCoord_list)
            baseBin = CreateBinImage(baseMouthCoord_list)

            shapeMatchDistant = cv.matchShapes(predBin, baseBin, cv.CONTOURS_MATCH_I2, 0)
            finalScore = shapeMatchDistant
            # print("finalScore  :  ", finalScore)
            matchScore = finalScore
            dimensionsScore = max(0, (1.0 - matchScore))
            dimensionsScore = 1 / (1 + math.exp(-15 * (2 * dimensionsScore - 1.3)))
            currentScore = dimensionsScore * 1000
            finalscore_add = finalscore_add + currentScore
            counter = counter + 1

            print("current score is  :  ", currentScore)

    return finalscore_add, counter

def CreateBinImage(mouthPoints):
    compareCanvasWidth = 100
    leftpoint = mouthPoints[0]
    toppoint = mouthPoints[5]
    rightpoint = mouthPoints[10]
    bottompoint = mouthPoints[15]
    scaleratio = compareCanvasWidth/(rightpoint[0]-leftpoint[0])
    
    # print(mouthPoints,  "A")
    minX = min(map(lambda elt: elt[0] ,  mouthPoints))
    minY = min(map(lambda elt: elt[1] ,  mouthPoints))

    # // move points to zero, to int
    for num, elt in enumerate(mouthPoints):
        newX = int((elt[0] - minX) * 5)
        newY = int((elt[1] - minY) * 5)
        mouthPoints[num] = [newX, newY]

    # // create background image
    imgWidth = max(map(lambda elt: elt[0] ,  mouthPoints))
    imgHigh = max(map(lambda elt: elt[1] ,  mouthPoints))

    size = imgHigh+1, imgWidth+1
    img = np.zeros(size, dtype=np.uint8)

    # // array to Mat
    flatten_mouthPoints = np.array(mouthPoints).flatten()
    mat_mouthPoints =flatten_mouthPoints.reshape(20, 2) 
    cv.fillPoly(img, pts=[mat_mouthPoints], color=(255, 0, 0))
    
    return img

def total_score():
    finalscore_add, counter = ScoreMouth()
    finalavgScore = finalscore_add / counter
    print("Score out of 1000: ", finalavgScore)
    print("counter: ", counter)


if __name__ == '__main__':
    finalavgScore = 0
    total_score()