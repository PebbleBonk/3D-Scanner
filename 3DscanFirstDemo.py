import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator

import math
import numpy as np
import glob

def onclick(event):
    if event.xdata != None and event.ydata != None:
        cX = int(np.asscalar(event.xdata))
        lwb = int(np.asscalar(event.ydata))
        global coords
        coords = []
        coords.append((cX,lwb))
        
        if len(coords) == 1:
 #           fig.canvas.mpl_disconnect(cid)
            plt.close(1)
        print(' '.join(["Rot Axis:",str(cX),"Lower bound:",str(lwb)]))
        return coords

def main():

    
    filteredPics = []
       
    tol = 0.2
    cX = 250
    cY = 300

    path = "/Users/Olli/Desktop/laserTestPics/pngs/*.png"
    imgCount = len(glob.glob(path))
    
    
    print("Handling files:")
    print("["+imgCount*"#"+"]")
    

    # ------------------ SCANNING IMAGES ------------------
    count =0
    lim =5
    
    for fname in glob.glob(path):
        
        print('#',end="",flush=True)
        if count == 0:
##            lum_img = mpimg.imread(fname)[:,:,0]
##
####            ax = plt.gca()
####            fig = plt.gcf()
####            implot = ax.imshow(lum_img)
####
####            cid = fig.canvas.mpl_connect('button_press_event', onclick)
####
####
####            plt.show()

            cX = 300#int(input("Give the center of rotation (X.coordinate) as integer: "))
            lwb = 370#int(input("Give the level to filter off the image (Y-Coordinate): "))
            print(cX, lwb)

            filteredPics.append(scanPics(fname, cX, cY, tol, lwb))
            
            print("[", end="",flush=True)
        else:
            filteredPics.append(scanPics(fname, cX, cY, tol, lwb))
        count = count +1

    print("]")
    print("DONE!")

    nodeCount = 0
    minNodeCount = 10000
    for i in range(len(filteredPics)):
        nCount = len(filteredPics[i][0])
        nodeCount = nodeCount + nCount
        if nCount< minNodeCount:
            minNodeCount = nCount
            
    nodeCountAvrg = nodeCount/len(filteredPics)
    decimLen = min(int(nodeCountAvrg*0.6), 50)
    print(' '.join(["Minimum of datapoints detected in photo:", str(minNodeCount)]))
    print(' '.join(["Averge datapoints detected in photos:",str(nodeCountAvrg)]))
    #input("STOP")

    print("Rotating Nodes:")
    

##    ind = 0
##    for fname in glob.glob(path):
##        plt.close(1)
##        f = plt.figure()
##        axI1 = Axes3D(f)
##        axI2 = f.add_subplot(111)
##        
##        imgI =  mpimg.imread(fname)
##        
##        axI2.imshow(imgI)
##        
##        xI,yI,zI = filteredPics[ind]
##        Axes3D(f)
##        axI1.scatter(xI,yI,zI)
##        
##        ind = ind+1
##        plt.show()
##        input("")
        
    
        

    # ------------------ ROTATING NODES ------------------
    rotatedNodes = rotateRoundAxis(filteredPics, imgCount)

    
    
    firstOne = None
    for nd in range(len(rotatedNodes)):
        xAxS = np.array(rotatedNodes[nd][0])
        yAxS = np.array(rotatedNodes[nd][1])
        zAxS = np.array(rotatedNodes[nd][2])


        if len(xAxS) < decimLen: continue
        
        # Decimite the arrays to the length of 
        # the smallest array to concetenate.
        xAxS = np.array([decimateTo(xAxS, decimLen)])
        yAxS = np.array([decimateTo(yAxS, decimLen)])
        zAxS = np.array([decimateTo(zAxS, decimLen)])

        if firstOne == None:
            firstOne = [xAxS, yAxS, zAxS]
            xAxis = xAxS
            yAxis = yAxS
            zAxis = zAxS

        xAxis = np.concatenate((xAxis, xAxS))
        yAxis = np.concatenate((yAxis, yAxS))
        zAxis = np.concatenate((zAxis, zAxS))

    xAxis = np.concatenate((xAxis, firstOne[0]))
    yAxis = np.concatenate((yAxis, firstOne[1]))
    zAxis = np.concatenate((zAxis, firstOne[2]))

    gridAxisX =  xAxis[:, len(xAxis[0]) - 1:]
    gridAxisY =  yAxis[:, len(yAxis[0]) - 1:]
    gridAxisZ =  zAxis[:, len(zAxis[0]) - 1:]

    gaX = gridAxisX
    gaY = gridAxisY
    gaZ = gridAxisZ

    for i in range(len(gridAxisY)):
        gaX[i] = gridAxisX[(i+ (int(len(gaX)/2)))%len(gaX)]
        gaY[i] = gridAxisY[(i+ (int(len(gaY)/2)))%len(gaY)]
        gaZ[i] = gridAxisZ[(i+ (int(len(gaZ)/2)))%len(gaZ)]


    xAxis = np.concatenate((xAxis, gaX), axis=1)
    yAxis = np.concatenate((yAxis, gaY), axis=1)
    zAxis = np.concatenate((zAxis, gaZ), axis=1)


    print('Preparing plotting:',flush=True)
    print(', '.join([str(len(xAxis)), str(len(yAxis)), str(len(zAxis))]),flush=True)

    fig = plt.figure()
    ax2 = Axes3D(fig)

    #ax2.scatter(zAxis,xAxis, yAxis, color='red')
    ax2.plot_surface(zAxis,xAxis,yAxis,rstride=1, cstride=20,  label="test")
    
    print(xAxis.shape)
    
    ax2.set_xlim3d(-200, 200)
    ax2.set_ylim3d(-200,200)
    ax2.set_zlim3d(400,200)

    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    
    plt.show()




def scanPics(path, cntrX, cntrY, tolerance, lwb = -1):
    '''
    Scans the photo on the given path and returns the coordinates
    of scaled points that match the criteria.
    @param; path (str): the path to the photo
    @param; cntrX (int): the estimated x-coord of object's centre
    @param; cntrY (int): the estimated y-coord of object's centre
    @param; tolerance (float): the tolerance of laser( i.e. red pixels)

    §Returns (numpy.array): An array containing the scaled coordinates
                            of scanned points that matched tolerance.
    
    '''
    img = mpimg.imread(path)
    lum_img = img[:,:,0]


    
    filteredNodes = []
    centreMark = cntrX    #As Int
    centreHor = cntrY     #As Int
    camTiltAbb = math.pi/8     #In Radians
    
    lowBound = lwb
    

    # Scan the image for red pixels (i.e. the outline)
    topJErr = 0
    topIErr = 0
    for i in range(len(img)):
        
        if i == lowBound and lowBound != -1: break
        for j in range(len(img[i])):
           
            if float(lum_img[i,j]) >= tolerance:
                
                # ============= Only testing filtering =================
                if len(filteredNodes) != 0:
                    jAv, iAv = zip(*filteredNodes)
                    if abs(np.average(jAv) - j ) > topJErr:
                        topJErr = abs(np.average(jAv) - j )
                    if abs(np.average(iAv) - i ) > topIErr:
                        topIErr = abs(np.average(iAv) - i )
                # ============= Only testing filtering =================
                
                if len(filteredNodes) == 0:
                    filteredNodes.append(tuple([j,i]))
                elif abs(filteredNodes[-1][0] - j ) < 5 or \
                     abs(np.average(iAv) - i ) < 50:
                     filteredNodes.append(tuple([j,i]))

    if len(filteredNodes) == 0:
        filteredNodes.append(tuple([0,0]))
    zAx, yAx = zip(*filteredNodes)
    
    '''Generate X-Axis '''
    xAx = np.zeros(len(zAx))

    '''Scale Y-Axis    '''
    yAx_scld =tuple(y/math.cos(math.pi/4) for y in yAx)
    
    '''Scale Z-Axis    '''
    zAx_scld =tuple(z-centreMark for z in zAx)
    #print(path + " " + str(len(xAx)))
    scldNodes = np.array([xAx, yAx_scld, zAx_scld])

    return scldNodes


def rotateRoundAxis(nodeSets, steps):

    transformedNodes = []
    rvrs = True
    for r in range(steps):
        xAx, yAx, zAx = nodeSets[r]
        distArr = np.sqrt(np.square(xAx) + np.square(zAx))
        rotTransArr = np.array([distArr * math.sin(2*math.pi/steps * r),yAx, distArr * math.cos(2*math.pi/steps * r)])

        transformedNodes.append(rotTransArr)

    return transformedNodes

def decimateTo(A, toSize):

    '''
    Takes an array as an parameter and removes
    elements from it, evenly (e.g. every third
    and so on) until the size that is given as
    a parameter is met. Then return the array.
    @param; A (np.array): an array to decimate
    @param; toSize (int): limit to decimate to
    §Returns (np.array): the new smaller array
    '''

    if "numpy" in str(type(A)):
        A= A.tolist()

    # Delete elements until array is
    # dividable with the toSize parm
    preToDel = len(A)%toSize

    if preToDel != 0:
        delDist = int(len(A)/preToDel-0.5)
        if delDist == len(A): delDist = int(len(A)/2)

        for i in range(1,preToDel+1):
            if len(A) <= i*delDist: break
            del A[i * delDist - 1]

    # then remove else in sequenses
    toRem = len(A) - toSize
    delStep = int(toRem/toSize)
    delCount = toSize
    
    for i in range(0,toRem):
        del A[i+1:(i+1+delStep)]

    for i in range(len(A)-toSize):
        del A[int(len(A)/2)]
    
    return A



    
if __name__ == "__main__":
    main()

