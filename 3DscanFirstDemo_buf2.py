import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from scipy import signal as scisig
import math
import numpy as np
import glob

def main():

    #======================
    for i in range(1,19):
        X = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        Y = decimateTo(X, i)
        print(' '.join(["Now decimiting to:"+ str(i)]))
        print(Y)
    print(Y)


    input("STOP")
    #======================
    
    filteredPics = []
    
    xAxis = np.array([[]])
    yAxis = np.array([[]])
    zAxis = np.array([[]])
       
    tol = 0.3
    cX = 300
    cY = 300

    path = "/Users/Olli/Desktop/laserTestPics/pngs/*.png"
    imgCount = len(glob.glob(path))
    
    
    print("Handling files:")
    print("["+imgCount*"#"+"]")
    print("[", end="",flush=True)

    # ------------------ SCANNING IMAGES ------------------
    count =0
    lim =5
    for fname in glob.glob(path):
        #if count == lim: break
        print('#',end="",flush=True)
        filteredPics.append(scanPics(fname, cX, cY, tol))
        count = count +1

    print("]")
    print("DONE!")

    yolo = 0
    for i in range(len(filteredPics)):
        yolo = yolo + len(filteredPics[i][0])

    av = yolo/len(filteredPics)
    print(av)
    print(yolo)
    print(len(filteredPics))
  

    print("Rotating Nodes:")

    # ------------------ ROTATING NODES ------------------
    rotatedNodes = rotateRoundAxis(filteredPics, imgCount)

    
    
    print(len(rotatedNodes))
    for nd in range(len(rotatedNodes)):
        xAxS = np.array([rotatedNodes[nd][0]])
        yAxS = np.array(rotatedNodes[nd][1])
        zAxS = np.array(rotatedNodes[nd][2])
        np.concatenate
        
        
        xAxis = np.concatenate((xAxis, xAxS))
        yAxis = np.concatenate((yAxis, yAxS))
        zAxis = np.concatenate((zAxis, zAxS))

        print(xAxis.shape)
        print(yAxis.shape)
        print("-----")
        
 #       ax2.plot_surface(zAxS,xAxS,yAxS, label="test")

    print("hoy!")
    print(xAxis.shape)
    
    # --------------- DECIMATING NODES --------------
    print(', '.join([str(len(xAxis)), str(len(yAxis)), str(len(zAxis))]),flush=True)
    sortNodes = list(zip(xAxis, yAxis, zAxis))
    
    #sortNodes = sorted(sortNodes, key = lambda n:n[1])
    

##    for i in range(len(sortNodes)):
##        print (sortNodes[i])

    delStep = 20
    dCount = int(len(sortNodes)/delStep - 0.5)
    for t in range(dCount-1, -1, -1):
        for t2 in range(delStep -1):
            del sortNodes[delStep * t]

    xAxis, yAxis, zAxis = zip(*sortNodes)


    print('Preparing plotting:',flush=True)
    print(', '.join([str(len(xAxis)), str(len(yAxis)), str(len(zAxis))]),flush=True)


    fig = plt.figure()
    ax2 = Axes3D(fig)

    #ax2.plot3D(zAxis,xAxis, yAxis)
    ax2.scatter(zAxis,xAxis, yAxis, color='red')
    
    xAxis, yAxis = np.meshgrid(xAxis, yAxis)
    print(xAxis.shape)
    print(yAxis.shape)
    print(len(zAxis))
    ax2.plot_wireframe(zAxis,xAxis,yAxis,rstride=100, cstride=50, label="test")
    
    
    

    ax2.set_xlim3d(-200, 200)
    ax2.set_ylim3d(-200,200)
    ax2.set_zlim3d(450,200)

    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    #ax2.w_zaxis.set_major_locator(LinearLocator(6))
    
    plt.show()




def scanPics(path, cntrX, cntrY, tolerance):
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

    # Scan the image for red pixels (i.e. the outline)
    topJErr = 0
    topIErr = 0
    for i in range(len(img)):
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

##   print(' '.join(["top I error:", str(topIErr), "| top J error:", str(topJErr)]))
##   Set the values to be used with the plotting tools
##   print("pixel matching the criterian found: " + str(len(filteredNodes)))
                     
    zAx, yAx = zip(*filteredNodes)
    
    '''Generate X-Axis '''
    xAx = np.zeros(len(zAx))

    '''Scale Y-Axis    '''
    yAx_scld =tuple(y/math.cos(math.pi/4) for y in yAx)
    
    '''Scale Z-Axis    '''
    zAx_scld =tuple(z-centreMark for z in zAx)

    scldNodes = np.array([xAx, yAx_scld, zAx_scld])
    return scldNodes


def rotateRoundAxis(nodeSets, steps):

    transformedNodes = []
    rvrs = True
    for r in range(steps):
        xAx, yAx, zAx = nodeSets[r]
        distArr = np.sqrt(np.square(xAx) + np.square(zAx))
        rotTransArr = np.array([distArr * math.sin(2*math.pi/steps * r),yAx, distArr * math.cos(2*math.pi/steps * r)])
##        if rvrs:
##            rotTransArr = np.array([rotTransArr[0][::-1], rotTransArr[1][::-1], rotTransArr[2][::-1]])
##            rvrs = False
##        else:
##            rvrs = True
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

