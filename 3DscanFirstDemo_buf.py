import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
import math
import numpy as np
import glob

def main():
    filteredPics = []
    
    xAxis = np.array([])
    yAxis = np.array([])
    zAxis = np.array([])
       
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

    print("Rotating Nodes:")
    print(len(filteredPics[0][0]))

    # ------------------ ROTATING NODES ------------------
    rotatedNodes = rotateRoundAxis(filteredPics, imgCount)

    
    
    print(len(rotatedNodes))
    for nd in range(len(rotatedNodes)):
        
        xAxS, yAxS, zAxS = rotatedNodes[nd]
        xAxis = np.append(xAxis, xAxS)
        yAxis = np.append(yAxis, yAxS)
        zAxis = np.append(zAxis, zAxS)

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
    
##    filtCoords = []
##    for i in range(3):
##        step = 10
##        temp = finalNodes[i]
##        sCount = int(len(finalNodes[i])/step - 0.5)
##        for t in range(sCount-1,-1,-1):
##            for t2 in range(step-1):
##                temp = np.delete(temp,step * t)
##        filtCoords.append(temp)


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
        if rvrs:
            rotTransArr = np.array([rotTransArr[0][::-1], rotTransArr[1][::-1], rotTransArr[2][::-1]])
            rvrs = False
        else:
            rvrs = True
        transformedNodes.append(rotTransArr)

    return transformedNodes


if __name__ == "__main__":
    main()

