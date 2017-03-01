import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np


def scanPhotos():
    filteredNodes = []

    img = mpimg.imread('/Users/Olli/Desktop/laser_test0.png')

    fig1 = plt.figure()
    plt.imshow(img)
    #plt.show()

    #centreMark = input("Please enter the center of the picture (X-coordinate)\n:")
    #plt.imshow(img)
    centreMark = 290
    centreHor = 300
    camTiltAbb = 30 #In degrees

    plt.ylim([0,600])
    plt.xlim([0,600])
    plt.gca().invert_yaxis()
    lum_img = img[:,:,0]

    # Scan the image for red pixels (i.e. the outline)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if float(lum_img[i,j]) >= 0.3:
                filteredNodes.append(tuple([j,i]))


    # Set the values to be used with the plotting tools
    print(len(filteredNodes))
    zAx, yAx = zip(*filteredNodes)

    # Plot the 2D comparison of the read pixels
    fig2 = plt.figure()
    plt.scatter(zAx,yAx)
    plt.title('First Iteration')
    plt.xlabel('Z')
    plt.ylabel('Y')

    # Define the axes a bit
    plt.ylim([0,600])
    plt.xlim([0,600])
    plt.gca().invert_yaxis()

    # Project the nodes to 3D space

    '''generate X-Axis '''
    xAx = np.zeros(len(zAx))

    '''Scale Y-Axis '''
    yAx_scld =tuple(y/math.cos(math.pi/4) for y in yAx)

    zAx_scld =tuple(z-centreMark for z in zAx)
    #yAx_scaled = yAx - (np.ones(len(yAx))*centreMark)


    fig3 = plt.figure()
    ax = fig3.gca(projection='3d')
    ax.scatter(xAx,yAx_scld,zAx_scld, label="test")
    ax.legend

    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(0,600)
    ax.set_zlim3d(-100,100)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


    fig3.gca().invert_zaxis()




# Rotate nodes around the centreMark
transformedNodes = []
yAx_scaled = yAx - (np.ones(len(yAx))*centreMark)
distArr = np.sqrt(np.square(xAx) + np.square(zAx_scld))

steps = 8
for r in range(steps):
    
    rotTransArr = np.array([distArr * math.sin(2*math.pi/steps * r),yAx_scaled, distArr * math.cos(2*math.pi/steps * r)])
    
    transformedNodes.append(rotTransArr)

fig4 = plt.figure()
ax2 = fig4.gca(projection='3d')

for nd in range(len(transformedNodes)):
    #testArr = np.transpose(transformedNodes[nd])
    #xAx, yAx, zAx = testArr

    xAxS, yAxS, zAxS = transformedNodes[nd]
    ax2.scatter(xAxS,yAxS,zAxS, label="test")

ax2.set_xlim3d(-300, 300)
ax2.set_ylim3d(-200,200)
ax2.set_zlim3d(-200,200)

ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')

plt.show()

