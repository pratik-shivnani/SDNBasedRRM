import time
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
import plotly.express as px
from math import dist
from numpy import sum

roadsiz = int(input("Enter highway simulation length : "))
sections = int(input("Enter the number of sections of the highway : "))
numVeh = int(input("Enter number of vehicles : "))

status = []
roadsize = (roadsiz, 1000)

#pi , vi , diri , dsti , PWi

zorBLX = []
zorBLY = []
zorR = []
zorT = []
lengthEachSection = roadsize[0] / sections

a = [0, lengthEachSection]

sectionsCoordinates = [[
  a[0] + i * lengthEachSection, a[1] + i * lengthEachSection
] for i in range(sections)]

# print(sectionsCoordinates)

zor = []
'''Zor Pre-Decided'''
for i in range(sections):
  zor.append([[sectionsCoordinates[i][0] + 100, 200],
              [sectionsCoordinates[i][1] - 100, 800]])
'''Zor random'''
# for i in range(sections):
#     if i != 0:

#         zorBottomLeftX = sectionsCoordinates[i-1][1] + random.random()*(lengthEachSection*2/5)
#         zorBottomLeftY = roadsize[1]/2 + random.random()*(roadsize[1]*2/5)
#         zorBottom = [zorBottomLeftX, zorBottomLeftY]
#         zorRight = zorBottomLeftX + lengthEachSection//1.5
#         zorTop = zorBottomLeftY + lengthEachSection//1.5
#         zorTopRight = [zorRight, zorTop]
#         zor.append([zorBottom, zorTopRight])
#     else:
#         zorBottomLeftX = random.random()*(sectionsCoordinates[i][0]*2/5)
#         zorBottomLeftY = random.random()*(roadsize[1]*2/5)
#         zorBottom = [zorBottomLeftX, zorBottomLeftY]
#         zorRight = zorBottomLeftX + sectionsCoordinates[i][1]//1.5
#         zorTop = zorBottomLeftY + lengthEachSection//1.5
#         zorTopRight = [zorRight, zorTop]
#         zor.append([zorBottom, zorTopRight])

print('Zor : ', zor)
for i in range(numVeh):
  status.append([[], [], [], [], 0])
'''Position'''

lanes = [100, 300, 500, 700, 900]
vehLanes = [[] for i in range(len(lanes))]
for i in range(numVeh):
  lane = random.randint(0, 4)
  status[i][0].append(random.random() * (roadsize[0] - 50) + 100)
  status[i][0].append(lanes[lane])
  vehLanes[lane].append(i)
above = []

for i in range(len(status)):
  above.append(i)

# @note sorting velocity part
velocities = []
for i in range(len(vehLanes)):
  temp = []
  for j in range(len(vehLanes[i])):
    temp.append(random.random()*60)
  temp.sort()
  velocities.append(temp)   

vel = [{} for i in range(len(lanes))]
for i in range(len(vehLanes)):
  for j in vehLanes[i]:
    vel[i][str(j)] = status[j][0][0]

for i in range(len(vel)):
  vel[i] = sorted(vel[i].items(), key = lambda x:x[1])
  vel[i] = dict(vel[i])


# print(vel)

for i in range(len(vehLanes)):
  listV = list(vel[i].keys())
  for j in range(len(listV)):
    status[int(listV[j])][1].append(velocities[i][j])
    status[int(listV[j])][1].append(0)




print("Vehicles : ", above)
'''Velocity'''
# right moving vehicles
# xVel -> +ve
# yVel -> 0
# for i in above:

#   status[i][1].append(random.random() * 60)
#   status[i][1].append(0)
''' Direction '''

# right moving vehicles
# xDir -> +1
# yDir -> 0
for i in above:

  status[i][2].append(1)
  status[i][2].append(0)
'''Destination'''

for i in above:

  status[i][3].append(roadsiz)
  status[i][3].append(status[i][0][1])

#Printing properly:
for i in range(len(status)):
  print(
    "Vehicle Number : {}, Position : [{:.2f},{:.2f}], Velocity : [{:.2f},{:.2f}], Direction : {}, Destination: {},{:.2f}"
    .format(i, status[i][0][0], status[i][0][1], status[i][1][0],
            status[i][1][1], status[i][2][0], status[i][3][0],
            status[i][3][1]))

xpoints_a = [status[x][0][0] for x in above]
ypoints_a = [status[x][0][1] for x in above]

# print(xpoints_a, ' ' ,ypoints_b)
'''PLOTLY PLOT


s = time.time()

fig1 = px.scatter(x=xpoints,y=ypoints)
# fig1.show()
e = time.time()
print("Plotly Time : ", e-s)

'''
'''Only Working with above for now'''

# print("inZor : ", inZor)

# leaderMax = 0
# leaderInd = -1
# for i in above:
#     inZorIndices = [int(i.split(':')[0]) for i in inZor]
#     if status[i][0][0] > leaderMax and i in inZorIndices:
#         leaderMax = status[i][0][0]
#         leaderInd = i
# print(f"Leader for Above : {leaderInd, leaderMax}")
'''Predictions'''
"""

vSafeAbove = [0 for i in inZor]
vDesAbove = [0 for i in inZor]

tou = 0.75
l = 5 #m
b = 4.572
a = 27

vmax = 0
deltaT = 0.01
ita = 0.01
for i in status:
    if i[1][0] > vmax:
        vmax = i[1][0]
for i in range(len(inZor)):
    if inZor[i] != leaderInd:
        index = inZor[i]
        vl = status[leaderInd][1][0]
        gt = status[leaderInd][0][0] - status[index][0][0]-l
        gdes = tou * status[leaderInd][0][0]
        vbar = (status[leaderInd][0][0] + status[index][0][0])/2
        touB = vbar/b
        

        vsafe = vl + (gt - gdes)/(touB+tou)

        # print(vsafe)

        vSafeAbove[i] = vsafe

        ## V-DESIRED

        vdes = min(vmax, vsafe , status[index][1][0] + a*deltaT )
        # print("trial : ",status[index][1][0] + a*deltaT , " max : ", vmax , " vsafe : ", vsafe )

        vtdeltaT = max(0, vdes - ita)
        xtdeltaT = status[index][0][0] + status[index][1][0] * deltaT


        print("V. no: {}, V-safe : {:.2f}, V-des : {:.2f}, v(t+deltaT) : {:.2f}, x(t+deltaT) : {:.2f}".format(index, vsafe, vdes,vtdeltaT,xtdeltaT))


"""
'''MATLAB PLOTTING'''

optDel = []
norDel = []

timeChange = float(
  input(
    "Enter the time after which the position should be predicted (seconds) : ")
)
iterations = int(input("Enter the number of iterations : "))

v2vNum = [0 for i in range(iterations)]
celNum = [0 for i in range(iterations)]
main_through = []
for o in range(iterations):
  """inZor"""
  inZor = []
  for i in range(len(status)):
    for j in range(len(zor)):

      if status[i][0][0] >= zor[j][0][0] and status[i][0][0] <= zor[j][1][
          0] and status[i][0][1] >= zor[j][0][1] and status[i][0][1] <= zor[j][
            1][1]:
        inZor.append(str(i) + ':' + str(j))
  leaderMax = [0 for i in range(sections)]
  leaderInd = [-1 for i in range(sections)]

  for k in range(sections):
    inZorIndices = [int(i.split(':')[0]) for i in inZor]
    inZorSections = [int(i.split(':')[1]) for i in inZor]

    for i in inZorIndices:

      sectionElement = inZorSections[inZorIndices.index(i)]
      if status[i][0][0] > leaderMax[sectionElement] and i in inZorIndices:

        leaderMax[sectionElement] = status[i][0][0]
        leaderInd[sectionElement] = i

  # print(f"Leader for Section {k} : {leaderInd[k], leaderMax[k]}")
  xpoints_a = [
    status[x][0][0] for x in above
    if status[x][0][0] >= 0 and status[x][0][0] <= roadsiz
  ]
  xcor_a = [
    x for x in above if status[x][0][0] >= 0 and status[x][0][0] <= roadsiz
  ]

  ypoints_a = [
    status[x][0][1] for x in above
    if status[x][0][0] >= 0 and status[x][0][0] <= roadsiz
  ]
  ycor_a = [
    x for x in above if status[x][0][0] >= 0 and status[x][0][0] <= roadsiz
  ]

  s1 = time.time()
  fig = plt.figure()
  fig.set_figwidth(12)
  fig.set_figheight(6)
  ax = fig.add_subplot()

  for i in range(sections):

    a = ax.add_patch(
      Rectangle((zor[i][0][0], zor[i][0][1]),
                lengthEachSection - 200,
                600,
                fc='none',
                ec='g',
                lw=1))

  # ax.text(zorBottomLeftX,zorBottomLeftY-35, 'Zone Of Relevance(ZOR)', fontsize = 10)

  ax.plot(xpoints_a, ypoints_a, '>')

  # print(above,below,sep="___")
  counter = 0
  for x, y in zip(xpoints_a, ypoints_a):

    label = "{}".format(xcor_a[counter])
    counter += 1

    ax.annotate(
      label,  # this is the text
      (x, y),  # these are the coordinates to position the label
      textcoords="offset points",  # how to position the text
      xytext=(0, 10),  # distance from text to points (x,y)
      ha='center')  # horizontal alignment can be left, right or center

  plt.xlabel("X-coordinate", fontsize=15)
  plt.ylabel("Y-coordinate", fontsize=15)

  # x = [0,roadsize[0]]
  # y = [roadsize[1]/2,roadsize[1]/2]

  # ax.plot(x,y)

  for i in range(sections):
    x = [sectionsCoordinates[i][1], sectionsCoordinates[i][1]]
    y = [0, roadsize[1]]
    ax.plot(x, y)

  for i in range(sections):
    posY = 1200
    posX = (sectionsCoordinates[i][0] + sectionsCoordinates[i][1]) / 2
    if leaderInd[i] != -1:
      index = leaderInd[i]
      posX2 = status[index][0][0]
      posY2 = status[index][0][1]
      ax.plot([posX, posX2], [posY, posY2], '--')
      if (roadsiz - posX2) > (100 * roadsiz / 1000) / 2:
        ax.add_patch(
          Ellipse((posX2, posY2),
                  100 * roadsiz / 1000,
                  200,
                  angle=0,
                  color='red',
                  fill=False))
      # print('PosX2, PosY2 : ', posX2, posY2, sep = ' ')

    ax.scatter(posX, posY, c='r')

  # ax.add_patch(Circle((100,200), 100))

  e1 = time.time()

  print("Matplotlib time : ", e1 - s1)

  # @note Graphing part
  # - Decide which vehicle is in which section

  VehicleSection = [[] for i in range(sections)]
  LaneVehicles = [[] for i in range(len(lanes))]
  for i in above:
    for j in range(len(sectionsCoordinates)):
      if status[i][0][0] >= sectionsCoordinates[j][0] and status[i][0][
          0] < sectionsCoordinates[j][1]:
        VehicleSection[j].append(i)
    LaneVehicles[(status[i][0][1] //100 +1) //2 -1].append(i)
  print('Lane Vehicles : ', LaneVehicles)
  def MyFn(s):
    return status[s][0][0]
  for i in range(len(LaneVehicles)):
    
    LaneVehicles[i] = sorted(LaneVehicles[i], key=MyFn, reverse=True)
  print('Lane Vehicles : ', LaneVehicles)

    
  # print('Vehicle Section : ' ,VehicleSection)
  # @note Variables regarding delay
  avgTime = []
  normTime = []
  delay = 200  #ms
  delayLeader = 100  #ms
  numd2d = [0 for i in range(sections)]
  numcel = [0 for i in range(sections)]
  throughput = []

  # @note Section Loop(main)
  for i in range(sections):
    tp = 1000 #ms
    temp_tp = 0
    packet_loss = 0

    numVehSec = len(VehicleSection[i])
    normTime.append(numVehSec * delay)
    leaderSec = leaderInd[i]

    if leaderSec == -1:
      if numVehSec == 0:
        avgTime.append(0)
        throughput.append(0)
        continue
      else:
        # print(" No Leader, Section Number : {}".format(i))
        avgTime.append(numVehSec * delay)
        temp_tp = tp - (numVehSec * delay)
        if temp_tp < 0:
          packet_reached = numVehSec + (temp_tp / delay )
          throughput.append(packet_loss/numVehSec)
        else:
          throughput.append(1.0)

        numcel[i] += numVehSec
        continue

    distMat = [[0 for i in range(numVehSec)] for i in range(numVehSec)]

    for j in range(numVehSec):
      for k in range(j + 1, numVehSec):
        # print(f'j , k : {j}, {k}, section : {i}, dist : {dist([status[j][0][0], status[j][0][1]],[status[k][0][0], status[k][0][1]] )}')
        distMat[j][k] = dist([
          status[VehicleSection[i][j]][0][0],
          status[VehicleSection[i][j]][0][1]
        ], [
          status[VehicleSection[i][k]][0][0],
          status[VehicleSection[i][k]][0][1]
        ])

    msgReached = VehicleSection[i]
    if len(msgReached) == 1:
      avgTime.append(delay)
      throughput.append(1)
      numcel[i] += 1
      continue
    localDel = 0
    thresh = 500
    # print(f'Vehicle Section : {VehicleSection[i]}')
    for j in VehicleSection[i]:
      if j == leaderSec:
        tp -= delay
        localDel += delay
        numcel[i] += 1
      else:
        # print(f' Distance between {j} , {leaderSec} : { distMat[VehicleSection[i].index(min(j, leaderSec))] [VehicleSection[i].index(max(j,leaderSec))]}')
        if distMat[VehicleSection[i].index(min(j, leaderSec))][VehicleSection[i].index(max(j,leaderSec))] < thresh:
          localDel += delayLeader
          tp-= delayLeader
          numd2d[i] += 1
        else:
          localDel += delay
          tp-=delay
          numcel[i] += 1
    avgTime.append(localDel)
    vehNum = len(VehicleSection[i])
    temp_tp = tp
    print('tp : ', temp_tp)

     
    if temp_tp < 0:
      packet_reached = vehNum + (temp_tp / delay )
      print('pr = ', packet_reached)
      throughput.append(packet_reached/vehNum)
    else:
      throughput.append(1.0)
    print('throughput = ', throughput)
    # if leaderSec != -1:
    #     for i in VehicleSection[i]:

    # print(f'Section Number : {i}, distMat, {distMat}')

  # print(f'Average Delay : {avgTime}')
  # print(f'Normal Delay : {normTime}')
  print(throughput)
  main_through.append(np.average(throughput))
  optDel.append(avgTime)
  norDel.append(normTime)

  # @note Section prediction
  currentSection = []
  newSection = []
  # print(sectionsCoordinates)
  for i in range(len(status)):
    for j in range(len(sectionsCoordinates)):
      if status[i][0][0] >= sectionsCoordinates[j][0] and status[i][0][
          0] <= sectionsCoordinates[j][1]:
        currentSection.append(j)
        break
      elif status[i][0][0] > roadsiz:
        currentSection.append(sections)
        break
      elif status[i][0][0] < 0:
        currentSection.append(-1)
        break

    newPosition = status[i][1][0] * timeChange + status[i][0][0]
    for j in range(sections):
      # print(sectionsCoordinates[j][0], sectionsCoordinates[j][1])
      if newPosition >= sectionsCoordinates[j][
          0] and newPosition <= sectionsCoordinates[j][1]:
        newSection.append(j)
        break
      elif newPosition < sectionsCoordinates[0][0]:
        newSection.append(-1)
        break
      elif newPosition > sectionsCoordinates[sections - 1][1]:
        newSection.append(sections)
        break
    status[i][0][0] = newPosition
    # print("i : ", i, "Old Position : " , status[i][0][0], " | New Position : " , newPosition)
  v2vNum[o] = sum(numd2d)
  celNum[o] = sum(numcel)

  # print(currentSection)
  # print(newSection)
  print('Iteration Number : ', o)
  for i in range(numVeh):
    if currentSection[i] != newSection[i]:
      print("Vehicle {} changed from Section {} to {}".format(
        i, currentSection[i], newSection[i]))

  plt.savefig('gif/foo{}.png'.format(o), bbox_inches='tight')
  plt.close()

  # plt.show()
from image2gif import img2gif

print(f'Normal Delay : {norDel}\nOptimized Delay : {optDel}')

fig, axs = plt.subplots(2, 2, figsize=(12, 6))

x_c = [0, 0, 1, 1]
y_c = [0, 1, 0, 1]

for i in range(sections):
  x = [i for i in range(len(norDel))]
  yNor = [norDel[j][i] for j in range(len(norDel))]
  yOpt = [optDel[j][i] for j in range(len(optDel))]
  axs[x_c[i], y_c[i]].plot(x, yNor, label='Normal Delay')
  axs[x_c[i], y_c[i]].plot(x, yOpt, '--', label='Optimized Delay')
  axs[x_c[i], y_c[i]].set_title(f'Section : {i}')
  axs[x_c[i], y_c[i]].legend(loc='upper left')

for ax in axs.flat:
  ax.set(xlabel='Iterations', ylabel='Delay')

img2gif(iterations)
print("Done")
plt.savefig('graphs/iterationsGraph.png')
plt.close()


print('Cellular Connections : ', celNum)
print('V2V Connections : ', v2vNum)

fig = plt.figure()
fig.set_figwidth(8)
fig.set_figheight(4)
ax = fig.add_subplot()
x = [i for i in range(iterations)]
ax.plot(x, celNum, label='Cellular Connections') # change it to v2n
ax.plot(x, v2vNum, label='V2V Connections')
ax.set_title('Number of Cellular connections vs V2V connections')
ax.legend(loc='upper left')
ax.set(xlabel='Iterations', ylabel='Number of Connections')

plt.savefig('graphs/connections.png')
plt.close()

band_saved = [v2vNum[i]/(v2vNum[i] +celNum[i]) *100 for i in range(len(v2vNum)) ]

fig = plt.figure()
fig.set_figwidth(8)
fig.set_figheight(4)
ax = fig.add_subplot()
x = [i for i in range(iterations)]

ax.plot(x,band_saved, label='Cellular Band Saved')
print('band_saved :', band_saved)
ax.set_title('Percentage of Cellular Bandwidth reduced by V2V')
ax.legend(loc='upper left')
ax.set(xlabel='Iterations', ylabel='Percentage of bandwidth saved')

plt.savefig('graphs/bandSaved.png')
plt.close()


fig = plt.figure()
fig.set_figwidth(8)
fig.set_figheight(4)
ax = fig.add_subplot()
x = [i for i in range(iterations)]
ax.plot(x, main_through, label='Throughput')
ax.set_title('Throughput')
ax.legend(loc='upper left')
ax.set(xlabel='Iterations', ylabel='Throughput')
plt.savefig('graphs/throughput.png')
plt.close()


# plt.show()

# x = [i for i in range(iterations)]
# ax.plot(x, v2vNum, label  = 'V2V Users')
# ax.plot(x, celNum, label = 'Cellular Users')
# ax.legend()
# plt.savefig('Users.png')


print('Done')
time.sleep(5)
