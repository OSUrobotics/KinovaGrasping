#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # ref: https://stackoverflow.com/questions/27023068/plotting-3d-vectors-using-matplot-lib


#keys=["CylinderM", "CubeB","CubeS", "CylinderS", "CylinderB","Cube45S", "Cube45B","Cone1S", "Cone1B","Cone2S", "Cone2B","Vase1S", "Vase1B","Vase2S", "Vase2B", "HourS", "HourB", "VaseB", "VaseS", "BottleB", "BottleS", "BowlB", "BowlS", "LemonB", "LemonS", "TBottleB", "TBottleS", "RBowlB", "RBowlS"]
keys=["CubeS", "CubeB", "CylinderS", "CylinderB"]
#Pid = ["Expert", "Normal"]
mode = ["rotated", "nrotated"]

for shape in keys:
    # fig_num = str(shape)+'_'+str(PID) 
    # fig = plt.figure(fig_num)
    # ax = fig.add_subplot(111, projection='3d')
    for status in mode:
        coords_filename = str(shape)+'_normal_'+str(status)+'.csv'
        output = []
        x_cord = []
        y_cord = []
        z_cord = []
        x_orr = []
        y_orr = []
        z_orr = []
        finger_values = {}
        j = 1
        with open(coords_filename) as csvfile:
            checker=csvfile.readline()
            if ',' in checker:
                delim=','
            else:
                delim=' '
            reader = csv.reader(csvfile, delimiter=delim)
            for i in reader:
                output.append(i[0])
                x_cord.append(float(i[1][2:]))
                y_cord.append(float(i[2]))
                z_cord.append(float(i[3][:-1]))
                x_orr.append(float(i[4]))
                y_orr.append(float(i[5]))
                z_orr.append(float(i[6]))
                finger_values[j] = [float(i[7][2:]), float(i[8]), float(i[9]), float(i[10]), float(i[11]), float(i[12]), float(i[13]), float(i[14]), float(i[15]), float(i[16]), float(i[17]), float(i[18]), float(i[19]), float(i[20]), float(i[21]), float(i[22]), float(i[23]), float(i[24][:-1])]
                j += 1

                
                
        
            
        #ax.scatter(x_cord, y_cord, z_cord, color=color_graph)
        
        
        # plt.xlabel('X-Axis', color="C0")
        # plt.ylabel('Y-Axis', color="C2")
        # plt.title(fig_num)
        # plt.savefig("Plot/"+str(fig_num) +".jpg")            
#plt.show()