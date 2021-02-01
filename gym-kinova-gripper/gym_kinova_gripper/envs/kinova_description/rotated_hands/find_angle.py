import numpy as np 

a = np.loadtxt("CubeB_side_rotation.txt")
a_x_max = np.max(a[:][0])
a_x_min = np.min(a[:][0])

a_y_max = np.max(a[:][1])
a_y_min = np.min(a[:][1])

a_z_max = np.max(a[:][2])
a_z_min = np.min(a[:][2])

print("X")
print ("Max")
print (a_x_max)
print ("Min")
print (a_x_min)
print ("Centre")
print (a_x_min + (a_x_max - a_x_min)/2)

print("Y")
print ("Max")
print (a_y_max)
print ("Min")
print (a_y_min)
print ("Centre")
print (a_y_min + (a_y_max - a_y_min)/2)

print("Z")
print ("Max")
print (a_z_max)
print ("Min")
print (a_z_min)
print ("Centre")
print (a_z_min + (a_z_max - a_z_min)/2)


