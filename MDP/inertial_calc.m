clear;
clc;

mass = 0.01;
radius = 0.004;
height = 0.03;

% z axis
z_ixx = 0.083333 * mass * (3*radius*radius + height*height)
z_iyy = 0.083333 * mass * (3*radius*radius + height*height)
z_izz = 0.5*mass*radius*radius

% y axis 
y_ixx = 0.083333 * mass * (3*radius*radius + height*height)
y_iyy = 0.5*mass*radius*radius
y_izz = 0.083333 * mass * (3*radius*radius + height*height)

% x axis
x_ixx = 0.5*mass*radius*radius
x_iyy = 0.083333 * mass * (3*radius*radius + height*height)
x_izz = 0.083333 * mass * (3*radius*radius + height*height)

