clear;
clc;
% 
% Ry = [cos(pi) 0 sin(pi); 0 1 0; -sin(pi) 0 cos(pi)];
% Rx = [1 0 0; 0 cos(-pi/2) -sin(-pi/2); 0 sin(-pi/2) cos(-pi/2)];
% Rz = [cos(pi) -sin(pi) 0; sin(pi) cos(pi) 0; 0 0 1];
% M = Ry*Rx*Rz
[Rxb, Ryb, Rzb] = RotM(pi/2, pi/2, -pi/2); % joint 1
[Rx, Ry, Rz] = RotM(-pi/2,0, pi); % joint 2
[Rx1, Ry1, Rz1] = RotM(0, pi, 0); % joint 5
M_1_2 = Rzb*Rx1*Ry1*Rz1;
M_3 = Rx1*Ry1*Rz1;
M_t = M_3*Rz;
quat_f = rotm2quat(M_1_2)
% eu = [pi pi -pi/2];
% quat = eul2quat(eu)

% q = [0 -1 0 0];
% e = quat2eul(q) 
% quat_m = [(cos(pi))/2 (sin(pi))/2 (sin(pi))/2 (sin(pi))/2] 

