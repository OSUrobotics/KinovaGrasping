function [Rx,Ry,Rz] = RotM(x, y, z)
Ry = [cos(y) 0 sin(y); 0 1 0; -sin(y) 0 cos(y)];
Rx = [1 0 0; 0 cos(x) -sin(x); 0 sin(x) cos(x)];
Rz = [cos(z) -sin(z) 0; sin(z) cos(z) 0; 0 0 1];

end