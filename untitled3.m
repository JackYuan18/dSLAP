close all
figure()
p=1/5;
x=linspace(0,1,100);
y=linspace(0,1,100);
[X,Y]=meshgrid(x,y);
Z=(X.^p+Y.^p).^(1/p);
Z(Z<=1)=1;
Z(Z>1)=2;
contourf(Z)