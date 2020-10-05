function x_next=car(X,u)
    theta=X(3);
    xdot=0.5*cos(theta);
    ydot=0.5*sin(theta);
    theta_dot=0.5/0.05*tan(u);
    
    Xdot=[xdot,ydot,theta_dot];
    
    x_next=X+0.01*Xdot;
end