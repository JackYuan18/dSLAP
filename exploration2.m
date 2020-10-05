close all

n=4;
x=linspace(0,1,n+1);
y=linspace(0,1,n+1);
initial=[0.5,0.5,0];

u=[-0.3,-0.15,0,0.15,0.3];

X=[];
for i=1:n+1
    for j=1:n+1
        X=[X;x(i),x(j),0];
    end
end
car_x=initial;
figure(1)

scatter(car_x(1),car_x(2));
axis([-0.5 1.5 -0.5 1.5]);
hold on
for i=1:1000
    rand_state=rand(2,1);
    dist=1000;
    for j=1:5
        car_next_=car(car_x,u(j));
        car_next__=car(car_next_,u(j));
        dist_=norm(rand_state-car_next__(1:2));
        if dist_<dist
            dist=dist_;
            record=j;
            car_next=car_next_;
        end
        
    end
    car_x=car_next;
    scatter(car_x(1),car_x(2));
    drawnow;
end