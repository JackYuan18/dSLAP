close all

n=4;
x=linspace(0,1,n+1);
y=linspace(0,1,n+1);
initial=[0.1,0.15,0];

u=[-0.3,-0.15,0,0.15,0.3];

X=[];
for i=1:n+1
    for j=1:n+1
        X=[X;x(i),x(j),0];
    end
end
car_x=initial;
XX=X(:,1:2);
mdl=KDTreeSearcher(XX);
figure(1);
scatter(X(:,1),X(:,2),'filled','black');
hold on
CAR=[car_x];    
for i=1:3000
    
    record=knnsearch(mdl,car_x(1:2));
    
    if X(record,3)~=1

        X(record,3)=1;
    end
    

    flag=1;
    while flag
        flag=0;
        for k=1:(n+1)^2
            if X(k,3)~=1
                index=rangesearch(mdl,X(k,1:2),1/n);
                index=index{1};
                index(index==k)=[];
                [val,ind]=max(X(index,3));
                if X(k,3)~=val-0.5
                    X(k,3)=val-0.5;
                    flag=1;
                end
            end
        end
    end
    
    
    Diff=X(:,1:2)-car_x(1:2);
    
   
    Y=X(:,3);    
    [val,index]=min(Y);

    dist=1000;
    for j=1:5
        car_next_=car(car_x,u(j));
        car_next__=car(car_next_,u(j));
        dist_=norm(X(index,1:2)-car_next__(1:2));
        if dist_<dist
            dist=dist_;
            record=j;
            car_next=car_next_;
        end
        
    end
    if mod(i,100)==0
            index
    end
    car_x=car_next;
%     CAR=[CAR;car_next];
    
    scatter(X(index,1),X(index,2),'filled','green');
    scatter(car_next(1),car_next(2),'blue');
    drawnow;
        
    
end

        
        