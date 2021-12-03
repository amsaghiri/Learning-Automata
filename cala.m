%NUMBER_OF_FEATURES
n=2;
%NUMBER OF ACTIONS
a=2;
%LEARNING RATE
landa=0.01;
%BOUNDRIES
L=ones(1,n+1)*20;
% INERTIA
K=0.1;
sigmal=0.1;
%INTERNAL STATE
U=rand(a,n+1)*2-1;
LT=rand(2,n+1);
LT(1,:)=LT(1,:)*10-5;
LT(2,:)=LT(2,:);
RT=rand(2,n+1);
RT(1,:)=RT(1,:)*10-5;
RT(2,:)=RT(2,:);
%give a context vector and select an action
for t=1:10000
    X=rand(1,n+1)*4-2;
    X(end)=1;
    if(X(1)<0)
        z=10*X(1)+3*X(2)+2.5;
    else
        z=1.2*X(1)-5*X(2)+1.5;
    end
    g=exp(-X*U')/sum(exp(-X*U'));
    r=rand;
    m=0;
    for i=1:a
        if r<m+g(i)
            action=i;
            break;
        else
            m=m+g(i);
        end
    end
    if action==1
        %activating left team of CALA
        acts=LT(1,:)+(randn(1,n+1).*max(sigmal,LT(2,:)));
        R1=sum(acts.*X);
        R2=sum(LT(1,:).*X);
        B1=1/(1+(z-R1)^2);
        B2=1/(1+(z-R2)^2);
        LT(1,:)=LT(1,:)+landa*((B1-B2)./max(sigmal,RT(2,:))).*((acts-RT(1,:))./max(sigmal,RT(2,:)));
        LT(2,:)=LT(2,:)+landa*((B1-B2)./max(sigmal,RT(2,:))).*(((acts-RT(1,:))./max(sigmal,RT(2,:))).^2-1)...
            -landa*K*(LT(2,:)-sigmal);
    else
        %activating right team of CALA
        acts=RT(1,:)+(randn(1,n+1).*max(sigmal,RT(2,:)));
        R1=sum(acts.*X);
        R2=sum(RT(1,:).*X);
        B1=1/(1+(z-R1)^2);
        B2=1/(1+(z-R2)^2);
        RT(1,:)=RT(1,:)+landa*((B1-B2)./max(sigmal,RT(2,:))).*((acts-RT(1,:))./max(sigmal,RT(2,:)));
        RT(2,:)=RT(2,:)+landa*((B1-B2)./max(sigmal,RT(2,:))).*(((acts-RT(1,:))./max(sigmal,RT(2,:))).^2-1)...
            -landa*K*(RT(2,:)-sigmal);
    end
    beta=B2;
    HU=U;
    for i=1:a
        HU(i,:) = max(min(HU(i,:), L), -L);
    end
    g=exp(-X*HU')/sum(exp(-X*HU'));
    for i=1:a
        U(i,:)=U(i,:)+landa*beta*-X*((i==action)-g(i))+landa*K*(HU(i,:)-U(i,:));
    end
END