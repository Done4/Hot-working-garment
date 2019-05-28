%������
clear 
clc
close all
domain = 250; % �ռ�����(0,domain)
time = 20000; % ʱ������(0,time)
a =0.045; % ��ɢϵ��
c1 =72.75; % ���ֵ
c2 = 25; % �ұ�ֵ
g=25;
h = 1.5; % �ռ䲽��
tau = 1; % ʱ�䲽��
lambda = tau / h^2; % �����
J = fix(domain / h); % �ռ�����ȷ���
N = fix(time / tau); % ʱ������ȷ���
X = 0:h:domain;
T = 0:tau:time;
[X,T] = meshgrid(X,T); % ��������

U = zeros(N+1,J+1);
U(:,1) = c1; % ���ֵ
U(:,end) = c2; % �ұ�ֵ

U(1,:) = g;
for n=1:N
    for j=2:J
        U(n+1,j) = (1-2*a*lambda)*U(n,j) + a*lambda*(U(n,j+1)+U(n,j-1));
    end
end

figure;
XX=X(:,1:50);
TT=T(:,1:50);
UU=U(:,1:50);
surf(XX,TT,UU);
title('�¶ȵķֲ�');
xlabel('x');
ylabel('t');
zlabel('�¶�U');
shading interp;

u90=U(:,6);
n=length(u90);
maxd=u90(1:3.7:n);