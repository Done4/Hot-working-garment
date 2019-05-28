%第一层
clear 
clc
close all
domain = 250; % 空间区域(0,domain)
time = 20000; % 时间区域(0,time)
a =0.082; % 扩散系数
c1 =75; % 左边值
c2 =25; % 右边值
g=25;
h = 1.5; % 空间步长
tau = 1; % 时间步长
lambda = tau / h^2; % 网格比
J = fix(domain / h); % 空间区域等分数
N = fix(time / tau); % 时间区域等分数
X = 0:h:domain;
T = 0:tau:time;
[X,T] = meshgrid(X,T); % 生成网格

U = zeros(N+1,J+1);
U(:,1) = c1; % 左边值
U(:,end) = c2; % 右边值

U(1,:) = g;
for n=1:N
    for j=2:J
        U(n+1,j) = (1-2*a*lambda)*U(n,j) + a*lambda*(U(n,j+1)+U(n,j-1));
    end
end

figure;
surf(X,T,U);
title('温度的分布');
xlabel('x')
ylabel('t')
zlabel('温度U')
shading interp;

u90=U(:,2);
n=length(u90);
maxd=u90(1:3.7:n);