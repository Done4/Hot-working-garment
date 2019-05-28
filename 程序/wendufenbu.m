 %温度分布
clear %调试
clc
close all
domain = 250; % 空间区域(0,domain)
time = 20000; % 时间区域(0,time)
a =0.082; % 扩散系数
c1 =75; % 左边值
c2 = 25; % 右边值



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
%U(1,:) = g(X(1,:)); % 初值
U(1,:) = g;

for n=1:N
    for j=2:J
        U(n+1,j) = (1-2*a*lambda)*U(n,j) + a*lambda*(U(n,j+1)+U(n,j-1));
    end
end

figure
u3=U(:,3);
x3=(1:length(u3))/3.7;
u8=U(:,4);
x8=(1:length(u8))/3.7;
u12=U(:,5);
x12=(1:length(u12))/3.7;
u16=U(:,6);
x16=(1:length(u16))/3.7;
plot(x3,u3,x8,u8,x12,u12)
xlabel('t')
ylabel('某位置温度随时间变化情况')
% legend('x=0.035;','x=0.05','x=0.059','x=0.07');
legend('第一层','第二层','第三层');






