 %�¶ȷֲ�
clear %����
clc
close all
domain = 250; % �ռ�����(0,domain)
time = 20000; % ʱ������(0,time)
a =0.082; % ��ɢϵ��
c1 =75; % ���ֵ
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
%U(1,:) = g(X(1,:)); % ��ֵ
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
ylabel('ĳλ���¶���ʱ��仯���')
% legend('x=0.035;','x=0.05','x=0.059','x=0.07');
legend('��һ��','�ڶ���','������');






