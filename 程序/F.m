%稳态的方程求解
clear 
clc
close all
result=solve('119*x-37*y=6150','222*x-267*y+45*z=0','225*y-325.8*z=-4846.464','x','y','z');
result.x
result.y
result.z