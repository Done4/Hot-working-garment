options = optimset('largescale','off');
[x]=llw('fun1',rand(3,1),[],[],[],[],zeros(3,1),[],'fun2',options)
%初始值是个随意的数字
y=5.5