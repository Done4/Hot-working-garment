options = optimset('largescale','off');
[x]=llw('fun1',rand(3,1),[],[],[],[],zeros(3,1),[],'fun2',options)
%��ʼֵ�Ǹ����������
y=5.5