
clc
tic;%tic1
t1=clock;
for i=1:3
    tic ;%tic2
    t2=clock;
    pause(3*rand)
    % ���㵽��һ������tic��ʱ�䣬���仰˵����ÿ��ѭ����ʱ��
    disp(['toc�����',num2str(i),'��ѭ������ʱ�䣺',num2str(toc)]);
    %����ÿ��ѭ����ʱ��
    disp(['etime�����',num2str(i),'��ѭ������ʱ �䣺',num2str(etime(clock,t2))]);
    %��������ܹ�������ʱ��
    disp(['etime�������ӿ�ʼ���������е�ʱ��:',num2str(etime(clock,t1))]);
    disp('======================================')
end
%�����ʱ��tic2��ʱ�䣬�������һ������tic����forѭ����i=3ʱ�����Լ���������һ��ѭ����ʱ��
disp(['toc�������һ��ѭ������ʱ��',num2str(toc)])
disp(['etime����������ʱ�䣺',num2str(etime(clock,t1))]);