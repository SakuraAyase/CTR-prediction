clear all;
clc;


% % %%
% d1 = double(sc_load('data\embed_201710280200.dat')');
% d2 = double(sc_load('data\embed_201710281000.dat')');
% d3 = double(sc_load('data\embed_201710281800.dat')');
% % d4 = double(sc_load('data\embed_201711020200.dat')');
% % d5 = double(sc_load('data\embed_201711021000.dat')');
% % d6 = double(sc_load('data\embed_201711021800.dat')');
% %label_1 = double(sc_load('data\label_201710280200.dat')');
% 
% data1 = [d1;d2;d3];
% % data2 = [d4;d5;d6];
% save data1;
%% 
load('data1.mat');
data1=data1(1:100,:);
k=1;   %将样本降到k维参数设置  
G=35;
D=100;
[m,n] = size(data1);
randSelect = randi(m, D, 1);
pca_result=zeros(D,G);
tic;
for i=1:G
    f_i=data1(:,(30*(i-1)+1):30*i);
    fi_s = f_i(randSelect, :);     
    [pca_fi, COEFF] = fastPCA(fi_s, k);
    pca_result(:,i) = pca_fi;
end
disp(['计算第pca运行时间：',num2str(toc)]);
%%
% load('pca_r4.mat')
r1=pca_result;
cor_d1=ones(35,35);
tic;
for i =1:34
    for j=i+1:35
        x=r1(:,i);
        y=r1(:,j);
        p_i=corrcoef(x,y); %fi~fi-35;
        cor_d1(i,j)=p_i(1,2);
        cor_d1(j,i)=cor_d1(i,j);
    end
end
disp(['计算第correlation运行时间：',num2str(toc)]);
save cor_d1
%%
% cor_d1=ones(35,35);
% tmp=zeros(30,1);
% for j=0:34
%     for k=j+1:34        
%         for i=1:30
%             x=data1(:,i+j*30);
%             y=data1(:,i+k*30);
%             p_i=corrcoef(x,y); %fi~fi-35;
%             tmp(i,:)=(p_i(1,2));
%         end
%         cor_d1(j+1,k+1)=(mean(tmp,1));
%         cor_d1(k+1,j+1)=cor_d1(j+1,k+1);
%     end
% end
% cor_d2=ones(35,35);
% tmp=zeros(30,1);
% for j=0:34
%     for k=j+1:34        
%         for i=1:30
%             x=data2(:,i+j*30);
%             y=data2(:,i+k*30);
%             p_i=corrcoef(x,y); %fi~fi-35;
%             tmp(i,:)=(p_i(1,2));
%         end
%         cor_d2(j+1,k+1)=(mean(tmp,1));
%         cor_d2(k+1,j+1)=cor_d2(j+1,k+1);
%     end
% end
% 
% s=0.0;
% for i=1:35
%     for j=1:35
%         t = (cor_d1(i,j)-cor_d2(i,j)).^2;
%         s=s+t;
%     end
% end
% s=s/35.0/35.0;

%%
% load('cor_d2.mat');
% m1=mean(p1,1);
% m2=mean(p2,1);
% m3=mean(p3,1);
% m4=mean(p4,1);
% m5=mean(p5,1);
% m6=mean(p6,1);
% m7=mean(p7,1);
% m8=mean(p8,1);
% m9=mean(p9,1);
% m10=mean(p10,1);
% m11=mean(p11,1);
% m12=mean(p12,1);
% m13=mean(p13,1);
% m14=mean(p14,1);
% m15=mean(p15,1);
% m16=mean(p16,1);
% m17=mean(p17,1);
% m18=mean(p18,1);
% m19=mean(p19,1);
% m20=mean(p20,1);
% m21=mean(p21,1);
% m22=mean(p22,1);
% m23=mean(p23,1);
% m24=mean(p24,1);
% m25=mean(p25,1);
% m26=mean(p26,1);
% m27=mean(p27,1);
% m28=mean(p28,1);
% m29=mean(p29,1);
% m30=mean(p30,1);
% m31=mean(p31,1);
% m32=mean(p32,1);
% m33=mean(p33,1);
% m34=mean(p34,1);
%%
% 
% d1 = double(sc_load('data\embed_201710280200.dat')');
% d2 = double(sc_load('data\embed_201710281000.dat')');
% d3 = double(sc_load('data\embed_201710281800.dat')');
% d4 = double(sc_load('data\embed_201711020200.dat')');
% d5 = double(sc_load('data\embed_201711021000.dat')');
% d6 = double(sc_load('data\embed_201711021800.dat')');
% %label_1 = double(sc_load('data\label_201710280200.dat')');
% 
% data1 = [d1;d2;d3];
% data2 = [d4;d5;d6];
% 
% p1=zeros(30,34);
% for j=1:34
%     for i=1:30
%         x=data2(:,i);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f1~f2-35;
%         p1(i,j)=p_i(1,2);
%     end
% end
% 
% p2=zeros(30,33);
% for j=2:34
%     for i=1:30
%         x=data2(:,i+30);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f2~f3-35;
%         p2(i,j-1)=p_i(1,2);
%     end
% end
% 
% p3=zeros(30,32);
% for j=3:34
%     for i=1:30
%         x=data2(:,i+30*2);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f3~f4-35;
%         p3(i,j-2)=p_i(1,2);
%     end
% end
% 
% p4=zeros(30,31);
% for j=4:34
%     for i=1:30
%         x=data2(:,i+30*3);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f4~f5-35;
%         p4(i,j-3)=p_i(1,2);
%     end
% end
% 
% p5=zeros(30,30);
% for j=5:34
%     for i=1:30
%         x=data2(:,i+30*4);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f5~f6-35;
%         p5(i,j-4)=p_i(1,2);
%     end
% end
% 
% p6=zeros(30,29);
% for j=6:34
%     for i=1:30
%         x=data2(:,i+30*5);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f6~f7-35;
%         p6(i,j-5)=p_i(1,2);
%     end
% end
% 
% p7=zeros(30,28);
% for j=7:34
%     for i=1:30
%         x=data2(:,i+30*6);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f7~f8-35;
%         p7(i,j-6)=p_i(1,2);
%     end
% end
% 
% p8=zeros(30,27);
% for j=8:34
%     for i=1:30
%         x=data2(:,i+30*7);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f8~f9-35;
%         p8(i,j-7)=p_i(1,2);
%     end
% end
% 
% p9=zeros(30,26);
% for j=9:34
%     for i=1:30
%         x=data2(:,i+30*8);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f9~f10-35;
%         p9(i,j-8)=p_i(1,2);
%     end
% end
% 
% p10=zeros(30,25);
% for j=10:34
%     for i=1:30
%         x=data2(:,i+30*9);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f10~f11-35;
%         p10(i,j-9)=p_i(1,2);
%     end
% end
% 
% p11=zeros(30,24);
% for j=11:34
%     for i=1:30
%         x=data2(:,i+30*10);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f11~f12-35;
%         p11(i,j-10)=p_i(1,2);
%     end
% end
% 
% p12=zeros(30,23);
% for j=12:34
%     for i=1:30
%         x=data2(:,i+30*11);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f12~f13-35;
%         p12(i,j-11)=p_i(1,2);
%     end
% end
% 
% p13=zeros(30,22);
% for j=13:34
%     for i=1:30
%         x=data2(:,i+30*12);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f13~f14-35;
%         p13(i,j-12)=p_i(1,2);
%     end
% end
% 
% p14=zeros(30,21);
% for j=14:34
%     for i=1:30
%         x=data2(:,i+30*13);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p14(i,j-13)=p_i(1,2);
%     end
% end
% 
% p15=zeros(30,20);
% for j=15:34
%     for i=1:30
%         x=data2(:,i+30*14);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p15(i,j-14)=p_i(1,2);
%     end
% end
% 
% p16=zeros(30,19);
% for j=16:34
%     for i=1:30
%         x=data2(:,i+30*15);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f15~f16-35;
%         p16(i,j-15)=p_i(1,2);
%     end
% end
% 
% p17=zeros(30,18);
% for j=17:34
%     for i=1:30
%         x=data2(:,i+30*16);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p17(i,j-16)=p_i(1,2);
%     end
% end
% 
% p18=zeros(30,17);
% for j=18:34
%     for i=1:30
%         x=data2(:,i+30*17);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p18(i,j-17)=p_i(1,2);
%     end
% end
% 
% p19=zeros(30,16);
% for j=19:34
%     for i=1:30
%         x=data2(:,i+30*18);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p19(i,j-18)=p_i(1,2);
%     end
% end
% 
% p20=zeros(30,15);
% for j=20:34
%     for i=1:30
%         x=data2(:,i+30*19);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p20(i,j-19)=p_i(1,2);
%     end
% end
% 
% p21=zeros(30,14);
% for j=21:34
%     for i=1:30
%         x=data2(:,i+30*20);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p21(i,j-20)=p_i(1,2);
%     end
% end
% 
% p22=zeros(30,13);
% for j=22:34
%     for i=1:30
%         x=data2(:,i+30*20);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p22(i,j-21)=p_i(1,2);
%     end
% end
% 
% p23=zeros(30,12);
% for j=23:34
%     for i=1:30
%         x=data2(:,i+30*22);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p23(i,j-22)=p_i(1,2);
%     end
% end
% 
% p24=zeros(30,11);
% for j=24:34
%     for i=1:30
%         x=data2(:,i+30*23);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p24(i,j-23)=p_i(1,2);
%     end
% end
% 
% p25=zeros(30,10);
% for j=25:34
%     for i=1:30
%         x=data2(:,i+30*24);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p25(i,j-24)=p_i(1,2);
%     end
% end
% 
% p26=zeros(30,9);
% for j=26:34
%     for i=1:30
%         x=data2(:,i+30*25);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p26(i,j-25)=p_i(1,2);
%     end
% end
% 
% p27=zeros(30,8);
% for j=27:34
%     for i=1:30
%         x=data2(:,i+30*26);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p27(i,j-26)=p_i(1,2);
%     end
% end
% 
% p28=zeros(30,7);
% for j=28:34
%     for i=1:30
%         x=data2(:,i+30*27);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p28(i,j-27)=p_i(1,2);
%     end
% end
% 
% p29=zeros(30,6);
% for j=29:34
%     for i=1:30
%         x=data2(:,i+30*28);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p29(i,j-28)=p_i(1,2);
%     end
% end
% 
% p30=zeros(30,5);
% for j=30:34
%     for i=1:30
%         x=data2(:,i+30*29);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p30(i,j-29)=p_i(1,2);
%     end
% end
% 
% p31=zeros(30,4);
% for j=31:34
%     for i=1:30
%         x=data2(:,i+30*30);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p31(i,j-30)=p_i(1,2);
%     end
% end
% 
% p32=zeros(30,3);
% for j=32:34
%     for i=1:30
%         x=data2(:,i+30*31);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p32(i,j-31)=p_i(1,2);
%     end
% end
% 
% p33=zeros(30,2);
% for j=33:34
%     for i=1:30
%         x=data2(:,i+30*32);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p33(i,j-32)=p_i(1,2);
%     end
% end
% 
% p34=zeros(30,1);
% for j=34:34
%     for i=1:30
%         x=data2(:,i+30*33);
%         y=data2(:,30*j+i);
%         p_i=corrcoef(x,y); %f14~f15-35;
%         p34(i,j-33)=p_i(1,2);
%     end
% end

