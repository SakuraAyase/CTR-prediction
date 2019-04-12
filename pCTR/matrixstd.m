function std=matrixstd(t)

[M,N]=size(t);
%%
sum=0.0;
for i=1:M
    for j=i+1:N
        sum=sum+t(i,j);
    end
end
mean=sum/(double(M*N/2.0-M));
sum_err=0.0;
for i=1:M
    for j=i+1:N
        sum_err=sum_err+(t(i,j)-mean).^2;
    end
end
std=sum_err/(double(M*N/2.0-M));
%%
% sum=0.0;
% for i=1:M
%     for j=1:N
%         sum=sum+t(i,j);
%     end
% end
% mean=sum/(double(M*N));
% sum_err=0.0;
% for i=1:M
%     for j=1:N
%         sum_err=sum_err+(t(i,j)-mean).^2;
%     end
% end
% std=sum_err/(double(M*N));