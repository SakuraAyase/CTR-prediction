function R = Mappingfuc(key, value)

% MapF = containers.Map(key, value);
MapR = containers.Map(value, key);

load('Ava_Base.mat');
Base=ava_base;
R=zeros(22,22);

for i=1:22
   for j=1:22
       R(i,j)=Base(MapR(i),MapR(j));
   end
end