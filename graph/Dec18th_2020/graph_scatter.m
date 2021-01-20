
close all


FilesNN = dir('/Users/zhousq/Documents/Research/GNN/data/pose/4/pose/DroneNN_main/*.txt');
FilesNP = dir('/Users/zhousq/Documents/Research/GNN/data/pose/4/pose/DroneNP_main/*.txt');
FilesPN = dir('/Users/zhousq/Documents/Research/GNN/data/pose/4/pose/DronePN_main/*.txt');
FilesPP = dir('/Users/zhousq/Documents/Research/GNN/data/pose/4/pose/DronePP_main/*.txt');
FilesZZ = dir('/Users/zhousq/Documents/Research/GNN/data/pose/4/pose/DroneZZ_main/*.txt');
n = length(FilesNN);



%initilizing all drones
NN = zeros(n,8); %each camera pose has 8 rows and n columns, this line is for initialization
NP = zeros(n,8);
PN = zeros(n,8);
PP = zeros(n,8);
ZZ = zeros(n,8);


for k = 1:n
    Filesname = ['/Users/zhousq/Documents/Research/GNN/data/pose/4/pose/DroneNN_main/', FilesNN(k).name];
    NN(k,:) = importdata(Filesname);
      

    Filesname = ['/Users/zhousq/Documents/Research/GNN/data/pose/4/pose/DroneNP_main/', FilesNP(k).name];
    full_data2{k} = importdata(Filesname);
    NP(k,:) = full_data2{k};
    
    
    Filesname = ['/Users/zhousq/Documents/Research/GNN/data/pose/4/pose/DronePN_main/', FilesPN(k).name];
    full_data3{k} = importdata(Filesname);
    PN(k,:) = full_data3{k};
    
    Filesname = ['/Users/zhousq/Documents/Research/GNN/data/pose/4/pose/DronePP_main/', FilesPP(k).name];
    full_data4{k} = importdata(Filesname);
    PP(k,:) = full_data4{k};
    
    Filesname = ['/Users/zhousq/Documents/Research/GNN/data/pose/4/pose/DroneZZ_main/', FilesZZ(k).name];
    full_data5{k} = importdata(Filesname);
    ZZ(k,:) = full_data5{k};
end




%https://www.mathworks.com/matlabcentral/answers/342856-how-do-i-specify-the-size-of-the-circles-in-the-scatter-plot-so-that-the-length-of-their-diameters-i
%https://www.mathworks.com/matlabcentral/answers/166856-creating-a-scatter-plot-with-smooth-lines-and-markers
%figure 99
scatter3(PN(:,2), PN(:,3), PN(:,4),[],[0.92920, 0.6940, 0.1250],'filled'); 
hold on
line(PN(:,2), PN(:,3), PN(:,4))
hold off

%scatter3(PN(:,2), PN(:,3), PN(:,4),'b.');

% for k = 1:n
%     plot3(PN(k,2), PN(k,3),PN(k,4),'b.');
%     %'Color',[0.92920, 0.6940, 0.1250]
%     grid on
%     hold on
% end

    
xlabel('X-axis in meters')
ylabel('Y-axis in meters') 
zlabel('Z-axis in meters')

hold off
