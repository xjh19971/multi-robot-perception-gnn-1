
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


for k = 1:n
    tempQ_NN = [];
    tempQ_NP = [];
    tempQ_PN = [];
    tempQ_PP = [];
    tempQ_ZZ = [];
    
    for i = 5:8
        tempQ_NN = [tempQ_NN,NN(k,i)];
        tempQ_NP = [tempQ_NP,NP(k,i)];
        tempQ_PN = [tempQ_PN,PN(k,i)];
        tempQ_PP = [tempQ_PP,PP(k,i)];
        tempQ_ZZ = [tempQ_ZZ,ZZ(k,i)];
    end
   
    R_NN = quat2rotm(tempQ_NN);
    R_NP = quat2rotm(tempQ_NP);
    R_PN = quat2rotm(tempQ_PN);
    R_PP = quat2rotm(tempQ_PP);
    R_ZZ = quat2rotm(tempQ_ZZ);
    
    
    t_NN = [];
    t_NP = [];
    t_PN = [];
    t_PP = [];
    t_ZZ = [];
    for j = 2:4
        t_NN = [t_NN,NN(k,j)];
        t_NP = [t_NP,NP(k,j)];
        t_PN = [t_PN,PN(k,j)];
        t_PP = [t_PP,PP(k,j)];
        t_ZZ = [t_ZZ,ZZ(k,j)];
    end
    
    pose_NN = rigid3d(R_NN,t_NN);
    cam_NN = plotCamera('AbsolutePose',pose_NN,'Size',0.1,'Color',[0 0.4470 0.7410]);
    
    pose_NP = rigid3d(R_NP,t_NP);
    cam_NP = plotCamera('AbsolutePose',pose_NP,'Size',0.1,'Color',[0.8500 0.3250 0.0980]);
    
    pose_PN = rigid3d(R_PN,t_PN);
    cam_PN = plotCamera('AbsolutePose',pose_PN,'Size',0.1,'Color',[0.92980 0.6940 0.1250]);
    
    pose_PP = rigid3d(R_PP,t_PP);
    cam_PP = plotCamera('AbsolutePose',pose_PP,'Size',0.1,'Color',[0.4660 0.6740 0.1880]);
    
    pose_ZZ = rigid3d(R_ZZ,t_ZZ);
    cam_ZZ = plotCamera('AbsolutePose',pose_ZZ,'Size',0.1,'Color',[0.3010 0.7450 0.9330]);
    
    
    xlim([50 75])
    ylim([-52 -30])
    zlim([2 4])
    
%     pose_NP = rigid3d(R_NP,t_NP);
%     cam_NP = plotCamera('AbsolutePose',pose_NP,'Size',0.1,'Color',[0.8500 0.3250 0.0980]);
%     
%     pose_PN = rigid3d(R_PN,t_PN);
%     cam_PN = plotCamera('AbsolutePose',pose_PN,'Size',0.1,'Color',[0.92980 0.6940 0.1250]);
%     
%     pose_PP = rigid3d(R_PP,t_PP);
%     cam_PP = plotCamera('AbsolutePose',pose_PP,'Size',0.1,'Color',[0.4660 0.6740 0.1880]);
%     
%     pose_ZZ = rigid3d(R_ZZ,t_ZZ);
%     cam_ZZ = plotCamera('AbsolutePose',pose_ZZ,'Size',0.1,'Color',[0.3010 0.7450 0.9330]);
%     
    
%     xlim([20 60])
%     ylim([-35 -15])
%     zlim([0 7])
    xlabel('X-axis in meters')
    ylabel('Y-axis in meters') 
    zlabel('Z-axis in meters')
    hold on;
end

