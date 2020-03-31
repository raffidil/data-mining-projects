clc;close all;clear;
for epoch=1:1
    %train    
    output=[];
    X=[];
    trainList=[];
    for num=1:40
        randomlist=[];
        for j=1:5
            s1='./att_faces/s';
            s2='.pgm';
            pic_num= randi([1,10]);
            %pic_num=j;
            %genarate 5 nonrepeatedly number
            randomlist=[randomlist pic_num]
            while size(find(randomlist==pic_num))~=0
                pic_num= randi([1,10]);
                %pic_num=j;
            end
            randomlist=[randomlist pic_num];            
            trainList=[trainList pic_num];
            s=strcat(s1,num2str(num),'/',num2str(pic_num),s2);
            f1=imread(s);
            f1=imresize(f1, [32 32]);
            X1=[];
            %1024
            for i=1:32
                X1=[X1,f1(i,:)];
            end
            X=[X;X1];
            %200           
        end
    end   
    %1024*200
    X=X';
    %###################################Insert your code here
    
end
