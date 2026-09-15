
clear;
close all;

%%
green4 = [linspace(0.7, 0, 256)', linspace(0.9, 0.2, 256)', linspace(0.5, 0, 256)'];
figure; hold on;
d_fiber = [0, 0.1, 0.5, 1.0];
for i = 1:4
    yline(d_fiber(i),'color',green4(80*i-70,:),'linewidth',3, 'Alpha', 1); 
end
hold off; box on
xlim([0,1.5]); ylim([-0.1,1.2]);
set(gca,'fontsize',16,'fontname','Arial'); 

%%
T_fiber = [0, 0.2, 0.4, 0.8];

fiber0 = 1;
fiber1 = 0;
fiber_de = fiber0 - fiber1;

speed_de = 5;

t = linspace(-0.1:0.01,1.5);
%blue1 = [linspace(0.85, 0, 256)', linspace(0.9, 0.3, 256)', linspace(0.95, 0.6, 256)', linspace(1, 0.9, 256)'];

%blue3 = [linspace(0.8, 0, 256)', linspace(0.9, 0.1, 256)', linspace(0.95, 0.3, 256)', linspace(1, 0.6, 256)'];
blue3 = [linspace(0.9, 0, 256)', linspace(0.95, 0.3, 256)', linspace(1, 0.6, 256)'];
figure; hold on;

for i = 1:4
    fiber = fiber0-fiber_de.*(exp(-exp(-speed_de.*(t-T_fiber(i))+1)));
    plot(t,fiber,'color',blue3(50*i,:),'linewidth',3);
    xline(T_fiber(i),'--','color',blue3(50*i,:),'linewidth',1.2);
        text(T_fiber(i), 1.1, num2str(round(T_fiber(i),2)), ...
         'color', 'k', 'HorizontalAlignment', 'center','fontsize',14, 'fontweight', 'bold');
end

hold off; box on;
xlim([-0.1,1.5]); ylim([-0.1,1.2]);
set(gca,'fontsize',16,'fontname','Arial'); 

%%
[data,name] = xlsread('Results_all_tlag_0-1_new.xlsx',1);
t = data(:,1);

xgrid = t;
xgrid1 = linspace(0,10);
model_gpz = @(p,x) p(1) + p(2).*exp(-exp(-p(3).*(x - p(4)) + 1));
init = [1 0 5 1];

i = 1;
close all;
figure; hold on;

blue1 = [linspace(0.85, 0, 256)', linspace(0.9, 0.3, 256)', linspace(0.95, 0.6, 256)', linspace(1, 0.9, 256)'];

%colormap(blue1);
    
for j = [14,15,16,18]

    curv = data(:,j);
    
    nlModel = fitnlm(t,curv,model_gpz,init); 
    coeffs = table2array(nlModel.Coefficients); 
    turn(i) = coeffs(4,1);
    p_values = coeffs(4,4);
    
    fitted(:,i) = predict(nlModel,xgrid);
    %fitted(:,i) = predict(nlModel,xgrid1');
    
    %p1 = plot(t,curv,'*');
          
    %line(xgrid,0.015.*exp(fitted(:,i))+1,'color',blue3(50*i,:),'linewidth',2); hold on;
    line(xgrid,2.^(fitted(:,i))./10,'color',blue1(50*i,:),'linewidth',3, ...
        'DisplayName', [' \itT\rm_i_n_i_-_f_i_b_e_r = ' num2str(T_fiber(i))]); 
    %line(xgrid,curv,'color',blue3(50*i,:),'linewidth',2); hold on;
    %line(xgrid1,fitted(:,i),'color',blue3(50*i,:),'linewidth',2); hold on;
    xline(turn(i),'--','color',blue1(50*i,:),'linewidth',1.2, 'HandleVisibility', 'off'); 
    text(turn(i), min(ylim) + 0.05 * i* (max(ylim) - min(ylim))+0.5, num2str(round(turn(i),2)), ...
         'color', 'k', 'HorizontalAlignment', 'center','fontsize',14);
    xlim([0.5,1.5]); box on;
    %ylabel('0.015*exp(curvature)+1')
     set(gca,'fontsize',16,'fontname','Arial');    
    i = i + 1;
end
ylabel('Converted Curvature','fontsize',18,'fontweight','bold')
h = legend('show', 'Location', 'best','fontsize',14);
%set(h, 'fontsize',16)
hold off;
%ylim([0.9,6]);

%%
[data,name] = xlsread('Results_all_rFD_0-1_new.xlsx',1);
t = data(:,1);

close all;
figure; hold on;
%green4 = [linspace(0.6, 0, 256)', linspace(0.7, 0.1, 256)', linspace(0.9, 0.2, 256)', linspace(0.5, 0, 256)'];
i = 1;

for j = [18,19,23,25]

    curv = data(:,j);
   
    %line(t,curv,'color',green4(85*(i-1)+5,:),'linewidth',2); hold on;
    line(t, curv, 'color', green4(80*i-70,:), 'linewidth', 3, 'DisplayName', [' \itd\rm = ' num2str(d_fiber(i))]); 
    
    xlim([0.5,1.5]); box on;
    set(gca,'fontsize',14,'fontname','Arial');    
    i = i + 1;
end
ylabel('Curvature','fontsize',18,'fontweight','bold')
ylim([0.5,7]);
h = legend('show', 'Location', 'best');
set(h, 'fontsize',18)
hold off;
