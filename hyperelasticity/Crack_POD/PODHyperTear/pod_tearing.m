%% Author: Minglang Yin, minglang_yin@brown.edu
% Need permission for code modification. All rights reserve.

clear all
close all
clc

dataset = load("dataset.mat");
snapshot = dataset.snapshot;
snapshot = snapshot(:, :, 2)';
coord = dataset.coord(1, :, :);
coord = squeeze(coord);
coord_init = dataset.coord_init;
[U_POD, S_POD, V_POD] = svd(snapshot,'econ');
S_POD=diag(S_POD);

% t=linspace(0,1,31);

%% plot domain
x = linspace(0, 1, 101);
y = linspace(0, 1, 101);
[xx, yy] = meshgrid(x, y);
xx_reshape = reshape(xx, [], 1);
yy_reshape = reshape(yy, [], 1);

%% notch domain
left_up = [0, 0.531];
right_dw = [0.099, 0.471];
tmp_x = linspace(left_up(1), right_dw(1), 21);
tmp_y = linspace(left_up(2), right_dw(2), 21);
tmp_x_inv = linspace(right_dw(1), left_up(1), 21);
line_1 = cat(1, tmp_x, left_up(2)*ones(size(tmp_x)) )';
line_2 = cat(1, right_dw(1)*ones(size(tmp_y)), tmp_y)';
line_3 = cat(1, tmp_x_inv, right_dw(2)*ones(size(tmp_y)))';

notch_line = cat(1, line_1, line_2, line_3);
[in,on] = inpolygon(xx,yy,notch_line(:, 1),notch_line(:, 2));

%% figure 1: in and out
figure()
plot(notch_line(:, 1),notch_line(:, 2)) % polygon
axis equal
hold on
scatter(xx_reshape(in),yy_reshape(in),'r+') % points inside
scatter(xx_reshape(~in),yy_reshape(~in),'bo') % points outside
hold off

%% figure 2: full modes
figure()
scatter(coord(:, 1),coord(:, 2), [], snapshot(:, end))
% F = scatteredInterpolant(coord(:, 1),coord(:, 2), snapshot(:, end));
% stress_intp = F(xx_reshape, yy_reshape);
% imagesc(x, y, reshape(stress_intp, size(xx)))
% set(gca,'YDir','normal')

%% Plots the mode energies
% ModeEnergies=S_POD.^2;

figure();
% subplot(1, 2, 1)
ModeEnergies=S_POD.^2;
ModeEnergyFraction=ModeEnergies/sum(ModeEnergies);
semilogy(1:length(ModeEnergies),ModeEnergyFraction,'k');
title('Mode Energies sq');
% subplot(1, 2, 2)
% ModeEnergies=S_POD;
% ModeEnergyFraction=ModeEnergies/sum(ModeEnergies);
% semilogy(1:length(ModeEnergies),ModeEnergyFraction,'k');
% title('Mode Energies');

%% Plots the mode shape m for visualization purposes:
m=3;
figure();
for id = 1:9
    subplot(3, 3, id);

    scatter(coord_init(:, 1),coord_init(:, 2), [], U_POD(:, id));
    colorbar('southoutside')
%     F = scatteredInterpolant(coord(:, 1),coord(:, 2), snapshot(:, id));
%     mode_intp = F(xx_reshape, yy_reshape);
%     imagesc(x, y, reshape(stress_intp, size(xx)))
    caxis([min(U_POD(:, id)) max(U_POD(:, id))])
    %colormap redblue
    xlabel('x');
    ylabel('y');
    title(['Mode ' num2str(id)]);
end

%% Plots the time coefficient matrix V for modes 1 and 2, for visualization purposes:
% TimeCoefficients1=V_POD(:,1);
% TimeCoefficients2=V_POD(:,2);
% TimeCoefficients_m=V_POD(:,m); %For mth mode
% fi3=figure('Color','w','Position',[969 102 811 357]);
% plot(t,TimeCoefficients1,'k-'); hold on;
% plot(t,TimeCoefficients2,'b-');
% plot(t,TimeCoefficients_m,'r-');
% legend('Mode 1','Mode 2',['Mode ' num2str(m,'%0.0f')]);
% title('Time Coefficients from POD V matrix')

function [U, S, V] = pod(X)
    dims=size(X);
    X=X-repmat(mean(X,1),[dims(1) ones(1,length(dims)-1)]);
    X=reshape(X,dims(1),prod(dims(2:end)));
    [U, S, V]=svd(X.','econ');
    U=U.';
    dims_min=dims; dims_min(1)=min(dims(1),prod(dims(2:end))); %Redefines dims, if there are more snapshots than sensors. (Required because the 'econ' tag in svd will make U square in this case)
    U=reshape(U,dims_min);
    S=diag(S); %Zero entries in S are pointless.
end