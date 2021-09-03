clear; clc; close all;
%This example defines the mathematical solution to the piston in an infinite duct problem:
%        ---------=========------------
%<<-inf                                   inf>>
%        ------------------------------
%Walls -> ---------
%Piston -> =========
%Considering a 2D domain subject to the wave equation:
%d2v(x,y,t)/dx2 + d2v(x,y,t)/dy2 = d2v(x,y,t)/dt2
%under B.C.s 
%v(y=0)=0;
%v(y=H)=0, if |x|<L; and epsilon*sin(omega*t) if |x|<L
%This computes the Fourier modes of the solution for a specific case:
H=10;
omega=1;
L=3;
t=linspace(0,10*pi,100);
SNR=0.5; %Signal-to-noise-ratio for adding noise to the solution (greater values mean better signal quality)
x=linspace(-10*L,10*L,1000);
y=linspace(0,H,200);
[X,Y]=meshgrid(x,y);
%%
V=zeros(size(X));
for n=1:10
    k0=sqrt(omega.^2-(n*pi/H).^2);
    
    An=(2*1i*pi^2/(H^2))*(n*((-1)^n))*sin(n*pi*Y/H)/(k0^2);
    
    %Combined Solution 
    Iout=sign(X).*An.*2.*1i.*exp(1i*abs(X)*k0).*sin(k0.*L);
    Iout(abs(X)<L)=0;
    
    Iin=2*1i*An.*exp(1i*L*k0).*sin(k0.*X);
    Iin(abs(X)>L)=0;
    
    V=V+Iout+Iin;   
end
%Now puts this solution inside a 3D matrix for POD computation.
%Here we'll also add speckle noise to make it more interesting.
Vfield_Snapshots=zeros(length(t),size(V,1),size(V,2));
for i=1:length(t)    
    Noise=rand(size(V))>0.7; %Speckle noise
    Vfield_Snapshots(i,:,:)=imag(exp(-1i*omega*t(i))*V) + Noise*max(imag(V(:)))/SNR;
end
%Plots the complete solution animation for visualization purposes
fi=figure('Color','w','Position',[965 620 811 357]);
for i=1:length(t)  
    imagesc(x/L,y/H,squeeze(Vfield_Snapshots(i,:,:)));
    caxis([-max(abs(V(:))) max(abs(V(:)))])
    %colormap redblue
    xlabel('x/L');
    ylabel('y/H');
    set(gca,'ydir','normal')
    title('Complete solution, 10 Fourier modes');
    drawnow;
    pause(0.01)    
end
close all;
%%
%Now uses this solution to compute the POD with the wrapper:
[U_POD, S_POD, V_POD] = pod(Vfield_Snapshots);
%%
%Plots the mode energies
ModeEnergies=S_POD.^2;
ModeEnergyFraction=ModeEnergies/sum(ModeEnergies);
figure('Color','w','Position',[146 620 403 357]);
bar(1:length(ModeEnergies),ModeEnergyFraction,'k');
title('Mode Energies');
%Note only the first two modes have significant energy. The noise is spread out through all modes.
%%
%Plots the mode shape m for visualization purposes:
m=3;
fi2=figure('Color','w','Position',[965 620 811 357]);
modeShape=squeeze(U_POD(m,:,:));
imagesc(x/L,y/H,modeShape);
caxis([-max(abs(modeShape(:))) max(abs(modeShape(:)))])
%colormap redblue
xlabel('x/L');
ylabel('y/H');
set(gca,'ydir','normal')
title(['Mode Shape ' num2str(m,'%0.0f')]);
%Note Modes 1 and 2 are a wave pair. This is because POD is a real-valued function.
%So this means that, in order to produce a propagating wave, a pair of two phase-shifted modes are required,
%similar to a sine wave - cosine wave pair.
%Since the wave equation has a linear dispersion relationship (i.e., wave speed is constant), then only modes
%1 and 2 are relevant to this problem, and they encapsulate the information of all Fourier modes.
%In fluid dynamics, where there are dispersive waves, multiple modes will appear in the modal decomposition.
%%
%Plots the time coefficient matrix V for modes 1 and 2, for visualization purposes:
TimeCoefficients1=V_POD(:,1);
TimeCoefficients2=V_POD(:,2);
TimeCoefficients_m=V_POD(:,m); %For mth mode
fi3=figure('Color','w','Position',[969 102 811 357]);
plot(t,TimeCoefficients1,'k-'); hold on;
plot(t,TimeCoefficients2,'b-');
plot(t,TimeCoefficients_m,'r-');
legend('Mode 1','Mode 2',['Mode ' num2str(m,'%0.0f')]);
title('Time Coefficients from POD V matrix')
%Note the V matrix for modes 1 and 2 is also a pair of sine waves; confirming the traveling wave behavior of the modes.
%**However, if one did not have time-resolved snapshots (try that by shuffling the original Vfield_Snapshots matrix)
%then the U matrix will still pick up the mode shapes, though the V matrix will have little physical significance.



function [U, S, V] = pod(X)
%POD - Proper Orthogonal Decomposition
%   By Fernando Zigunov, 2019
%   [U, S, V] = POD(X) returns the proper orthogonal decomposition
%   of the data matrix X whose first dimension is time/snapshots. X can have any
%   number of additional spatial dimensions or variable indices.
%POD has several names (POD, SVD, PCA, ...) and is a mathematical tool that
%highlights the principal modes of a random series of variables (i.e.,
%financial measurements, fluid flow fields, structural vibrations, neuron
%firing patterns, video frames, and basically anything imaginable!)
%The POD tool is extremely useful in data analysis to identify the most
%energetic modes of a complex system by simply performing measurements when
%its operating. (But if you're already here you already know all of this.)
%This function is a nice wrapper for Matlab's built-in SVD function, which
%only operates in 2D data sets. The wrapper just reshapes the matrices
%before and after applying SVD such that the input series X can be
%n-dimensional and ensures the output modes U are also n-dimensional.
%============Inputs:=============
%X - Time series or snapshots to be analyzed. Can be a n-dimensional matrix
%(n>=2). First dimension is time/snapshot, and all other dimensions are
%conserved for output.
%============Outputs:=============
%U - Mode matrix, has the same size of X. First dimension of X was
%"time/snapshot", therefore first dimension of U will be "mode". Modes are
%automatically organized from most energetic U(1,:,:...) to least
%energetic.
%S - Mode eigenvalues. S.^2 gives the mode energies. S is a diagonal
%matrix, so returns just the vector S to save one line of post-processing.
%V - Mode time-series or snapshot coefficients - V contains the
%contribution of each mode to a specific snapshot of X. U*S*V gives back X
%(given U and X are reshaped into 2D matrices again).
dims=size(X);
%Removes mean. Note: Not removing the mean biases the modes as the
%data points centroid is shifted. If one wants to capture only the
%oscillations around the mean, the mean MUST be removed.
X=X-repmat(mean(X,1),[dims(1) ones(1,length(dims)-1)]);
%Reshapes X
X=reshape(X,dims(1),prod(dims(2:end)));
%Performs SVD
[U, S, V]=svd(X.','econ');
%Reshapes U back
U=U.';
dims_min=dims; dims_min(1)=min(dims(1),prod(dims(2:end))); %Redefines dims, if there are more snapshots than sensors. (Required because the 'econ' tag in svd will make U square in this case)
U=reshape(U,dims_min);
S=diag(S); %Zero entries in S are pointless.
end