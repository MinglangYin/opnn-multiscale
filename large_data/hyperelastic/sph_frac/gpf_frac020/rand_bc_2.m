clear all
close all
clc

data = load("rand_disp.dat");
data_size = size(data);
nmodes = 5;
x = data(:, 1);
x(end+1) = 1.5;
disp_bc = string.empty;
fid = fopen('rand_disp_exp.txt','w');
for i = 1:data_size(2)-1
    dat = data(:, i+1);
    dat(end+1) = dat(1);
    [fourier_int, T] = read_inflow(x, dat, nmodes);
    disp_bc(i) = fourier_int;
    
    fprintf(fid,'%s\n', fourier_int);
end

fclose(fid)

function [fourier_int, T] = read_inflow(x, dat,nmodes)

%% Scaling factors
t  = x;
Q  = dat;
fit_Q = fit(t,Q,'cubicinterp');
t = linspace(0,t(end),301);
Q = feval(fit_Q,t);
T = t(end);

%%
n = length(t)-1; 
m = n/2; 
L = T;

tt = t(1:n)';
QQ = Q(1:n)';

N = n; 
mN = N/2; 
c = fft(QQ,N);

a = 2*real(c(1:mN+1))/N; 
b = -2*imag(c(1:mN+1))/N;

aa = a';
bb = b';

t_int = 0:0.001:T;
Q_int = 0.5*a(1)*ones(1,length(t_int));

str = cell(1,nmodes+1);

str{1} = sprintf('%.4f',Q_int(1));

aaa = sort(a,'descend');
bbb = sort(b,'descend');

for j = 1 : nmodes 
   Q_int = Q_int + a(1+j)*cos(2*pi*j*t_int/L) + b(1+j)*sin(2*pi*j*t_int/L);
   str{j+1} = sprintf('+%.4f*cos(%.4f*x)+%.4f*sin(%.4f*x)',a(1+j),2*pi*j/L,b(1+j),2*pi*j/L);   
end

for i = 1:nmodes
    str{1} = strcat(str{1},str{i+1});
end

fourier_int = str{1};

g = eval(strcat(fourier_int));

% 
% plot(t,Q,'g'); 
% hold on;
% plot(t,g,'*');
end
