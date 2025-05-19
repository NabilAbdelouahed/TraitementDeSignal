% trace2.m

clear all, close all
Te1 = 0.02;N1 = 100;
x1 = 0:Te1:(N1-1)*Te1;
y1 = sin(2*pi*x1);
Te2 = 0.04;N2 = 50;
x2 = 0:Te2:(N2-1)*Te2;
y2 = sin(2*pi*x2);

figure
plot(x1,y1,'.',x2,y2,'+')
grid on, zoom on
xlabel('t(s)');
ylabel('valeur');
legend('y1','y2')
title('Signaux y1 et y2');

figure
plot(x1,y1,'.-')
title('Signal y1');
hold on
plot(x2,y2,'or')
xlim([0.45 1])
hold off
grid on, zoom on
xlabel('t(s)');
ylabel('valeur');
legend('y1','y2')
title('Signaux y1 et y2');

figure
subplot(2,1,1)
plot(x1,y1,'.')
ylabel('y1');
title('Signaux y1 et y2');
grid on
subplot(2,1,2)
plot(x2,y2,'.')
grid on, zoom on
xlabel('t(s)');
ylabel('y2');
