% trace1.m

Te = 0.1;
N = 200;
x = 0:Te:(N-1)*Te;  % création d'un vecteur x = [0 Te 2*Te ... (N-1)*Te]
y = sin(x);

figure
plot(x,y,'+')
grid on, zoom on
xlabel('t(s)');
ylabel('valeur');
title('Signal y = sin(t)');
