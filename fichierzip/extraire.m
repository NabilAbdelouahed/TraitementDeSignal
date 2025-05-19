% extraire.m

clear all, close all
[x,fe] = audioread('PAROLE.wav',[121800 147500]);
figure
plot(x) % c'et une des rares utilisations de plot sans préciser d'abscisses
grid on, zoom on
xlabel('Numéro de l''échantillon')
ylabel('Valeur')
title('Extrait du signal de parole')
n1 = 13800;
n2 = 20000;
mot1 = x(1:n1);
mot2 = x(n1+1:n2);
mot3 = x(n2+1:length(x));
% Vous pouvez écouter le signal initial puis les signaux scindés
% en utilisant la fonction sound
