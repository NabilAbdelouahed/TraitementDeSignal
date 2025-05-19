% acquérir un signal de bruit

clear all;
close all;

fe=11000;
taille=fe;             %%% 1 seconde de son
xxx=randn(1,taille);   %%% bruit blanc gaussien
xxx=xxx/max(abs(xxx)); %%% attention à la saturation !!!

audiowrite('monbruit.wav',xxx,fe);

