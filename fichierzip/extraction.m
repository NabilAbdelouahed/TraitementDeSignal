% function [res1,res2] = extraction(sig,coupure)
% 
% Scinde le signal sig en deux signaux res1 et res2 à l'indice coupure
% Entrées:
%   sig: signal d'origine
%   coupure: indice de coupure
% Sorties:
%   res1 = sig(1:coupure)
%   res2 = sig(coupure+1:end)

function [res1,res2] = extraction(sig,coupure)

res1 = sig(1:coupure);
res2 = sig(coupure+1:end);
