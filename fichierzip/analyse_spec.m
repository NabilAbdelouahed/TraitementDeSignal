% function fenetre = analyse_spec(nom_fenetre,longueur)
% Exemples d'utilisation:
% fenetre = analyse_spec(@hann,512);
% fenetre = analyse_spec(@rectwin,1024);

function fenetre = analyse_spec(nom_fenetre,longueur)

fenetre = window(nom_fenetre,longueur);
