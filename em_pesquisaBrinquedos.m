%% ALGORITMO EM PARA A PESQUISA DE PREFERENCIA ENTRE 4 BRINQUEDOS %%
% Y = [Y1 Y2 Y3 Y4] ~ multi(n, p(theta)), onde: theta � desconhecido e
% usamos o algoritmo EM para calcul�-lo.

clear all;
clc;
close all;


%%  PARAMETROS  %%
N = 1e6;                        % numero de crian�as que respondera a pergunta
RES = randi([1 4], 1, N);       % resultado das pesquisas.
theta = 0.5;                    % valor do theta inicial
iteracoes = 0;                  % variavel de itera��o que ser� incrementada a cada la�o           
max_iteracoes = 1e6;            % numero maximo de itera��es
tolerancia = 1e-7;              % crit�rio de parada

% y1 = length(find(RES == 1));    % numero de crina�as que escolheram o brinquedo 1
% y2 = length(find(RES == 2));    % numero de crina�as que escolheram o brinquedo 2
% y3 = length(find(RES == 3));    % numero de crina�as que escolheram o brinquedo 3
% y4 = length(find(RES == 4));    % numero de crina�as que escolheram o brinquedo 4
y1 = 125;                       % y1 do artigo original    
y2 = 18;                        % y2 do artigo original
y3 = 20;                        % y3 do artigo origal
y4 = 34;                        % y4 do artigo original
Y = [y1 y2 y3 y4];              % histograma das escolhas, i.e, numero de crian�as que escolheram cada brinquedo (DADOS OBSERVADOS)

vtTheta = [theta];              % vetores de theta


%% ALGORITMO %%
while (iteracoes < max_iteracoes) 

    % calculo do novo theta
    u = theta/(2+theta);
    novo_theta = (u*y1 + y4)/((u*y1)+y2+y3+y4);
    
    vtTheta = [vtTheta, novo_theta];
    
    % se a dist�ncia euclidiana entre o bias novo e o antigo for menor que a tolerancia
    if sqrt(sum((novo_theta - theta) .^ 2)) <=  tolerancia
        display('break')
        display(iteracoes)
        break;                          % sai do la�o while 
    else
       iteracoes = iteracoes + 1;         
       theta = novo_theta;              % theta da itera��o (i+1) sera igual o novo_theta calculado na itera��o (i) 
     end
    
end

% grafico: Thetas vs Itera��es
plot(vtTheta)



