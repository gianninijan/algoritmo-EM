%% ALGORITMO EM PARA LANÇAMENTO DE MOEDA %%
% O experimento consiste na escolha entre duas moedas A e B aleatoriamente, 
% sem saber sua identidade realizamos 10 lançamentos. Qual a probabilidade 
% de que a moeda escolhida seja A ou B para esse conjunto de resultados. 
% Repetimos o mesmo experiento e vezes.

clear all;
clc;
close all;


%% PARAMETROS
L = 5;                   % numero de escolhas de moeda
N = 10;                  % numero de lançamento para cada escolha da moeda 
bias = [0.6 0.5];        % vetor de bias anterior para cada moeda, i.e, prob. de sair cara (head) para cada uma das moedas " PARAMETRO DESCONHECIDO"
Z = randi([0 1], 1 ,L);  % vetor de identidade de moeda aleatorio. (0 -> Moeda A) e (1 -> Moeda B) "DADOS AUSENTES"  
R = randi([0 1], L, N);  % matriz com todos os resultados do experimento, onde: (0 -> Cara) e (1 -> Coroa) "DADOS OBSERVADOS"
%R = [0 1 1 1 0 0 1 0 1 0; 0 0 0 0 0 1 0 0 0 0; 0 1 0 0 0 0 0 1 0 0; 0 1 0 1 1 1 0 0 1 1; 1 0 0 0 1 0 0 0 1 0]; % matriz de teste tiradas do artigo


%%  ALGORITMO EM %%
nbias = ones(1, length(bias));      % novo vetor de bias calculado pelo algoritmo
iteracoes = 0;                      % variavel de iteração que será incrementada a cada laço           
max_iteracoes = 1e6;                % numero maximo de iterações
tolerancia = 1e-6;                  % critério de parada
mtBias = [bias'];                   % matriz de bias

while (iteracoes < max_iteracoes) 
    
    moedaA = [];    
    moedaB = [];
    
    % laço percorrendo as linhas da matrix
    for linha = 1:L
        heads = length(find(~R(linha,:)));          % numero de caras ou 0's na linha
        tails = length(find(R(linha,:)));           % numero de coroas ou 1's na linha 
        
        % estimador de verossimilhança binominal
        LikehoodA = binopdf(heads, N, bias(1));
        LikehoodB = binopdf(heads, N, bias(2));
        
        % conjunto de pesos - p(z/y, theta_n)
        Prob_A = LikehoodA/(LikehoodA + LikehoodB);
        Prob_B = LikehoodB/(LikehoodA + LikehoodB);
                
        % numero esperado de caras e coroas para cada tipo de moeda -> 
        % -> nº heads e tails = p(y, z/ theta)
        MoedaA_Cara = heads*Prob_A; 
        MoedaA_Coroa = tails*Prob_A;
        
        moedaA = [moedaA, [MoedaA_Cara; MoedaA_Coroa]];
        
        MoedaB_Cara = heads*Prob_B;  
        MoedaB_Coroa = tails*Prob_B;
                
        moedaB = [moedaB, [MoedaB_Cara; MoedaB_Coroa]];
    
    end
    
    aux1 = sum(moedaA,2); % medias de caras para todo o experimento da moeda A
    aux2 = sum(moedaB,2); % medias das coroas para todo o experimento da moeda B
    
    
    % calculando o novo bias (STEP-M)
    nbias = [aux1(1)/sum(aux1), aux2(1)/sum(aux2)];
    mtBias = [mtBias, nbias'];
    
    
    % se a distância euclidiana entre o bias novo e o antigo for menor que a tolerancia
    if sqrt(sum((nbias - bias) .^ 2)) <=  tolerancia
        display('break')
        break;                          % sai do laço while 
    else
       iteracoes = iteracoes + 1;         
       bias = nbias;                    % bias da iteração (i+1) sera igual o nbias calculado na iteração (i) 
     end
     
end

 subplot(2,1,1)
 plot(mtBias(1,:), 'LineWidth', 2.0);
 legend('\theta_{a}: Probabilidade de sair cara com a moeda A'); 
 xlabel('iterações');                               % titulo do eixo-x
 ylabel('\theta_{a}');                              % titulo do eixo-y
 title('Algoritmo EM calculado para Moeda A')       % Titulo do grafico
 
 
 subplot(2,1,2)
 plot(mtBias(2,:), 'r', 'LineWidth', 2.0);
 legend('\theta_{b}: Probabilidade de sair cara com a moeda B');
 xlabel('iterações');                               % titulo do eixo-x
 ylabel('\theta_{B}');                              % titulo do eixo-y
 title('Algoritmo EM calculado para Moeda B')       % Titulo do grafico

