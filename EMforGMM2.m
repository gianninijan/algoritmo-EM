%% ALGORITMO EM PARA GMM %%

clear all;
close all;
clc;


%% SETUP %%
L = 1000;                   % quantidade de amostras
mu1_real = [0 4];           % medias da primeira componente gaussiana    
mu2_real = [-2 0];          % medias da segunda componente gaussiana
w1_real = 0.6;              % peso da primeira gaussiana
w2_real = 0.4;              % peso da segunda gaussiana
cov1_real = [3 0;0 0.5];    % matriz de covariancia da primeira componente gaussiana
cov2_real = [1 0;0 0.5];    % matriz de covariancia da primeira componente gaussiana

% compactando os dados
mu_real = [mu1_real; mu2_real];
w_real = [w1_real w2_real];
cov_real(:,:,1) = cov1_real;
cov_real(:,:,2) = cov2_real;


%% GERANDO AS AMOSTRAS %%
X1 = mvnrnd(mu_real(1,:), cov_real(:,:,1), L*w_real(1));  % gera L*w_real(1) pontos bidimensionais com dist. normal bivariada
X2 = mvnrnd(mu_real(2,:), cov_real(:,:,2), L*w_real(2));  % gera L*w_real(2) pontos bidimensionais com dist. normal bivariada
% mvnrnd(MU,SIGMA,N) - gera mt. [NxD] de vt. aleatorios de uma distribuição normal multivarida, 
% onde: MU [1xN] - vt de médias, SIGMA [DxD] - matrix de covariancia

% agrupando os dados
X = [X1; X2];                       % todos os pontos de dados que pertence aos dois clusters
X = X(randperm(size(X,1)),:);       % EMBARALHANDO os pontos de dados

% Plotando os pontos gerados
subplot(3, 3, 1);
plot(X1(:,1), X1(:,2), 'x', X2(:,1), X2(:,2), 'o');     % plotagem diferenciando os clusters
xlim([min(X(:)) max(X(:))]) 
ylim([min(X(:)) max(X(:))])

% valores de contorno para os valores reais
% gm = gmdistribution(mu_real,cov_real, w_real);            % distribuição de misturas gaussianas.
% subplot(2, 2, 2);
% ezsurf(@(x,y)pdf(gm,[x y]),[-10 10],[-10 10])
% subplot(2, 2, 3);
% ezcontourf(@(x,y)pdf(gm,[x y]),[-10 10],[-10 10]);


%% Palpites iniciais:   %%
vtWeightZero = [0.5 0.5];                                    % vetor de pesos. w = (1/k) = (1/2) = 0.5
mtMuZero = [0.0823 3.9189;-2.0706 -0.2327];                  % vetor de médias. mt(i,:) - média(s) do componente i
mtCovZero(:,:,1) = eye(2);                                   % matriz identidade para a matriz de covariancia do elemento 1
mtCovZero(:,:,2) = eye(2);                                   % matriz identidade para a matriz de covariancia do elemento 2
gmOne = gmdistribution(mtMuZero, mtCovZero, vtWeightZero);   % distribuição de misturas gaussianas.
subplot(3, 3, 2);
ezcontour(@(x,y)pdf(gmOne,[x y]),[-10 10],[-10 10]);         % testar ezcontourf
xlim([min(X(:)) max(X(:))]) 
ylim([min(X(:)) max(X(:))])

subplot(3, 3, 3);
hold on;
plot(X1(:,1), X1(:,2), 'x', X2(:,1), X2(:,2), 'o');     % plotagem diferenciando os clusters
ezcontour(@(x,y)pdf(gmOne,[x y]),[-10 10],[-10 10]);         % testar ezcontourf
xlim([min(X(:)) max(X(:))]) 
ylim([min(X(:)) max(X(:))])
hold off;

%% valores finais encontrados no artigo     %%
subplot(3, 3, 4);
plot(X1(:,1), X1(:,2), 'x', X2(:,1), X2(:,2), 'o');     % plotagem diferenciando os clusters
xlim([min(X(:)) max(X(:))]) 
ylim([min(X(:)) max(X(:))])

vtWeightThree = [0.5945 0.4034];
mtMuThree = [0.0806 3.9445; -2.0181 -0.1740];
mtCovThree(:,:,1) = [2.7452 0.0568; 0.0568 0.4821];
mtCovThree(:,:,2) = [0.8750 -0.0153; -0.0153 1.7935];
gmThree = gmdistribution(mtMuThree, mtCovThree, vtWeightThree);   % distribuição de misturas gaussianas.

subplot(3, 3, 5);
ezcontour(@(x,y)pdf(gmThree,[x y]),[-10 10],[-10 10]);            % testar ezcontourf
xlim([min(X(:)) max(X(:))]) 
ylim([min(X(:)) max(X(:))])

subplot(3, 3, 6);
hold on;
plot(X1(:,1), X1(:,2), 'x', X2(:,1), X2(:,2), 'o');               % plotagem diferenciando os clusters
ezcontour(@(x,y)pdf(gmThree,[x y]),[-10 10],[-10 10]);            % testar ezcontourf
xlim([min(X(:)) max(X(:))]) 
ylim([min(X(:)) max(X(:))])
hold off;


%% Usando uma função do matlab para encontra o GMM para os dados

subplot(3, 3, 7);
plot(X1(:,1), X1(:,2), 'x', X2(:,1), X2(:,2), 'o');
xlim([min(X(:)) max(X(:))]) 
ylim([min(X(:)) max(X(:))])

GMModel = fitgmdist(X,2);
subplot(3, 3, 8);
ezcontour(@(x,y)pdf(GMModel,[x y]),[-8 6],[-8 6]);
xlim([min(X(:)) max(X(:))]) 
ylim([min(X(:)) max(X(:))])

subplot(3, 3, 9);
plot(X1(:,1), X1(:,2), 'x', X2(:,1), X2(:,2), 'o');
hold on;
ezcontour(@(x,y)pdf(GMModel,[x y]),[-8 6],[-8 6]);
xlim([min(X(:)) max(X(:))]) 
ylim([min(X(:)) max(X(:))])
hold off;

% distancia euclidiana entre as médias
dist = norm( mu1_real - mtMuThree(1,:) ) + norm(mu2_real - mtMuThree(2,:));