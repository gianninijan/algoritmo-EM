%% ALGORITMO EM PARA GMM %%

% Exemplo 3.2.4 - Theory and Use of the EM Algorithm

clear all;
close all;
clc;

% K = 2 (numero de componentes gaussianas)
% N = 2 (cada gaussiana é bivariada)
% L = 1000 (numero de amostras observadas, onde cada amostra pertence ao R^2) 


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

% agrupando os dados
X = [X1; X2];                       % todos os pontos de dados que pertence aos dois clusters
X = X(randperm(size(X,1)),:);       % EMBARALHANDO os pontos de dados

figure;

% Plotando os pontos gerados
subplot(2, 2, 1);
plot(X1(:,1), X1(:,2), 'x', X2(:,1), X2(:,2), 'o');     % plotagem diferenciando os clusters


%% GMM PARA OS DADOS REAIS %%

eixo_x = [-10:0.5:10];          
eixo_y = [-10:0.5:10];          
[x y]=meshgrid(eixo_x, eixo_y);     % criação da malha do grafico tridimensional  
mesh = [x(:) y(:)];                 % mesh é uma matrix em que cada linha representa um ponto (x, y) da malha

Z_real = w_real(1)*mvnpdf(mesh, mu_real(1,:), cov_real(:,:,1)) + ...
    w_real(2)*mvnpdf(mesh, mu_real(2,:), cov_real(:,:,2));       % calculando a GMM para os dados reais

subplot(2, 2, 2);
contour(x, y, reshape(Z_real,size(x,2),size(y,2)));          % plotando as curvas de nível da GMM

subplot(2, 2, 3);
surf(x, y, reshape(Z_real,size(x,2),size(y,2)));              % plotando o grafico tridimensional da GMM

subplot(2, 2, 4);
plot(X(:,1),X(:,2),'*');
hold on;
contour(x, y, reshape(Z_real,size(x,2),size(y,2)));
hold off;


%% PARAMETROS DE INICIALIZAÇÃO %%
w = (1/2)*ones(1, size(X, 2));  % coeficientes da mistura (pesos) 
cov(:,:,1) = eye(size(X, 2));   % matriz de covariancia inicial é a matriz de identidade
cov(:,:,2) = eye(size(X, 2));
mu = [0.0823 3.9189; -2.0706 -0.2327];  % médias iniciais retirado do artigo base (pag. 254)
limiar = 1e-3;  

% coef. de responsabilidade - P(Zi = j / xi)
gama = zeros(size(X,1), length(w));


%% ALGORITMO EM P/ CALCULAR OS PARAMETROS DA GMMM

iter = 100;  % numero de iterações do ALGORITMO

for ii = 1:iter

    % PASSO-E: CALCULO DA RESPONSABILIDADE DE CADA AMOSTRA PARA CADA CLUSTER
    for jj = 1:length(w)
        gama(:,jj) = w(jj)*mvnpdf(X, mu(jj,:), cov(:,:,jj));  % calcula o numerador da formula da responsabilidade
    end
        
    gama=gama./repmat(sum(gama,2), 1, size(gama,2));    % calcula a formula completa da responsabilidade p/ cada amostra
    
    % PASSO-M: CALCULAR OS PARAMETROS DA GMM COM BASE NA RESPONSABILIDADE DO PASSO-E
    w = sum(gama,1)./size(gama,1);      % novos pesos da GMM
    
    % calculo das novas médias da GMM
    mu_ = gama'*X;                                  % calculo do numerador da formula da média
    mu_ = mu_./repmat((sum(gama,1))',1,size(mu_,2));  % calculo final da media
    
    % calculo das novas matriz de covariancia da GMM
    for jj = 1:length(w)
         vari = repmat(gama(:,jj),1,size(X,2)).*(X - repmat(mu_(jj,:),size(X,1),1));
         cov(:,:,jj) = (vari'*vari)/sum(gama(:,jj),1);
    end
    
    dist = norm(mu_(1,:) - mu(1,:)) + norm(mu_(2,:) - mu(2,:));
    
    if dist <= limiar
        disp(ii)
        break;
    end
   
    mu = mu_;
    
end


%% PLOTANDO A ESTIMAÇÃO %%
figure;

% Plotando os pontos gerados
subplot(2, 2, 1);
plot(X1(:,1), X1(:,2), 'x', X2(:,1), X2(:,2), 'o');     % plotagem diferenciando os clusters

% calculando o novo GMM
z = w(1)*mvnpdf(mesh,mu(1,:),cov(:,:,1))+...
        w(2)* mvnpdf(mesh,mu(2,:),cov(:,:,2));

subplot(2, 2, 2);
contour(x,y,reshape(z,size(x,2),size(y,2)));

subplot(2, 2, 3);
surf(x, y, reshape(z,size(x,2),size(y,2)));              % plotando o grafico tridimensional da GMM


subplot(2, 2, 4);
plot(X(:,1),X(:,2),'*');
hold on;
contour(x, y, reshape(z,size(x,2),size(y,2)));
hold off;
