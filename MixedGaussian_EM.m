% EM method to solve a mixed (2D) Gaussian problem.
close all;

global M K
M = 200;
K = 3;

% ThetaProb = zeros(1,K+1);
% ThetaProb(1) = 0; ThetaProb(end)=1;
% ThetaProb(2:end-1) = sort(rand(1,K-1));
ThetaProb = linspace(0, 1, 4);

Mu = rand(K,2)*12;
Sigma = rand(K,2)*5;

x = zeros(100,2);

figure; hold on;

for i = 1:M
    rnd = rand;
    for j = 1:K
        if rnd>=ThetaProb(j) && rnd<ThetaProb(j+1)
            x(i,1) = normrnd(Mu(j,1), sqrt(Sigma(j,1)));
            x(i,2) = normrnd(Mu(j,2), sqrt(Sigma(j,2)));
            plot(x(i,1), x(i,2), '*', ...
                'MarkerFaceColor', [0+1/K*j, 1-1/K*j, 0], ...
                'MarkerEdgeColor', [0+1/K*j, 1-1/K*j, 0]);
            break;
        end
    end
end


% Target: maximize the likelihood function given x.
% Latent variable: Random variable category of each x(i,:).

% w(i,j): possibility that x(i,:) follows the j-th distribution.
w = zeros(M, K);
w_next = ones(M, K)/K;

% Note: start out as random numbers so they get different numbers.
Mu_est = rand(K,2);
Sigma_est = rand(K,2);
Phi_est = rand(K,1);
Phi_est = Phi_est/K;

Optimality = 1E-3;

cnt = 0;
while cnt<100
    % E-step
    for i=1:M
        for j=1:K
            w_next(i,j) = normpdf(x(i,1), Mu_est(j,1), sqrt(Sigma_est(j,1)))* ...
                normpdf(x(i,2), Mu_est(j,2), sqrt(Sigma_est(j,2)));
        end
        w_next(i,:) = w_next(i,:)/sum(w_next(i,:));
    end
    
    if sum(sum((w-w_next).^2))<Optimality
        break;
    end
    w = w_next;
    
    % M-step
    for j = 1:K
        Phi_est(j) = sum(w(:,j))/M;
        Mu_est(j,1) = x(:,1)'*w(:,j)/sum(w(:,j));
        Mu_est(j,2) = x(:,2)'*w(:,j)/sum(w(:,j));
        Sigma_est(j,1) = sum(w(:,j)'.*(x(:,1)'-Mu_est(j,1)').*(x(:,1)'-Mu_est(j,1)'))/sum(w(:,j));
        Sigma_est(j,2) = sum(w(:,j)'.*(x(:,2)'-Mu_est(j,2)').*(x(:,2)'-Mu_est(j,2)'))/sum(w(:,j));
    end
    
    cnt = cnt+1;
end

figure; hold on;
for i = 1:M
    [m, j] = max(w(i,:));
    plot(x(i,1), x(i,2), '*', ...
            'MarkerFaceColor', [0+1/K*j, 1-1/K*j, 0], ...
            'MarkerEdgeColor', [0+1/K*j, 1-1/K*j, 0]);
end

disp(Mu);
disp(Sigma);
disp(ThetaProb);
disp(Mu_est);
disp(Sigma_est);
disp(Phi_est);