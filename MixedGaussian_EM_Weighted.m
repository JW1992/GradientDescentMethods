% EM method to solve a mixed (2D), weighted Gaussian problem.
close all;

global M K
M = 200;
K = 2;

% ThetaProb: "total weight" of each Gaussian distribution
ThetaProb = [0.6, 0.5];
ThetaProb = ThetaProb/sum(ThetaProb);

% Mu = rand(K,2)*7;
% Sigma = 2+rand(K,2)*4;
Mu = [
        -2, -2;
        3, 9;
     ];
Sigma = [
            7, 2;
            3, 9;
        ];

global SAMPLE_SPACE SAMPLE_CORRD_X SAMPLE_CORRD_Y
SAMPLE_SPACE = [-5,15];
[SAMPLE_CORRD_X, SAMPLE_CORRD_Y] = meshgrid([SAMPLE_SPACE(1):SAMPLE_SPACE(2)], ...
    [SAMPLE_SPACE(1):SAMPLE_SPACE(2)]);
x = zeros(SAMPLE_SPACE(2)-SAMPLE_SPACE(1)+1, SAMPLE_SPACE(2)-SAMPLE_SPACE(1)+1);
weights = rand(M, 1);

global MAGIC_NUM
MAGIC_NUM = 1E3;

figure; hold on;
for i = 1:SAMPLE_SPACE(2)-SAMPLE_SPACE(1)+1
    for j = 1:SAMPLE_SPACE(2)-SAMPLE_SPACE(1)+1
        for k=1:K
            curSample = ThetaProb(k)* ...
                normpdf(i+SAMPLE_SPACE(1)-1, Mu(k,2), sqrt(Sigma(k,2)))* ...
                normpdf(j+SAMPLE_SPACE(1)-1, Mu(k,1), sqrt(Sigma(k,1)));
            curSample = max(curSample, 1E-8);
            x(i,j) = x(i,j)+curSample;
            plot(j+SAMPLE_SPACE(1)-1, i+SAMPLE_SPACE(1)-1, '.', ...
                'MarkerFaceColor', [0+1/K*k, 1-1/K*k, 0], ...
                'MarkerEdgeColor', [0+1/K*k, 1-1/K*k, 0], ...
                'MarkerSize', MAGIC_NUM*curSample ...
            );
        end
    end
end

figure; hold on;
imagesc(SAMPLE_SPACE(1):SAMPLE_SPACE(2), ...
    SAMPLE_SPACE(1):SAMPLE_SPACE(2), ...
    MAGIC_NUM*x);
% set(gca,'YDir','normal');
FWHMRatio =2.35;
for k=1:K
    DrawEllipse(Mu(k,1),Mu(k,2), ...
        FWHMRatio*sqrt(Sigma(k,1)),FWHMRatio*sqrt(Sigma(k,2)));
end
% Target: maximize the likelihood function given x.
% Latent variable: Random variable category of each x(i,:).

% w(i,j): possibility that x(i,:) follows the j-th distribution.
w = zeros(size(x,1), size(x,2), K);
w_next = ones(size(x,1), size(x,2), K)/K;

% Note: start out as random numbers so they get different numbers.
Mu_est = rand(K,2);
Sigma_est = rand(K,2);
Phi_est = rand(K,1);
Phi_est = Phi_est/sum(Phi_est);

Optimality = 1E-2;

% i: Y axis direction
% j: X axis direction

cnt = 0;
while cnt<30
    % E-step
    for i=1:1:SAMPLE_SPACE(2)-SAMPLE_SPACE(1)+1
        for j=1:1:SAMPLE_SPACE(2)-SAMPLE_SPACE(1)+1
            for k = 1:K
                w_next(i,j,k) = Phi_est(k)* ...
                    normpdf(i+SAMPLE_SPACE(1)-1, Mu_est(k,2), sqrt(Sigma_est(k,2)))* ...
                    normpdf(j+SAMPLE_SPACE(1)-1, Mu_est(k,1), sqrt(Sigma_est(k,1)));
            end
            w_next(i,j,:) = w_next(i,j,:)/sum(squeeze(w_next(i,j,:)));
        end
    end
    
    if sum(sum(sum((w-w_next).^2)))<Optimality
        break;
    end
    w = w_next;
    
    % M-step
    for k = 1:K
        Phi_est(k) = sum(sum(x(:,:).*w(:,:,k)))/(SAMPLE_SPACE(2)-SAMPLE_SPACE(1)+1)^2;
        Mu_est(k,1) = sum(sum(SAMPLE_CORRD_X.*x(:,:).*w(:,:,k)))/ ...
            sum(sum(x(:,:).*w(:,:,k)));
        Mu_est(k,2) = sum(sum(SAMPLE_CORRD_Y.*x(:,:).*w(:,:,k)))/ ...
            sum(sum(x(:,:).*w(:,:,k)));
        Sigma_est(k,1) = sum(sum(x(:,:).*w(:,:,k).*(SAMPLE_CORRD_X-Mu_est(k,1)).* ...
            (SAMPLE_CORRD_X-Mu_est(k,1))))/sum(sum(x(:,:).*w(:,:,k)));
        Sigma_est(k,2) = sum(sum(x(:,:).*w(:,:,k).*(SAMPLE_CORRD_Y-Mu_est(k,2)).* ...
            (SAMPLE_CORRD_Y-Mu_est(k,2))))/sum(sum(x(:,:).*w(:,:,k)));
    end
    Phi_est = Phi_est/sum(Phi_est);
    
    %%%%%%%%%%%%%%%%%%%%
%     close all;
%     figure; imagesc(SAMPLE_SPACE(1):SAMPLE_SPACE(2), ...
%         SAMPLE_SPACE(1):SAMPLE_SPACE(2), ...
%         MAGIC_NUM*x(:,:));
%     figure; imagesc(SAMPLE_SPACE(1):SAMPLE_SPACE(2), ...
%         SAMPLE_SPACE(1):SAMPLE_SPACE(2), ...
%         MAGIC_NUM*x(:,:).*w(:,:,1));
%     figure; imagesc(SAMPLE_SPACE(1):SAMPLE_SPACE(2), ...
%         SAMPLE_SPACE(1):SAMPLE_SPACE(2), ...
%         MAGIC_NUM*x(:,:).*w(:,:,2));
    
    %%%%%%%%%%%%%%%%
    cnt = cnt+1;
end

figure; hold on;
imagesc(SAMPLE_SPACE(1):SAMPLE_SPACE(2), ...
    SAMPLE_SPACE(1):SAMPLE_SPACE(2), ...
    MAGIC_NUM*x);
% set(gca,'YDir','normal');
FWHMRatio =2.35;
for k=1:K
    DrawEllipse(Mu_est(k,1),Mu_est(k,2), ...
        FWHMRatio*sqrt(Sigma_est(k,1)),FWHMRatio*sqrt(Sigma_est(k,2)));
end

disp(Mu);
disp(Sigma);
disp(ThetaProb);
disp(Mu_est);
disp(Sigma_est);
disp(Phi_est);

disp('Total steps');
disp(cnt);