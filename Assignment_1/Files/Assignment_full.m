%% TASK 4

% FIRST PART

% LOAD RELEVANT DATA
clear
load('A1_data.mat')

% ASSIGN VALUES LAMBDA = 0.1
omega_hat = lasso_ccdi(t, X, 0.1); 
y_hat = Xinterp*omega_hat;
y_data = X*omega_hat;

omega_hat2 = lasso_ccdi(t, X, 10); 
y_hat2 = Xinterp*omega_hat2;
y_data2 = X*omega_hat2;

omega_hat3 = lasso_ccdi(t, X, 2); 
y_hat3 = Xinterp*omega_hat3;
y_data3 = X*omega_hat3;

figure(1)
hold on
scatter(n, t, 30, 'r')
scatter(n, y_data, 30, 'filled')
plot(ninterp, y_hat, 'b')
legend('Real Data Instances', 'Synthesized Data Instaces', 'Reconstruction Line')
xlabel('Amount Data Instances')

figure(2)
hold on
scatter(n, t, 30, 'r')
scatter(n, y_data, 30, 'filled')
plot(ninterp, y_hat2, 'b')
legend('Real Data Instances', 'Synthesized Data Instaces', 'Reconstruction Line')
xlabel('Amount Data Instances')

figure(3)
hold on
scatter(n, t, 30, 'r')
scatter(n, y_data, 30, 'filled')
plot(ninterp, y_hat3, 'b')
legend('Real Data Instances', 'Synthesized Data Instaces', 'Reconstruction Line')
xlabel('Amount Data Instances')


% SECOND PART

non_z_coord = sum(omega_hat~=0);
non_z_coord2 = sum(omega_hat2~=0);
non_z_coord3 = sum(omega_hat3~=0);

%% TASK 5

% LOAD DATA

clear
load('A1_data.mat')

% SET PARAMETERS
min_lambda = 0.1;
max_lambda = max(abs(X'*t));
nbr_lambda = 200;
nbr_folds = 10;

% CONSTRUCT GRID FOR LAMBDA
grid_lambda = exp(linspace(log(min_lambda), log(max_lambda), nbr_lambda));

% CALCULATE OMEGA AND LAMBDA OPTIMAL + ROOT MEAN SQUARED ERRORS
[wopt, lambdaopt, RMSEval, RMSEest] = lasso_cvi(t, X, grid_lambda, nbr_folds);

% RECONSTRUCTION 
optimal_t = Xinterp*wopt;
data_t = X*wopt;

% FIRST PART

figure(4)
hold on
plot(log(grid_lambda), RMSEval)
plot(log(grid_lambda), RMSEest)
xline(log(lambdaopt), 'b');
legend('RMSEval', 'RMSEest', 'Optimal Lambda')
xlabel('Logarithmic Lambda')

% SECOND PART

figure(5)
hold on
scatter(n, t, 30, 'r')
scatter(n, data_t, 30, 'filled')
plot(ninterp, optimal_t, 'b')
legend('Reconstruction', 'Original points', 'Reconstructed points')
xlabel('Amount Data Instances')

%% TASK 6

% LOAD DATA
clear
load A1_data.mat

% SET PARAMETERS
min_lambda = 0.001;

% USE LENGTH OF DATA 
itr=floor(length(Ttrain)/352); 
max_lmbd = zeros(itr, 1); 
% ITERATE AND CALCULATE LAMBDAS
for i = 1:itr
    max_lmbd(i) = max(abs(Xaudio'*Ttrain(1+352*(i-1):i*352)));
end

% FIND LAMBDA MAX 
max_lambda = max(max_lmbd);
nbr_lambda = 100;
fold_amount = 3;
grid_lambda = exp(linspace(log(min_lambda), log(max_lambda), nbr_lambda));

%% CALCULATE OPTIMAL LAMBDA AND RSME VALUES

[wopt, lambdaopt, RMSEval, RMSEest] = multiframe_lasso_cvi(Ttrain, Xaudio, grid_lambda, fold_amount);

%% Task 6 Figures

figure(6)
hold on
plot(grid_lambda, RMSEval)
plot(grid_lambda, RMSEest)
xline(lambdaopt,'b')
legend('RMSEval', 'RMSEest', 'Optimal Lambda')
xlabel('Lambda')

figure(7)
hold on
plot(log(grid_lambda), RMSEval)
plot(log(grid_lambda), RMSEest)
xline(log(lambdaopt),'b')
legend('RMSEval', 'RMSEest', 'Optimal Lambda')
xlabel('Lambda')

%% Task 7

load A1_data.mat

% UNCOMMENT FOR QUICKER TEST
%lambdaopt = 0.0045;

Ytest = lasso_denoise(Ttest, Xaudio, lambdaopt);
soundsc(Ytest, fs)

save('denoised_audio','Ytest','fs')
