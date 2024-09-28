% % Macro Problem Set 1 % %
% % ------------------- % %
% Author: Nicolas Moreno Arias
% Date: Sep21/2024

% Preamble:
clear all  clc
if ~exist(['.' filesep 'Figures'],'dir')
    mkdir(['.' filesep 'Figures'])
end
Plot_opt = 'on';
% ---------------------------------%
% % Import Data File
DB  = readtable("GDPDEF.xls",Range="A12:B321"); % Read Database
DB.Properties.VariableNames = {'Dates','P_t'}; % Give Data proper names
DB.Dates.Format = 'yyyy,QQ'; % Transform Dates to Quarters
P_t = DB.P_t; % Create Auxiliary Var

%% Warm Up
% ---------------------------------%
% Calculate Annualized Inflation
% ---------------------------------%
Start_t = datefind(datetime(1983,4,1),DB.Dates);
X_t     = 400*log(P_t(Start_t:end)./P_t(Start_t-1:end-1));
% Report this year's values:
disp('Deflator Values for 2024 in Logs:')
disp({num2str(log(P_t(end-1))) num2str(log(P_t(end)))})
disp('Annualized Inflation Values for 2024:')
disp({num2str(X_t(end-1)) num2str(X_t(end))})

%% Part 1
% ---------------------------------%
% %    ESTIMATION OF AN AR(2)    % %
% ---------------------------------%
% Getting Data Ready
Num_Lags = 2;
Start_t2 = datefind(datetime(1984,1,1),DB.Dates) - (Start_t -1);
End_t    = datefind(datetime(2019,10,1),DB.Dates)- (Start_t -1);
X_t2     = X_t(Start_t2:End_t);
nObs     = length(X_t2);
Model_   = arima(Num_Lags,0,0); % Estimate ARIMA with toolbox as Benchmark (ML method)
Estim_   = estimate(Model_,X_t2(1+Num_Lags:end)); 

% Manual Estimation:
% Create Matrices for Estimation (XMAT = Constant, X_{t-1} X_{t-2}):
n_var    = 3; % Number of Independent terms in equation
X_MAT    = [ones(length(X_t2)-Num_Lags,1) X_t2(1+(Num_Lags-1):end-(Num_Lags-1)) X_t2(1:end-Num_Lags)];
Y_VEC    = X_t2(1+Num_Lags:end);
actual_n = length(Y_VEC); % Find effective number of datapoints
Estim_2  = fitlm(X_MAT(:,2:end),Y_VEC); % Estimate ARIMA with OLS command as 2nd Benchmark
% Manual Estimation and Implied Mean:
Beta_VEC   = (X_MAT'*X_MAT)\(X_MAT'*Y_VEC);
mu_Hat     = (Beta_VEC(1)/( 1-sum(Beta_VEC(2:end)) ))*ones(actual_n,1); % Analytical calculation
% Find the Predicted Y and Residuals:
Y_Hat      = Beta_VEC(1)*X_MAT(:,1) + Beta_VEC(2)*X_MAT(:,2) + Beta_VEC(3)*X_MAT(:,3);
samp_mean  = mean(Y_Hat)*ones(actual_n,1); % Create Vector with sample mean
Res_       = Y_VEC - Y_Hat; % Series of Residuals
check_Res  = mean(Res_);
Res_sq     = Res_.^2; % Series of Squared residuals
% Compute Variances and Varcov Matrix:
var_Hat    = (Res_'*Res_)/(actual_n-n_var); % Estimate Residuals variance w/ Linear Algebra
sigma_Hat  = sqrt(var_Hat); % Compute standard deviation of Residuals
varcov_MAT = var_Hat.*inv(X_MAT'*X_MAT); % Compute VARCOV MAT
Bvar_Hat   = diag(varcov_MAT); % Calculate the variance of each estimator
% Compare Beta VARCOV with Newey-West:
[NW_varcov,se,~]        = hac(Estim_2,Intercept=true,Display="full");
Check_MAT = varcov_MAT - NW_varcov;
if norm(Check_MAT,Inf) > 1e-4
    % spy(Check_MAT);shg;
    Bvar_Hat = diag(NW_varcov);
    disp('Autocorrelation Correction for Var(Beta) was done!')
end
% Compute t-statistics for Betas:
t_VEC      = Beta_VEC./sqrt(Bvar_Hat);
% Calculate implied variance of Series (Analytical Calculation):
gamma_Hat  = var_Hat/(1-Beta_VEC(2)^2-Beta_VEC(3)^2); 
% Final Reporting of results
Coef_Table = table(Beta_VEC, Bvar_Hat.^(0.5), t_VEC,...
                'VariableNames',{'Estimators','Std.Dev', 't-stat'},...
                'RowNames',{'Constant','X_{t-1}', 'X_{t-2}'});
disp(['Implied mean = ' num2str(mu_Hat(1),4)])
disp(['Residuals variance = ' num2str(var_Hat,4)])
disp(['Implied variance = ' num2str(gamma_Hat,4)])
%% Part 2
% -----------------------------------%
% %   FORECASTS 2020 WITH AR(2)    % %
% -----------------------------------%
% Get Vectors Ready:
Lags_X = [X_t(2:End_t) X_t(1:End_t-1)]; 
obsH   = length(Lags_X); % Last Observed Data
nSteps = 4; % Select number of steps ahead for forecast
Lags_X = [Lags_X;NaN(nSteps,size(Lags_X,2))];
Lags_X = [ones(length(Lags_X),1), Lags_X]; 
YPred  = NaN(nSteps,1); % Pre-allocate
% Compute the Forecasts
for jj=1:nSteps
    YPred(jj,1) = Lags_X(obsH+jj-1,:)*Beta_VEC; % Forecast
    Lags_X(obsH+jj,2) = YPred(jj,1); % Update Value for X_{t-1}
    Lags_X(obsH+jj,3) = Lags_X(obsH+jj-1,2); % Update Value for X_{t-2}
end
FE_Vec = X_t(End_t+1:End_t+nSteps)-YPred; % Forecast 
RMSE_  = sqrt(mean(FE_Vec'*FE_Vec));
%% Part 3
% -----------------------------------%
% %    COMPUTE MA'S 0-4 WEIGHTS    % %
% -----------------------------------%
%{ 
% We have the following equation:
X_t = Beta_VEC(1) + Beta_VEC(2)* X_{t-1} + Beta_VEC(3)*X_{t-2} + Eps_t
Iterating Backwards:
X_{t-1} = Beta_VEC(1) + Beta_VEC(2)* X_{t-2} + Beta_VEC(3)*X_{t-3} + Eps_{t-1}
Replacing in 1:
X_t = Beta_VEC(1) + Beta_VEC(1)*Beta_VEC(2) + Beta_VEC(2)^2*X_{t-2}+ Beta_VEC(2)*Beta_VEC(3)*X_{t-3} 
            + Beta_VEC(2)*Eps_{t-1} + Beta_VEC(3)*X_{t-2} + Eps_t
It's easy to see that if Beta_VEC(2:3) are both less than 1, then their
product, both by themselves and crossed, will tend to zero. 
%}
J_ = 5;
Weight_VEC      = ones(J_,1);
Weight_VEC(2,1) = Beta_VEC(2);
% Iteratively Compute the Weights:
for zz =3:J_
    Weight_VEC(zz,1) = flip(Weight_VEC(zz-2:zz-1,1)')*Beta_VEC(2:3,1);
end
%% Part 4
% -----------------------------------%
% %         COMPUTE IRFS           % %
% -----------------------------------%
% Compute the IRF using MA Weights:
Shock_VEC = [1 -2 sigma_Hat]';
nShocks   = length(Shock_VEC);
IRF_MAT   = NaN(nShocks,J_);
for kk = 1:nShocks
    % for hh = 1:J_
        IRF_MAT(kk,:) = Shock_VEC(kk,1)*Weight_VEC;
end

%% Wrapping up:
% -----------------------------------%
% %         COMPUTE PLOTS          % %
% -----------------------------------%
% Plot the IRFS
if strcmp(Plot_opt,'on')
% % Data:
    fig_0 = figure('Color','w','Position',[200 100 1200 800],'Visible','off');
    orient(fig_0,'landscape')
    ax = gca;
    plot(DB.Dates(Start_t:end),X_t,'Color',[0 0.6 0.45] , ...
                                   'LineStyle','-', 'Marker','o', ...
                                   'LineWidth',2)
    hold on
    plot(DB.Dates(Start_t:end),mean(X_t)*ones(length(X_t),1),'Color', [0 0 0],'LineStyle',':','LineWidth',1)
    hold off
    grid on
    ax.XAxis.TickLabelFormat = 'yyyy,QQ';
    numTicks = 12;  % Desired number of ticks
    dateTicks = linspace(DB.Dates(Start_t), DB.Dates(end), numTicks);
    xticks(dateTicks)
    ylabel('Annualized Inflation (%)')
    xlabel('Dates')
    legend({'Annualized Inflation',['Sample Mean = ' num2str(mean(X_t),2) '%']},'Location','northwest','FontSize',14)
    title('GDP Deflator (SA, Annualized Quarterly Growth Rate)','FontSize',14)
    ax.FontSize = 14;
    print(['Figures' filesep 'GDP_Deflator_Inflation'],'-dpdf','-r0')

% % Forecasts
    fig_1 = figure('Color','w','Position',[200 100 1200 800],'Visible','off');
    orient(fig_1,'landscape')
    ax = gca;
    plot(DB.Dates(Start_t+obsH-40:Start_t+obsH+nSteps),X_t(End_t-40:End_t+nSteps,1),'Color',[0 0.6 0.45] , ...
                                   'LineStyle','-', 'Marker','o', ...
                                   'LineWidth',2)
    hold on
    plot(DB.Dates(Start_t+obsH-40:Start_t+obsH+nSteps),[X_t(End_t-40:End_t);YPred],'Color', [0 0.3 0.9],'LineStyle','--','LineWidth',2)
    plot(DB.Dates(Start_t+obsH-40:Start_t+obsH+nSteps),mu_Hat(1:length(X_t(End_t-40:End_t+nSteps)),1),'Color', [0.4 0.4 0.4],'LineStyle',':','LineWidth',1.5)
    hold off
    grid on
    ax.XAxis.TickLabelFormat = 'yyyy,QQ';
    numTicks = 12;  % Desired number of ticks
    dateTicks = linspace(DB.Dates(Start_t+obsH-40), DB.Dates(Start_t+obsH+nSteps), numTicks);
    xticks(dateTicks)
    ylabel('Annualized Inflation (%)')
    xlabel('Dates')
    legend({'Actual Inflation','Forecast',['Implied Mean $$\hat{\mu}$$ = ' num2str(mu_Hat(1),4) '$$ \% $$']},'Location','northwest','FontSize',14,'Interpreter','latex')
    title('GDP Deflator (SA, Annualized Quarterly Growth Rate)','FontSize',14)
    ax.FontSize = 14;
    print(['Figures' filesep 'Inflation_Forecast'],'-dpdf','-r0')


% IRFs:  
    Horizon_ = J_-1;
    fig_2 = figure('Color','w','Position',[200 100 1200 800],'Visible','off');
    orient(fig_2,'landscape')
    ax = gca;
    plot(0:Horizon_,IRF_MAT(1,:)','Color',[0 0.6 0.45] , ...
                                   'LineStyle','-', 'Marker','o', ...
                                   'LineWidth',2)
    hold on
    plot(0:Horizon_,IRF_MAT(2,:)','Color', [0.75 0 0.7] ,'LineStyle','-.','LineWidth',2)
    plot(0:Horizon_,IRF_MAT(3,:)','Color', [0.4 0.4 0.4],'LineStyle',':','LineWidth',2)
    plot(0:Horizon_,zeros(J_,1),'Color', [0 0 0],'LineStyle','-','LineWidth',1)
    hold off
    grid on
    numTicks = 5;  % Desired number of ticks
    % xticks(dateTicks)
    ylabel('Annualized Inflation (%)')
    xlabel('Dates')
    legend({'1% Shock','-2% Shock', 'One Std. Dev. Shock'},'Location','southeast','FontSize',14)
    title('Impulse-Response Functions for 4-steps ahead','FontSize',14)
    ax.FontSize = 14;
    print(['Figures' filesep 'IRFs'],'-dpdf','-r0')
end

