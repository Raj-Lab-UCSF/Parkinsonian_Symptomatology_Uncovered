% PPMI Prediction Function Code
% Alex Anh-Tu Nguyen - Raj Lab - UCSF

% Purpose: This code is a simple run me code that will call functions to 
% set parameters, train the net, test the net, and calculate saliency maps.

% Note: Code requires Matlab R2018a and up and the following toolboxes:
% Neural Network Toolbox, Statistics and Machine Learning Toolbox,
% Communications System Toolbox, Bioinformatics Toolbox. Code has only been 
% validated on macOS High Sierra.

% Usage case:
clear all; close all; clc;

%%% CHANGE THESE 3 OPTIONS ONLY %%%
year = 2;                % change for predicting future year (1,2,3,or 4)
pred_type = 'UPDRS3';    % change for prediction type ('UPDRS3' or 'MoCA')
model = 'DBM';           % change for different model ('DBM' or 'NDM')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file_type = ['year' num2str(year) '_' pred_type '.mat'];
load(file_type)

set_params

[deepnet_cv] = AE_train_5cv(year,pred_type,model,iter,L2_weight,spars_reg,...
    spars_pro,prog_win,enc_tf,dec_tf,hiddenSize1,hiddenSize2,loss_func);

[deepnet] = AE_test(year,pred_type,model,iter,L2_weight,spars_reg,...
    spars_pro,prog_win,enc_tf,dec_tf,hiddenSize1,hiddenSize2,loss_func);

[dP_matrix] = calculate_saliency_map(per_change,pred_type,model); 
