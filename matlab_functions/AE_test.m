function [deepnet] = AE_test(year,pred_type,model,iter,L2_weight,spars_reg,...
    spars_pro,prog_win,enc_tf,dec_tf,hiddenSize1,hiddenSize2,loss_func)

disp('Starting to train model on entire training set...');
rng(1) % for reproducibility
n = length(evalin('base', 'PD_demo'));
if year == 1
    per_split = 0.2;
    try mod_ones_y = evalin('base', 'mod_ones_1y');
    catch mod_ones_y = evalin('base', 'MCI_ones_1y');
    end
elseif year == 2
    if strcmp(pred_type,'UPDRS3')
        per_split = 0.23;
    elseif strcmp(pred_type,'MoCA')
        per_split = 0.21;
    else
        error('Please check pred_type variable.')
    end
    try mod_ones_y = evalin('base', 'mod_ones_2y');
    catch mod_ones_y = evalin('base', 'MCI_ones_2y');
    end
elseif year == 3
    per_split = 0.23; 
    try mod_ones_y = evalin('base', 'mod_ones_3y');
    catch mod_ones_y = evalin('base', 'MCI_ones_3y');
    end
elseif year == 4
    per_split = 0.22;
    try mod_ones_y = evalin('base', 'mod_ones_4y');
    catch mod_ones_y = evalin('base', 'MCI_ones_4y');
    end
else
    per_split = NaN;
    mod_ones_y = NaN;
    error('Please change year to: 1, 2, 3, or 4')
end
c = cvpartition(n,'HoldOut',per_split);

idxTrain = c.training(1);
idxTest = ~idxTrain;

% combine all inputs into a single matrix
all_with_NDM = [evalin('base', 'PD_demo'),...
                evalin('base', 'PD_DBM'),...
                evalin('base', 'PD_NDM')]; % N by 192 double
    
all_with_2DBM = [evalin('base', 'PD_demo'),...
                 evalin('base', 'PD_DBM'),...
                 evalin('base', 'PD_DBM')]; % N by 192 double

% Statistically significant predictors for UPDRS3 (all timepoints) - DONE
keep_demo = [2,4,6,7,8,9,11,15,16,17,19,21,24,25,26,29,31,32,33,34,36];
keep_DBM = [2,3,9,15,24,30,31,39,45,49,60,63,67,71,72,78]+36;
keep_DBM2 = [2,3,9,15,24,30,31,39,45,49,60,63,67,71,72,78]+114;
keep_NDM = [2,3,9,13,15,24,30,31,34,39,45,47,49,67,68,78]+114;
keep = [keep_demo,keep_DBM,keep_NDM]; % total 53 remaining features
keep2 = [keep_demo,keep_DBM,keep_DBM2]; % total 53 remaining features

% Age,MDS-UPDRS-III,SBR L Caudate,SBR R Caudate,SBR L Putamen,SBR R Putamen
% A-syn,Hoehn & Yahr,HVLT Immediate Recall,HVLT Delayed Recognition Hits
% Symbol Digit Modalities Score,Benton Judgement of Line Orientation
% QUIP Positive Eating,QUIP Positive Buying,QUIP Positive Hobbies
% MDS-UPDRS Total Score, MDS-UPDRS-I PQ, MDS-UPDRS-II PQ
% Modified Schwab England ADL,UPSIT Total Score,SCOPA-AUT
% DBM Region 2,3,9,15,24,30,31,39,45,49,60,63,67,71,72,78
% NDM Region 2,3,9,13,15,24,30,31,34,39,45,47,49,67,68,78

% reduce features to only stat. sig.
reduced_with_NDM=all_with_NDM(:,keep);
reduced_with_2DBM=all_with_2DBM(:,keep2);
% NOTE: previously determined in EDA code using Spearman's correlation

% split into 2 datasets (1 each for DBMx2 and DBM+NDM)
training_NDM=reduced_with_NDM(idxTrain(:,1),:);
testing_NDM=reduced_with_NDM(idxTest(:,1),:);
training_DBM=reduced_with_2DBM(idxTrain(:,1),:);
testing_DBM=reduced_with_2DBM(idxTest(:,1),:);
assignin('base','testing_NDM',testing_NDM);
assignin('base','testing_DBM',testing_DBM);
      
% split output ones matrix
temp_ones = mod_ones_y';                      
training_mod_ones_y = temp_ones(idxTrain(:,1),:); 
testing_mod_ones_y = temp_ones(idxTest(:,1),:);   

%
if model == 'DBM' 
    features_input = training_DBM';
    features_input_test = testing_DBM';
elseif model =='NDM'
    features_input = training_NDM';
    features_input_test = testing_NDM';
else
    error('Please change model to ''DBM'' or ''NDM''');
end
features_output = training_mod_ones_y';  
features_output_test = testing_mod_ones_y'; 

%
autoenc1 = trainAutoencoder(features_input,hiddenSize1,...           
    'MaxEpochs',iter,... 
    'EncoderTransferFunction',enc_tf,... 
    'DecoderTransferFunction',dec_tf,... 
    'L2WeightRegularization',L2_weight,... 
    'ShowProgressWindow',prog_win,... 
    'SparsityRegularization',spars_reg,... 
    'SparsityProportion',spars_pro,... 
    'ScaleData',true,... 
    'UseGPU',false);     
features1 = encode(autoenc1,features_input);                         

autoenc2 = trainAutoencoder(features1,hiddenSize2,...
    'MaxEpochs',iter,... 
    'EncoderTransferFunction',enc_tf,... 
    'DecoderTransferFunction',dec_tf,... 
    'L2WeightRegularization',L2_weight,... 
    'ShowProgressWindow',prog_win,... 
    'SparsityRegularization',spars_reg,... 
    'SparsityProportion',spars_pro,... 
    'ScaleData',true,...
    'UseGPU',false);     
features2 = encode(autoenc2,features1);

softnet = trainSoftmaxLayer(features2,features_output,...    
    'MaxEpochs',iter,... 
    'LossFunction',loss_func); 

deepnet = stack(autoenc1,autoenc2,softnet);
deepnet = train(deepnet,features_input,features_output);            

model_pred_test = deepnet(features_input_test);                                
assignin('base','model_pred_test',model_pred_test);
plotconfusion(features_output_test,model_pred_test,'Testing set');      
disp('Finished prediction on test set.');

end
