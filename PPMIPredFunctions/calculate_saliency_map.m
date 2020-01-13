function [dP_matrix] = calculate_saliency_map(per_change,pred_type,model)

disp('Calculating saliency maps...');

if strcmp(pred_type,'UPDRS3')
    if model == 'NDM'
        for i=1:32
            testing_NDM_saliency = evalin('base', 'testing_NDM')';
            testing_NDM_saliency(i+21,:) = ...
                                  testing_NDM_saliency(i+21,:)*per_change;
            deepnet = evalin('base', 'deepnet');
            UPDRS3_p_matrix(i,:) = deepnet(testing_NDM_saliency);                           
        end
        dP_matrix = UPDRS3_p_matrix - evalin('base', 'model_pred_test');
    elseif model == 'DBM'
        for i=1:32
            testing_DBM_saliency = evalin('base', 'testing_DBM')';
            testing_DBM_saliency(i+21,:) = ...
                                  testing_DBM_saliency(i+21,:)*per_change;
            deepnet = evalin('base', 'deepnet');
            UPDRS3_p_matrix(i,:) = deepnet(testing_DBM_saliency);                           
        end
        dP_matrix = UPDRS3_p_matrix - evalin('base', 'model_pred_test');
    else
        error('Please check model variable.')
    end
elseif strcmp(pred_type,'MoCA')
    if model == 'NDM'
        for i=1:34
            testing_NDM_saliency = evalin('base', 'testing_NDM')';
            testing_NDM_saliency(i+18,:) = ...
                                  testing_NDM_saliency(i+18,:)*per_change;
            deepnet = evalin('base', 'deepnet');                  
            MoCA_p_matrix(i,:) = deepnet(testing_NDM_saliency);                          
        end
        dP_matrix = MoCA_p_matrix - evalin('base', 'model_pred_test');
    elseif model == 'DBM'
        for i=1:34
            testing_DBM_saliency = evalin('base', 'testing_DBM')';
            testing_DBM_saliency(i+18,:) = ...
                                  testing_DBM_saliency(i+18,:)*per_change;
            deepnet = evalin('base', 'deepnet');                  
            MoCA_p_matrix(i,:) = deepnet(testing_DBM_saliency);                          
        end
        dP_matrix = MoCA_p_matrix - evalin('base', 'model_pred_test');
    else
        error('Please check model variable.')
    end
else
    error('Please check if ''type'' is ''UPDRS3'' or ''MoCA''')
end

disp('Saliency maps complete.');

end