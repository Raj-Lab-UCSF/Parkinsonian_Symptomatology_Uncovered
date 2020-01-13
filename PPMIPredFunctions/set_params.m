function set_params

% check for release version
ver = version('-release');
ver_num = str2double(ver(1:4));
if ver_num < 2018
    error('Please update Matlab to R2018a and up!')
    return
else
    disp('Loading parameters...');
    % change below for autoencoder parameters
    assignin('base','iter',1000); % iter = 1000;
    assignin('base','L2_weight',0.001); % L2_weight = 0.001;
    assignin('base','spars_reg',2); % spars_reg = 2;
    assignin('base','spars_pro',0.05); % spars_pro = 0.05;
    assignin('base','prog_win',false); % prog_win = false;
    assignin('base','enc_tf','logsig'); % enc_tf = 'logsig'; % change ('logsig' or 'satlin')
    assignin('base','dec_tf','logsig'); % dec_tf = 'logsig'; % change ('logsig', 'satlin', or 'purelin')
    assignin('base','hiddenSize1',20); % hiddenSize1 = 20;
    assignin('base','hiddenSize2',10); % hiddenSize2 = 10;
    assignin('base','loss_func','crossentropy'); % loss_func = 'crossentropy'; % change ('crossentropy' or 'mse')
    % change below for saliency map calculation
    assignin('base','per_change',1.2); % 1.2 = 20% increase
    disp('Parameters loaded.')
end

end