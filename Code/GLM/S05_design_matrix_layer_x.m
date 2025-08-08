%Script to create the design matrix A in A*beta = b, where the columns of A are features extracted from layer x of the DQN.

function S05_design_matrix_layer_x(subject, game, DQN)

root_path = 'YOUR_DATA_PATH';
main_dir = strcat(subject, '/', game) ;
sess_no = 5;

GLM_name = 'GLMs_Layer_x';

TR = 0.987;

load('hrf_15hz_steps.mat');

if strcmp(DQN,'BaselineDQN')
    model_weight = "240";
elseif strcmp(DQN,'ApeX') && strcmp(game,'space_invaders')
    model_weight = "540000";
elseif strcmp(DQN,'ApeX') && strcmp(game,'breakout')
    model_weight = "660000";
elseif strcmp(DQN,'ApeX') && strcmp(game,'enduro')
    model_weight = "480000";
elseif strcmp(DQN,'SEED') && strcmp(game,'space_invaders')
    model_weight = "0-ckpt-130";
elseif strcmp(DQN,'SEED') && strcmp(game,'breakout')
    model_weight = "0-ckpt-98";
elseif strcmp(DQN,'SEED') && strcmp(game,'enduro')
    model_weight = "0-ckpt-98";
end

mkdir(strcat(root_path, main_dir, '/GLMs/', GLM_name, '/', DQN, '/'));

layer_data_sess = dlmread(strcat('PATH_TO_DQN_FEATURES/Second_Layer_RELU_', DQN, '_model_', model_weight,...
		 '_', game, '_', subject, '_Session_1.csv'));

layer_data_all = zeros(size(layer_data_sess,1)*sess_no,size(layer_data_sess,2));

layer_data_all(1:size(layer_data_sess,1),:) = layer_data_sess;

for run_no = 2:sess_no

    layer_data_sess = dlmread('PATH_TO_DQN_FEATURES/Second_Layer_RELU_', DQN, '_model_', model_weight,...
    			 '_', game, '_', subject, '_Session_', num2str(run_no), '.csv'));   
    layer_data_all(size(layer_data_sess,1)*(run_no-1)+1:size(layer_data_sess,1)*run_no,:) = layer_data_sess;
    
end

zero_columns = all(layer_data_all == 0); 
layer_data_all(:, zero_columns) = [];

for run_no = 1:sess_no
    
    mkdir(strcat(root_path, main_dir, '/GLMs/', GLM_name, '/', DQN, '/GLM_sess_', num2str(run_no)));
    
    clear layer_data
        
    layer_data = layer_data_all(size(layer_data_sess,1)*(run_no-1)+1:size(layer_data_sess,1)*run_no,:);

    layer_data = zscore(layer_data, 1, 2);
    layer_data = zscore(layer_data, 1, 1);
    
    layer_data_conv = NaN(size(layer_data));
    
    for unit_no = 1:size(layer_data,2)
        
        layer_unit_conv = conv(layer_data(:,unit_no), hrf_15hz_steps, 'full');
        
        layer_data_conv(:,unit_no) = layer_unit_conv(1:size(layer_data,1), :); 
        
    end
    
    clear no_of_volumes_to_remove
    clear last_volume
    clear time_in_sec_vec
    
    load(strcat(root_path, main_dir, '/GLMs/removed_first_last_timing_sess_', num2str(run_no), '.mat'),...
    	'no_of_volumes_to_remove', 'last_volume', 'time_in_sec_vec');
    
    time_in_TR_vec = time_in_sec_vec/TR;

    no_of_TRs = floor(max(time_in_TR_vec)); 
       
    shift_fraction_of_TR = 0.5; 

    TR_start = find(time_in_TR_vec < 1);
    TR_start = TR_start(end,1);

    downsample_ind_vec = NaN(no_of_TRs-1,1);

    for TR_no = 2:no_of_TRs 

        TR_end = find(time_in_TR_vec < TR_no);

        TR_end = TR_end(end,1);

        TR_length = TR_end - TR_start;

        TR_shift = floor(shift_fraction_of_TR*TR_length); 

        if TR_shift < 5 

            warning(['Run ',num2str(run_no), ', TR ', num2str(TR_no), ': TR_shift too short!']);

        end

        downsample_ind_vec(TR_no-1,1) = TR_start + TR_shift + 1; 

        TR_start = TR_end;

    end
    
    layer_data_conv_downsampled = layer_data_conv(downsample_ind_vec,:);
    
    K.RT = TR;
    K.row = 1:no_of_TRs-1;
    K.HParam = 128;
    
    nK = spm_filter(K);
    
    X0 = nK.X0;
    
    layer_data_conv_downsampled_hpf = (eye(no_of_TRs-1) - X0 * X0') * layer_data_conv_downsampled;
    
    X_hpf = layer_data_conv_downsampled_hpf; 
    
    save(strcat(root_path, main_dir, '/GLMs/', GLM_name, '/', DQN, '/GLM_sess_', num2str(run_no), '/X_hpf'), 'X_hpf');

end
