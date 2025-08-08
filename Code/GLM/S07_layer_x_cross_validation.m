function S07_layer_x_cross_validation(subject, game, DQN, reg_parameter)

root_path = 'YOUR_DATA_PATH';
main_dir = strcat(subject, '/', game) ;

sess_no = 5;

GLM_name = 'GLMs_Layer_x';

lambda_seq = reg_parameter;

load(strcat(root_path, main_dir, '/GLMs/GLM_empty/SPM.mat'));

beta_template_nii = load_nii(strcat(root_path, main_dir, '/GLMs/GLM_empty/beta_0001.nii'));

NNZ_per_model_all = zeros(1,sess_no);

load(strcat(root_path, main_dir, '/GLMs/', GLM_name, '/', DQN, '/GLM_sess_1/lambda_all_beta_all_sparse.mat'), 'xyz_vec');

N_voxels_in_mask = size(xyz_vec,1);

beta_value_all_runs = NaN(max(NNZ_per_model_all), N_voxels_in_mask, size(lambda_seq,2), sess_no, 'single');
beta_ind_all_runs = zeros(max(NNZ_per_model_all), N_voxels_in_mask, size(lambda_seq,2), sess_no, 'uint16');
intercept_all_runs = zeros(N_voxels_in_mask, size(lambda_seq,2), sess_no);

for run_no = 1:sess_no
       
    clear 'NNZ_per_model'
    clear xyz_vec
    clear beta_ind_mat
    clear beta_value_mat
    
    load(strcat(root_path, main_dir, '/GLMs/', GLM_name, '/', DQN, '/GLM_sess_', num2str(run_no), '/lambda_all_beta_all_sparse.mat'),...
    	 'NNZ_per_model', 'xyz_vec', 'beta_ind_mat', 'beta_value_mat', 'intercept');

    beta_value_all_runs(1:NNZ_per_model_all(1,run_no),:,:,run_no) = beta_value_mat; 
    beta_ind_all_runs(1:NNZ_per_model_all(1,run_no),:,:,run_no) = beta_ind_mat;
    intercept_all_runs(:,:,run_no) = intercept;
    
end

clear beta_ind_mat
clear beta_value_mat
clear intercept

corr_r_all_runs = NaN(N_voxels_in_mask, size(lambda_seq,2), sess_no);

for run_no = 1:sess_no 

    disp(run_no);
    
    res_nii = load_nii(strcat(root_path, main_dir, '/GLMs/GLM_empty/Res_all.nii'));
    res_ind = SPM.xX.K(run_no).row(2:end-1)';
    res_nii_img = NaN(size(res_ind, 1), N_voxels_in_mask);
    
    for vx_count = 1:N_voxels_in_mask
        
        x = xyz_vec(vx_count,1);
        y = xyz_vec(vx_count,2);
        z = xyz_vec(vx_count,3);
        
        res_nii_img(:,vx_count) = squeeze(res_nii.img(x,y,z,res_ind));
        
    end
    
    clear res_nii
    clear X_hpf
    
    load(strcat(root_path, main_dir, '/GLMs/', GLM_name, '/', DQN, '/GLM_sess_', num2str(run_no), '/X_hpf.mat'));    
    X_hpf = zscore(X_hpf(:,1:end-1),1);

    train_bin = true(sess_no,1);
    train_bin(run_no,1) = false;

    beta_value_train = beta_value_all_runs(:,:,:,train_bin);
    beta_ind_train = beta_ind_all_runs(:,:,:,train_bin);
    intercept_train = intercept_all_runs(:,:,train_bin);
    
    corr_r_curr_run_mat = NaN(N_voxels_in_mask, size(lambda_seq,2));

    for vx_count = 1:N_voxels_in_mask

        res_ts = res_nii_img(:, vx_count);
    
        res_ts = zscore(res_ts,1);

        beta_full_mat_all_lambda = NaN(size(X_hpf,2), size(lambda_seq,2));

        for lambda_no = 1:size(lambda_seq,2)

            beta_full_lambda_curr = zeros(size(X_hpf,2), sess_no-1, 'single');
            
            for train_run_no = 1:sess_no-1

                beta_value_vec_lambda_curr_train_run = squeeze(beta_value_train(:,vx_count,lambda_no,train_run_no));
                beta_ind_vec_lambda_curr_train_run = squeeze(beta_ind_train(:,vx_count,lambda_no,train_run_no));
                
                beta_value_vec_lambda_curr_train_run(beta_ind_vec_lambda_curr_train_run == 0,:) = []; 
                beta_ind_vec_lambda_curr_train_run(beta_ind_vec_lambda_curr_train_run == 0,:) = []; 
                
                beta_full_lambda_curr(beta_ind_vec_lambda_curr_train_run, train_run_no) = beta_value_vec_lambda_curr_train_run;
            end     

            beta_full_lambda_curr = double(beta_full_lambda_curr);
            
            beta_full_lambda_curr_mean = mean(beta_full_lambda_curr,2); 
            
            beta_full_mat_all_lambda(:,lambda_no) = beta_full_lambda_curr_mean;
            
        end

        intercept_all_runs_mean = zeros(1, size(lambda_seq,2));

        intercept_all_runs_mean(1,:) = mean(intercept_train(vx_count,:,:),3);
        
        y_pred_all_lambda = X_hpf*beta_full_mat_all_lambda + ones(size(X_hpf,1),1)*intercept_all_runs_mean; 
        
        corr_r_all_lambda = corrcoef([res_ts, y_pred_all_lambda]); 
        
        corr_r_curr_run_mat(vx_count,:) = corr_r_all_lambda(1, 2:end); 
        
    end
    
    corr_r_all_runs(:,:,run_no) = corr_r_curr_run_mat;
    
    
end

corr_r_all_runs_mean = mean(corr_r_all_runs,3);

for lambda_no = 1:size(lambda_seq,2)
    
    corr_3D = NaN(79,95,79);
    
    for vx_count = 1:N_voxels_in_mask
        
        x = xyz_vec(vx_count,1);
        y = xyz_vec(vx_count,2);
        z = xyz_vec(vx_count,3);
        
        corr_3D(x,y,z) = corr_r_all_runs_mean(vx_count, lambda_no);
        
    end
    
    beta_template_nii.img = single(corr_3D);
    
    save_nii(beta_template_nii, strcat(root_path, main_dir, '/GLMs/', GLM_name, '/', DQN, '/lambda_', num2str(lambda_no, '%04.0f'), '_correlation_map_layer_x'));
    
end

disp(' ');
disp('Done!');
disp(' ');

