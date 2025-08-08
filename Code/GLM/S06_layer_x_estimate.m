%Script to estimate the beta coefficients in A * beta = b

function S06_layer_x_estimate(subject, game, DQN, reg_parameter)

root_path = 'YOUR_DATA_PATH';
main_dir = strcat(subject, '/', game) ;

sess_no = 5;

GLM_name = 'GLMs_Layer_x';

lambda_seq = reg_parameter;

load(strcat(root_path, main_dir, '/GLMs/GLM_empty/SPM.mat'));
mask_nii = load_nii(strcat(root_path, main_dir, '/GLMs/GLM_empty/mask.nii'));

N_voxels_in_mask = sum(mask_nii.img(:));
xyz_vec = NaN(N_voxels_in_mask, 3);
vx_count = 0;

for x = 1:79

    for y = 1:95
        
        for z = 1:79
            
            if mask_nii.img(x,y,z) == 1
            
                vx_count = vx_count + 1;             
                xyz_vec(vx_count,:) = [x,y,z];
            
            end
            
        end
        
    end
    
end

clear mask_nii

for run_no = 1:sess_no 

    disp(run_no);

    res_ind = SPM.xX.K(run_no).row(2:end-1)';
    
    clear X_hpf
    
    load(strcat(root_path, main_dir, '/GLMs/', GLM_name, '/', DQN, '/GLM_sess_', num2str(run_no), '/X_hpf.mat'), 'X_hpf');
    
    X_hpf = zscore(X_hpf(:,1:end-1),1);

    NNZ_per_model = ceil(1.5*size(X_hpf,1));

    beta_value_mat = NaN(NNZ_per_model, N_voxels_in_mask, size(lambda_seq,2), 'single');

    beta_ind_mat = zeros(NNZ_per_model, N_voxels_in_mask, size(lambda_seq,2), 'uint16');

    res_nii_img = NaN(size(res_ind, 1), N_voxels_in_mask);

    res_nii = load_nii(strcat(root_path, main_dir, '/GLMs/GLM_empty/Res_all.nii'));

    for vx_count = 1:N_voxels_in_mask
        
        x = xyz_vec(vx_count,1);
        y = xyz_vec(vx_count,2);
        z = xyz_vec(vx_count,3);
        
        res_nii_img(:,vx_count) = squeeze(res_nii.img(x,y,z,res_ind));
        
    end

    clear res_nii

    res_nii_img = zscore(res_nii_img,1); 

    [B, B0]  = lasso_gpu(X_hpf, res_nii_img, lambda_seq);

    for lambda_no = 1:size(lambda_seq,2)
        for voxel_no  = 1:N_voxels_in_mask
            beta_nnz_ind  = find(squeeze(B(:,voxel_no, lambda_no)) ~= 0);
            
            if size(beta_nnz_ind) > NNZ_per_model
                disp('To many regressors');
            end
            
            beta_ind_mat(1:size(beta_nnz_ind), voxel_no , lambda_no) = uint16(beta_nnz_ind);
            beta_value_mat(1:size(beta_nnz_ind), voxel_no , lambda_no) = single(B(beta_nnz_ind, voxel_no, lambda_no));

        end
    end

    intercept = squeeze(B0);
    clear B0
    save(strcat(root_path, main_dir, '/GLMs/', GLM_name, '/', DQN, '/GLM_sess_', num2str(run_no), '/lambda_all_beta_all_sparse'),...
    	 'lambda_seq', 'xyz_vec', 'NNZ_per_model', 'beta_value_mat', 'beta_ind_mat', 'intercept', '-v7.3');

    clear beta_ind_mat
    clear beta_value_mat
    clear intercept
    clear B

end
