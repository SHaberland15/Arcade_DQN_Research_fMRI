% Script to perform motion regression by fitting a GLM to each voxel time series using the six motion parameters
% as regressors to remove motion-related variance.

function S03_GLM_empty(subject, game)

root_path = 'YOUR_DATA_PATH';
main_dir = strcat(subject, '/', game) ;
sess_no = 5;

GLM_name = 'GLM_empty';

clear matlabbatch

matlabbatch{1}.spm.stats.fmri_spec.dir = {strcat(root_path, main_dir, '/GLMs/', GLM_name)};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 0.987;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

for run_no = 1:sess_no
    
    clear first_volume
    clear last_volume
    clear no_of_volumes_to_remove
    
    load(strcat(root_path, main_dir, '/GLMs/removed_first_last_timing_sess_', num2str(run_no), '.mat'),...
    	'first_volume', 'last_volume', 'no_of_volumes_to_remove');
    
    n_EPIs = last_volume - no_of_volumes_to_remove;
    
    EPI_no_vec = (first_volume:last_volume)';
    
    EPI_dir = dir(strcat(root_path, main_dir, '/EPI/session_', num2str(run_no), '/swuaMF*.nii'));
    
    if size(EPI_dir,1) ~= 1
        
        error('EPIs not found or not unique!');
        
    end
    
    for EPI_count = 1:n_EPIs
        
         matlabbatch{1}.spm.stats.fmri_spec.sess(run_no).scans{EPI_count,1} = strcat(root_path, main_dir,...
         		'/EPI/session_', num2str(run_no), '/', EPI_dir(1).name, ',', num2str(EPI_no_vec(EPI_count,1)));
        
    end
    
    matlabbatch{1}.spm.stats.fmri_spec.sess(run_no).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(run_no).multi = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess(run_no).regress = struct('name', {}, 'val', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(run_no).multi_reg = {strcat(root_path, main_dir, '/GLMs/', GLM_name,...
    			'/movement_regs_sess_', num2str(run_no), '.mat')};
    matlabbatch{1}.spm.stats.fmri_spec.sess(run_no).hpf = 128;

end

matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'none';

matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', ...
	substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 1;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

save(strcat(root_path, main_dir, '/Batches/Batch_GLM_empty'), 'matlabbatch');
