function S04_residuals_3D_to_4D(subject, game)

root_path = 'YOUR_DATA_PATH';
main_dir = strcat(subject, '/', game) ;

GLM_name = 'GLM_empty';

clear matlabbatch
    
res_dir = remove_dots_from_dir(dir(strcat(root_path, main_dir, '/GLMs/', GLM_name, '/Res_*.nii')));

for i = 1:size(res_dir,1)

    if str2double(res_dir(i).name(1, 5:8)) ~= i

        error('Residual sequence corrupted!');

    end

end

output_path = strcat(root_path, main_dir, '/GLMs/', GLM_name, '/Res_all.nii');

for i = 1:size(res_dir,1)

    matlabbatch{1}.spm.util.cat.vols{i,1} = strcat(root_path, main_dir, '/GLMs/', GLM_name, '/', res_dir(i).name); 

end

matlabbatch{1}.spm.util.cat.name = output_path;
matlabbatch{1}.spm.util.cat.dtype = 0;
matlabbatch{1}.spm.util.cat.RT = 0.987;


save(strcat(root_path, main_dir, '/Batches/Batch_residuals_3D_to_4D'), 'matlabbatch');

