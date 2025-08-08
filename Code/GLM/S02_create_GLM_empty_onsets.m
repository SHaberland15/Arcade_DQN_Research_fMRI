function S02_create_GLM_empty_onsets(subject, game)

root_path = 'YOUR_DATA_PATH';
main_dir = strcat(subject, '/', game) ;
sess_no = 5;

GLM_name = 'GLM_empty';

mkdir(strcat(root_path, main_dir, '/GLMs/', GLM_name, '/'));

for run_no = 1:sess_no
    
    clear no_of_volumes_to_remove
    clear first_volume
    clear last_volume
    clear time_in_sec_vec
    
    load(strcat(root_path, main_dir, '/GLMs/removed_first_last_timing_sess_', num2str(run_no), '.mat'));    
        
    R_dir = dir(strcat(root_path, main_dir, '/EPI/session_', num2str(run_no), '/rp_*.txt'));
    
    if size(R_dir,1) ~= 1
        
        error('Movement regs txt-file not found or not unique!');
        
    end
    
    clear R
    
    R = dlmread(strcat(root_path, main_dir, '/EPI/session_', num2str(run_no), '/', R_dir(1).name));
    
    if size(R,2) ~= 6
        
        error('No. of columns of R not correct!');
        
    end
    
    R = R(first_volume:last_volume,:);
    
    save(strcat(root_path, main_dir, '/GLMs/', GLM_name, '/movement_regs_sess_', num2str(run_no)), 'R');
        
end


