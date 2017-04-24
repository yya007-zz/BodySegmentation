function evaluation_pipeline()

a='start run'
addpath(genpath('/fs4/masi/huoy1/FS3_backup/software/full-multi-atlas/masi-fusion/src'));

output_dir = '/fs4/masi/yaoy4/Analysis/04_20_2017/';
if ~exist(output_dir)
    mkdir(output_dir)
end
testimg_seg_dir = '/fs4/masi/yaoy4/res/norandomrun_1/';
resample_seg_dir = '/fs4/masi/yaoy4/rawdata/resample_rawlabel/test';
resample_img_dir = '/fs4/masi/yaoy4/rawdata/resample_rawimg/test';

img_files = get_fnames_dir(resample_img_dir,'p*_rawimg.nii.gz');
seg_files = get_fnames_dir(resample_seg_dir,'p*_organlabel.nii.gz');

SNames={'spleen','rightkidney','leftkidney','gallbaldder',...
    'esophagus','liver','stomach','aorta',...
    'IVC','portalsplenicvein','pancreas',...
    'rightadrenalgland','leftadrenalgland',...
    'muscle','fat','other'};

Dice_all_mat = [output_dir filesep 'dice_all.mat'];

if ~exist(Dice_all_mat)
    for ii = 1:length(img_files)
        subName = sprintf('%d_vote',ii-1);
        img_file = img_files{ii};
        trueseg_file = seg_files{ii};
        testing_npy = [testimg_seg_dir filesep sprintf('%s.npy',subName)];
        
        output_sub_dir = [output_dir filesep subName];
        if ~isdir(output_sub_dir);mkdir(output_sub_dir);end;
        
        testing_mat = [output_sub_dir filesep sprintf('%s.mat',subName)];
        testing_py = [output_sub_dir filesep sprintf('%s.py',subName)];
        testing_seg = [output_sub_dir filesep sprintf('%s.nii.gz',subName)];
        
        if ~exist(testing_mat)
            if ~exist(testing_py)
                fp = fopen(testing_py,'w+');
                fprintf(fp,'import numpy as np\n');
                fprintf(fp,'import scipy.io\n');
                fprintf(fp,'dat = np.load(''%s'')\n',testing_npy);
                fprintf(fp,'dat.astype(np.int16)\n');
                fprintf(fp,'scipy.io.savemat(''%s'',{''dat'':dat})',testing_mat);
                fclose(fp);
            end
            system(sprintf('python %s',testing_py));
        end
        
        nii_true = load_untouch_nii_gz(trueseg_file);
        img_true = nii_true.img;
        if ~exist(testing_seg)
            data = load(testing_mat);
            mat = int16(data.dat);
            img_testing = mat;
            nii_testing = nii_true;
            nii_testing.img = img_testing;
            save_untouch_nii_gz(nii_testing,testing_seg);
        else         
            nii_testing = load_untouch_nii_gz(testing_seg);
            img_testing = nii_testing.img;
        end
        [meanDsc(ii,1) allDsc{ii,1} labels{ii,1} ] = dice(img_true,img_testing);
        fprintf('%d/%d\n',ii,length(img_files));
    end
    
    save(Dice_all_mat,'meanDsc','allDsc','labels');
    
else
    load(Dice_all_mat);
end


Dice_organ = nan(length(allDsc),12);
for si = 1:length(allDsc)
    for li = 1:12
        ind = find(labels{si}==li);
        if ~isempty(ind)
            Dice_organ(si,li) = allDsc{si}(ind); 
        end
    end
end

boxplot(Dice_organ);
ax.XTickLabel = SNames(1:12);
end