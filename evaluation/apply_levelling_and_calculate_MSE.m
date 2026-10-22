setenv('KMP_DUPLICATE_LIB_OK', 'TRUE');
python_path = ''; % path to python.exe (must be set)

% Configure MATLAB to use this Python
pyenv(Version = python_path, ExecutionMode = 'InProcess');

try
    py.list();  % Simple test
    disp('Python loaded successfully');
    disp(pyenv);  % Should show: Status: Loaded
catch e
    disp('Python failed:');
    disp(e.message);
end

%load python script directory
script_dir = ''; %path to folder containing python model inference script `afmlevel_functions.py` (must be set)

% Models: download mask_unet.pth from https://huggingface.co/Heath-AFM-Lab/afMLevel-mask-unet
% and background_unet.pth from https://huggingface.co/Heath-AFM-Lab/afMLevel-background-unet,
% and put them in script_dir (next to afmlevel_functions.py). They are loaded from MATLAB's current
% folder, hence the cd below.

if count(py.sys.path, script_dir) == 0
    insert(py.sys.path, int32(0), script_dir);
end
cd(script_dir)


base_path = ''; %path to test data folder - see below for details on how to set up test data folder

% The files from the Image and Levelled directories within the test directory of the Hugging Face dataset
% (https://huggingface.co/datasets/Heath-AFM-Lab/afMLevel-data) should be copied into the test folder called
% `All Test Data` within the base path directory. Files should be named as follows: <image_id>.tiff and
% <image_id>_levelled.tiff.

testfolder = 'All Test Data';
testpath = fullfile(base_path, testfolder);


dim = [256 256]; % will resize for consistency with model testing
preprocess = 'zscale';
autolevel_type = {'plane-line', 'iterative stdev', 'Iterative fixed high low', ...
    'Iterative fixed low', 'iterative fixed high', 'gaussian high low', ...
    'iterative gaussian low', 'iterative gaussian high', 'otsu + line', 'iterative otsu', ...
    'multi-plane otsu', 'multi-plane otsu + line', 'multi-plane edges', 'multi-plane edges + line', ...
    'plane-line MLMask', 'iterative MLMask', 'multi-plane MLMask', 'multi-plane MLMask + line', ...
    'MLBackground 1x', 'MLBackground 2x', 'MLBackground 3x'};

paper_names = {'Plane-line', 'Iterative stdev', 'Iterative fixed high/low', ...
    'Iterative fixed low', 'Iterative fixed high', 'Gaussian high/low', ...
    'Iterative Gaussian low', 'Iterative Gaussian high', 'Otsu + line', 'Iterative Otsu', ...
    'Multi-plane Otsu', 'Multi-plane Otsu + line', 'Multi-plane edges', 'Multi-plane edges + line', ...
    'Plane-line MLMask', 'Iterative MLMask', 'Multi-plane MLMask', 'Multi-plane MLMask + line', ...
    'MLBackground 1x', 'MLBackground 2x', 'MLBackground 3x'};


% Make a list of image ids
image_id_list = {};
files = dir(fullfile(testpath, '*.tiff'));


filenames = {files.name}; %alphabetically order test data
[~, idx] = sort(lower(filenames));
files = files(idx);


for k = 1:length(files)
    filename = files(k).name;
    if ~endsWith(filename, '_levelled.tiff')
        % Extract image_id from the filename
        [~, image_id, ~] = fileparts(filename);
        image_id_list{end+1} = image_id;
    end
end

% Print number of images
disp(['Number of images: ', num2str(length(image_id_list))]);

% Initialize MSE table
msetable = table(image_id_list', 'VariableNames', {'Image_ID'});
output_folder =  ''; %Input/create folder to save data into (must be set)
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Preallocate MSE storage
MSE_values = zeros(length(image_id_list), length(autolevel_type));

for k = 1:length(image_id_list)
    image_id = image_id_list{k};
    image_file = sprintf('%s.tiff', image_id);
    image_path = fullfile(testpath, image_file);
    image = imread(image_path);
    imarray = double(image);


    levelled_file = sprintf('%s_levelled.tiff', image_id);
    levelled_path = fullfile(testpath, levelled_file);
    levelled = imread(levelled_path);
    levarray = double(levelled);
    p_levarray = (levarray - mean(levarray(:))) / std(levarray(:));

    % Create figure
    if exist('lutafm.mat', 'file'), load('lutafm.mat'), else, AFM_brown = gray(256); end
    fig = figure('Name', sprintf('Results for image: %s', image_id), 'Position', [50, 50, 1500, 700]);

    subplot(4, 6, 1);
    imshow(image, []);
    title(sprintf('Original Image %d', k));
    xlabel(sprintf('%s', image_id), 'FontSize', 7, 'Interpreter', 'none');
    colormap(AFM_brown);

    subplot(4, 6, 2);
    imshow(p_levarray, [-5, 5]);
    title('Ground Truth Levelled');
    colormap(AFM_brown);

    % Process each autolevel type
    for i = 1:length(autolevel_type)
        current_type = autolevel_type{i};
        [result, imgt] = level_auto_r(imarray, 1, current_type);

        % Preprocess result and calculate MSE
        p_result = (result - mean(result(:))) / std(result(:));
        mse_value = mean((p_result(:) - p_levarray(:)).^2);

        % Store MSE
        MSE_values(k, i) = mse_value;

        % Display result
        subplot(4, 6, 2 + i);
        imshow(p_result, [-5, 5]);
        title(sprintf('%s', current_type));
        xlabel(sprintf('MSE: %.2f', mse_value), 'FontSize', 10);
        colormap(AFM_brown);

    end

    % Save the figure
    saveas(fig, fullfile(output_folder, sprintf('results_%s.png', image_id)));
    close(fig);
end

% Save MSE table to CSV
for i = 1:length(autolevel_type)
    msetable.(paper_names{i}) = MSE_values(:, i);
end
writetable(msetable, fullfile(output_folder, 'MSE_results_TEST.csv'));