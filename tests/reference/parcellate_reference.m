% Reference run of parcellate_sc.m, to diff against to_atlas().
% to_atlas has only ever been checked against a hand-computed three-region
% case. This runs the original on the real subject at full scale.

REF = '/work/users/x/y/xya/sbci-reference/SBCI_Toolkit';
OUT = '/work/users/x/y/xya/matlab-reference';
addpath(fullfile(REF, 'analysis'));
addpath(fullfile(REF, 'io'));
if ~exist(OUT, 'dir'); mkdir(OUT); end

D = fullfile(REF, 'example_data/fsaverage_label');
SUB = fullfile(REF, 'example_data/SBCI_Individual_Subject_Outcome');

fprintf('loading SC\n');
tmp = load(fullfile(SUB, 'smoothed_sc_avg_0.005_ico4.mat'));
sc = tmp.sc;

fprintf('loading parcellation and mapping\n');
parc = load(fullfile(D, 'aparc_avg_roi_ico4.mat'));
sbci_parc.atlas = {'aparc'};
sbci_parc.sorted_idx = parc.sorted_idx;
sbci_parc.labels = parc.labels;
sbci_parc.names = parc.names;

tmp = load(fullfile(D, 'mapping_avg_ico4.mat'));
sbci_map = tmp.sbci_map;

fprintf('  labels %d, unique %d\n', numel(sbci_parc.labels), numel(unique(sbci_parc.labels)));
fprintf('  sorted_idx %d\n', numel(sbci_parc.sorted_idx));

fprintf('parcellating\n');
t = tic;
result = parcellate_sc(sc, sbci_parc, sbci_map);
fprintf('  %.1fs, result %dx%d\n', toc(t), size(result,1), size(result,2));
fprintf('  sum %.10g, max %.10g\n', sum(result(:)), max(result(:)));

save(fullfile(OUT, 'reference_parcellation.mat'), 'result', '-v7.3');
fprintf('done\n');
