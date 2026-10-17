% Reference run of the three SFC functions, for diffing against the port.
% Uses the same SC and FC the Python side reads, so any disagreement is in the
% SFC code and not in the inputs.

REF = '/work/users/x/y/xya/sbci-reference/SBCI_Toolkit';
OUT = '/work/users/x/y/xya/matlab-reference';
addpath(fullfile(REF, 'sfc'));
addpath(fullfile(REF, 'io'));
if ~exist(OUT, 'dir'); mkdir(OUT); end

SUB = fullfile(REF, 'example_data/SBCI_Individual_Subject_Outcome');

fprintf('loading SC and FC\n');
tmp = load(fullfile(SUB, 'smoothed_sc_avg_0.005_ico4.mat'));
sc = tmp.sc;
tmp = load(fullfile(SUB, 'fc_avg_ico4.mat'), 'fc');
fc = tmp.fc;
fprintf('  sc %dx%d, fc %dx%d\n', size(sc,1), size(sc,2), size(fc,1), size(fc,2));

% Both are stored as the strict upper triangle, so symmetrize once here and
% pass triangular=false, which keeps the diagonal handling unambiguous.
sc_sym = sc + sc';
fc_sym = fc + fc';
sc_sym = sc_sym - diag(diag(sc_sym));
fc_sym = fc_sym - diag(diag(fc_sym));

fprintf('global SFC\n');
t = tic; sfc_gbl = calculate_sfc_gbl(sc_sym, fc_sym); fprintf('  %.1fs\n', toc(t));

fprintf('discrete SFC (on the same vertex matrices, for a like-for-like diff)\n');
t = tic; sfc_dct = calculate_sfc_dct(sc_sym, fc_sym); fprintf('  %.1fs\n', toc(t));

fprintf('local SFC\n');
% The aparc parcellation, loaded the way the toolkit does.
parc = load(fullfile(REF, 'example_data/fsaverage_label/aparc_avg_roi_ico4.mat'));
sbci_parc.labels = parc.labels;
sbci_parc.names = parc.names;
fprintf('  labels: %d, unique %d\n', numel(sbci_parc.labels), numel(unique(sbci_parc.labels)));
t = tic; sfc_loc = calculate_sfc_loc(sc_sym, fc_sym, sbci_parc); fprintf('  %.1fs\n', toc(t));

labels_used = sbci_parc.labels;

fprintf('gbl: %d finite of %d, mean %.6f\n', sum(isfinite(sfc_gbl)), numel(sfc_gbl), mean(sfc_gbl(isfinite(sfc_gbl))));
fprintf('dct: %d finite of %d, mean %.6f\n', sum(isfinite(sfc_dct)), numel(sfc_dct), mean(sfc_dct(isfinite(sfc_dct))));
fprintf('loc: %d finite of %d, mean %.6f\n', sum(isfinite(sfc_loc)), numel(sfc_loc), mean(sfc_loc(isfinite(sfc_loc))));

save(fullfile(OUT, 'reference_sfc.mat'), 'sfc_gbl', 'sfc_dct', 'sfc_loc', 'labels_used', '-v7.3');
fprintf('done\n');
