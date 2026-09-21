% Reference rows and a region marginal, for diffing against seed().
% seed() has only been checked against its own dense() output on a five-vertex
% fixture. This extracts the same quantities independently, at full scale.

REF = '/work/users/x/y/xya/sbci-reference/SBCI_Toolkit';
OUT = '/work/users/x/y/xya/matlab-reference';
D = fullfile(REF, 'example_data/fsaverage_label');
SUB = fullfile(REF, 'example_data/SBCI_Individual_Subject_Outcome');

tmp = load(fullfile(SUB, 'smoothed_sc_avg_0.005_ico4.mat'));
sc = tmp.sc;
dense = sc + sc';
dense = dense - diag(diag(dense));
fprintf('dense %dx%d\n', size(dense,1), size(dense,2));

% A spread of vertices: first, last, both hemispheres, and the one the docs use.
vertices = [1, 2, 1235, 2562, 2563, 4000, 5124];
rows = zeros(numel(vertices), size(dense,1));
for k = 1:numel(vertices)
    rows(k,:) = dense(vertices(k), :);
end
fprintf('extracted %d rows\n', numel(vertices));

% The region marginal: area-weighted mean over one region's vertices.
parc = load(fullfile(D, 'aparc_avg_roi_ico4.mat'));
labels = parc.labels;
tmp = load(fullfile(D, 'mapping_avg_ico4.mat'));
sbci_map = tmp.sbci_map;
areas = arrayfun(@(t) nnz(sbci_map.map(2,:)==t), unique(sbci_map.map(2,:)));

% aparc label 11 in the toolkit's numbering; the Python side picks the same
% region by name and the test checks the vertex counts agree.
region_label = 11;
mask = (labels == region_label);
fprintf('region %d has %d vertices\n', region_label, sum(mask));

weights = zeros(size(areas));
weights(mask) = areas(mask);
marginal = (dense * weights') / sum(weights);

save(fullfile(OUT, 'reference_seed.mat'), 'vertices', 'rows', 'marginal', ...
     'region_label', 'mask', 'areas', '-v7.3');
fprintf('done\n');
