% The overlap of two subjects' bundles by the MATLAB the evaluations used, on the inputs
% overlap_inputs.py wrote (the package's own ico4 grids):
%   get_dice_score.m (the author's fork) and get_overlap_coefficient.m (the public repository at
%   27e4e7d), vertex-wise, at three thresholds -> vertex_overlap.mat;
%   get_connectivity_dice.m (the author's evaluation folder), connection-wise, as the ConSEAL
%   paper's evaluation calls it (threshold 0; the overlap coefficient, then Dice)
%   -> connection_overlap.mat.
% Nothing in those folders is changed. They are added by absolute path, never made current
% (MATLAB's current folder outranks the path), the fork's AABBtree and get_dice_score first.
%     module load matlab/2024b
%     matlab -batch "run('tests/reference/overlap_reference.m')"
EVALUATION = '/users/x/y/xya/encore_dice_scores_2';     % get_connectivity_dice.m
PUBLIC = '/work/users/x/y/xya/encore-upstream/analysis'; % get_overlap_coefficient.m
FORK = '/users/x/y/xya/Encore';                         % get_dice_score.m, AABBtree and its MEX
OUT = '/work/users/x/y/xya/matlab-reference/overlap';
addpath(EVALUATION); addpath(PUBLIC);
addpath(fullfile(FORK, 'analysis')); addpath(FORK); addpath(fullfile(FORK, 'interpolation'));
cd(OUT)
assert(startsWith(which('AABBtree'), [FORK '/']), 'AABBtree resolves to %s', which('AABBtree'));
assert(startsWith(which('get_dice_score'), [FORK '/']), 'get_dice_score resolves to %s', which('get_dice_score'));
assert(startsWith(which('get_overlap_coefficient'), [PUBLIC '/']), 'get_overlap_coefficient resolves to %s', which('get_overlap_coefficient'));
assert(startsWith(which('get_connectivity_dice'), [EVALUATION '/']), 'get_connectivity_dice resolves to %s', which('get_connectivity_dice'));
in = load('overlap_inputs.mat');
lh_grid.V = in.lh_V; lh_grid.T = double(in.lh_T);
rh_grid.V = in.rh_V; rh_grid.T = double(in.rh_T);
X = in.X; Y = in.Y;
X.st_hemi = X.st_hemi(:); X.en_hemi = X.en_hemi(:); Y.st_hemi = Y.st_hemi(:); Y.en_hemi = Y.en_hemi(:);

thresholds = [0, 1e-6, 1e-4];
dice = zeros(size(thresholds)); overlap = zeros(size(thresholds));
for k = 1:numel(thresholds)
    [dice(k), x_mass, y_mass] = get_dice_score(X, Y, lh_grid, rh_grid, thresholds(k));
    overlap(k) = get_overlap_coefficient(X, Y, lh_grid, rh_grid, thresholds(k));
    fprintf('vertex-wise, threshold %g: Dice %.12f, overlap %.12f\n', thresholds(k), dice(k), overlap(k));
end
save('vertex_overlap.mat', 'thresholds', 'dice', 'overlap', 'x_mass', 'y_mass');

[overlap, Ax, Ay] = get_connectivity_dice(X, Y, lh_grid, rh_grid, 0, false);
dice = get_connectivity_dice(X, Y, lh_grid, rh_grid, 0, true);
nx = nnz(Ax > 0); ny = nnz(Ay > 0); shared = nnz((Ax > 0) & (Ay > 0));
fprintf('connection-wise: %d and %d pairs, %d shared; overlap %.12f, Dice %.12f\n', nx, ny, shared, overlap, dice);
save('connection_overlap.mat', 'overlap', 'dice', 'nx', 'ny', 'shared');
