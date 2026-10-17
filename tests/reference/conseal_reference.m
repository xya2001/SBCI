% conseal_reference.m
%
% Run the public ConSEAL MATLAB (MartyCole/Encore @ 27e4e7d, CPU) on a
% synthetic ico4 example and dump every intermediate quantity the Python port
% reproduces, so tests/reference/conseal_compare.py can diff them:
%
%   grid geometry, the endpoints, K, dK, the adjacency, F, Q, its derivatives,
%   one gradient and step per hemisphere, the composed warp after that step,
%   the endpoints after riding along with it, the cost trace of a short
%   registration, and a three-subject template.
%
% Usage (Longleaf):
%   module load matlab/2024b
%   matlab -batch "ENCORE='/work/users/x/y/xya/encore-upstream'; OUT='/work/users/x/y/xya/conseal-ref/conseal_reference.mat'; run('conseal_reference.m')"
%
% The mex binaries shipped in the repository (mexfiles/*.mexa64) are used
% as-is; nothing is compiled here.

if ~exist('ENCORE', 'var'); ENCORE = '/work/users/x/y/xya/encore-upstream'; end
if ~exist('OUT', 'var'); OUT = '/work/users/x/y/xya/conseal-ref/conseal_reference.mat'; end
addpath(genpath(ENCORE));

rng(23711228);

L_WARP = 6;          % warp basis order (small, so the run is quick)
H_KERNEL = 30;       % kernel truncation degree, as the paper
SIGMA = 0.005;       % bandwidth, as the paper
N = 40000;           % streamlines per synthetic subject
GPU = false;

ico_mesh = icosphere(4);
grid = SphericalGrid(ico_mesh, L_WARP, GPU);

% --- three synthetic subjects: von Mises-Fisher bundles ---------------------
subjects = cell(3, 1);
for s = 1:3
    [sp, ep] = random_connectome(N, 1, [pi/2, 0.3*s], 2, [pi/2, pi - 0.2*s]);
    n = size(sp, 2);
    hemi1 = randsample([0 1], n, true);
    hemi2 = hemi1;
    flip = randperm(n, floor(0.2 * n));
    hemi2(flip) = 1 - hemi2(flip);
    subjects{s} = struct('sp', sp.', 'ep', ep.', 'h1', hemi1, 'h2', hemi2);
end

F1 = SConcon(grid, grid, subjects{1}.sp, subjects{1}.ep, subjects{1}.h1, subjects{1}.h2, GPU);
F2 = SConcon(grid, grid, subjects{2}.sp, subjects{2}.ep, subjects{2}.h1, subjects{2}.h2, GPU);
F3 = SConcon(grid, grid, subjects{3}.sp, subjects{3}.ep, subjects{3}.h1, subjects{3}.h2, GPU);

% --- kernel ------------------------------------------------------------------
builder = SphericalHeatKernel(grid, grid, H_KERNEL, GPU);
[K, dK] = builder.compute_sigma(SIGMA);
K = double(K); dKx = double(dK.x); dKy = double(dK.y); dKz = double(dK.z);

% --- densities and derivatives -----------------------------------------------
A2 = double(F2.get_adjacency());
[Fm, Fe1, Fe2] = F2.evaluate(K, dK);
[Q2, Q2e1, Q2e2] = F2.Q_transform(K, dK);
Q1 = F1.Q_transform(K);
Fm = double(Fm); Fe1 = double(Fe1); Fe2 = double(Fe2);
Q1 = double(Q1); Q2 = double(Q2); Q2e1 = double(Q2e1); Q2e2 = double(Q2e2);

% --- one gradient step, by the class's own protected methods -----------------
% Encore's methods are protected, so replicate compute_gradient/compute_step_size
% line by line here (they are eight lines) and check them against the cost trace.
FmM = Q1 - Q2;
P = size(grid.V, 1);
idx_a = 1:P; idx_b = (P+1):(2*P);
lh_idx = [idx_a, idx_b]; rh_idx = [idx_b, idx_a];

lh_dH = local_gradient(FmM, Q2, Q2e1, Q2e2, idx_a, lh_idx, grid);
rh_dH = local_gradient(FmM, Q2, Q2e1, Q2e2, idx_b, rh_idx, grid);
lh_step = local_step(lh_dH, 0.05, 0.2);
rh_step = local_step(rh_dH, 0.05, 0.2);

lh_warp = SphericalWarp(grid);
rh_warp = SphericalWarp(grid);
lh_warp.compose(lh_step);
rh_warp.compose(rh_step);
F2s = copy(F2);
[w_sp, w_ep] = F2s.warp_connectome(lh_warp, rh_warp);
Q2_after = double(F2s.Q_transform(K));
cost_after_one_step = sum((Q1 - Q2_after).^2, 'all');

% --- a short registration and a template -------------------------------------
encore = Encore(ico_mesh, ico_mesh, L_WARP, 0.05, 6, 1e-12, GPU);
[reg_lh, reg_rh, cost] = encore.register(F1, F2, K, dK);
template = double(encore.get_template({F1; F2; F3}, K, 5));

save(OUT, '-v7.3', ...
    'ico_mesh', 'L_WARP', 'H_KERNEL', 'SIGMA', 'subjects', ...
    'K', 'dKx', 'dKy', 'dKz', 'A2', 'Fm', 'Fe1', 'Fe2', 'Q1', 'Q2', 'Q2e1', 'Q2e2', ...
    'lh_dH', 'rh_dH', 'lh_step', 'rh_step', 'w_sp', 'w_ep', 'Q2_after', 'cost_after_one_step', ...
    'cost', 'template');
lh_V = lh_warp.V; rh_V = rh_warp.V; lh_v = lh_warp.v; rh_v = rh_warp.v; lh_J = lh_warp.J; rh_J = rh_warp.J;
reg_lh_V = reg_lh.V; reg_rh_V = reg_rh.V; reg_lh_J = reg_lh.J; reg_rh_J = reg_rh.J;
grid_A = grid.A; grid_basis = grid.basis; grid_div = grid.divergence; grid_e1 = grid.e1; grid_e2 = grid.e2;
save(OUT, '-append', 'lh_V', 'rh_V', 'lh_v', 'rh_v', 'lh_J', 'rh_J', ...
    'reg_lh_V', 'reg_rh_V', 'reg_lh_J', 'reg_rh_J', 'grid_A', 'grid_basis', 'grid_div', 'grid_e1', 'grid_e2');
fprintf('wrote %s\n', OUT);

function dH = local_gradient(FmM, M, M_e1, M_e2, idx, hemi_idx, grid)
    FmM_loc = FmM(idx, hemi_idx);
    S1 = sum(FmM_loc .* M_e1(idx, hemi_idx), 2);
    S2 = sum(FmM_loc .* M_e2(idx, hemi_idx), 2);
    S3 = sum(FmM_loc .* M(idx, hemi_idx), 2);
    integrand = 2 * S1 .* grid.basis(:, :, 1) + 2 * S2 .* grid.basis(:, :, 2) + S3 .* grid.divergence;
    dH_vec = 2 * sum(integrand, 1);
    dH = squeeze(sum(dH_vec .* grid.basis, 2));
end

function step = local_step(dH, delta, threshold)
    step_norm = max(vecnorm(dH, 2, 2));
    if step_norm > threshold
        dH = dH * (threshold / step_norm);
    end
    step = -delta * dH;
end
