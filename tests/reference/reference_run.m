% Reference run of rdk_smoothed_concon_compute.m, saved for comparison
% against the Python port. This is the acceptance test PORTING.md item 1 asks
% for: same inputs, same kappa, diff to float64 rounding.

REF = '/work/users/x/y/xya/sbci-reference/SBCI_Toolkit';
OUT = '/work/users/x/y/xya/matlab-reference';
addpath(fullfile(REF, 'concon_estimate'));
if ~exist(OUT, 'dir'); mkdir(OUT); end

fprintf('loading eigenpairs\n');
load(fullfile(REF, 'concon_estimate', 'EV_LBO_ds_ico4_L.mat'));  % Lambda_L, U_L
load(fullfile(REF, 'concon_estimate', 'EV_LBO_ds_ico4_R.mat'));  % Lambda_R, U_R

nndsL = 2562; nndsR = 2562; nnds = nndsL + nndsR;

% Bandwidth exactly as the reference script derives it.
nk = 10;
kappa_min = sqrt(-2 * log(0.9) / Lambda_L(end));
kappa_max = sqrt(-2 * log(0.001) / Lambda_L(200));
kappa_set = exp(linspace(log(kappa_min), log(kappa_max), nk))';
kappa = kappa_set(4);
fprintf('kappa = %.15g\n', kappa);

fprintf('building kernels\n');
t = tic;
KM_L = compute_diffusion_kernel_matrix(kappa, Lambda_L, U_L);
KM_R = compute_diffusion_kernel_matrix(kappa, Lambda_R, U_R);
fprintf('  kernels in %.1fs\n', toc(t));

fprintf('building adjacency\n');
load(fullfile(REF, 'example_data/SBCI_Individual_Subject_Outcome/mesh_intersections_ico4.mat'), ...
     'surf_in', 'surf_out', 'vtx_in', 'vtx_out');

ind = (surf_in == 0) & (surf_out == 0);
A11 = accumarray([vtx_in(ind)', vtx_out(ind)'], 1, [nndsL, nndsL]);
A11 = A11 + A11';

ind = (surf_in == 1) & (surf_out == 1);
A22 = accumarray([vtx_in(ind)', vtx_out(ind)'], 1, [nndsR, nndsR]);
A22 = A22 + A22';

ind = (surf_in == 0) & (surf_out == 1);
A12 = accumarray([vtx_in(ind)', vtx_out(ind)'], 1, [nndsL, nndsR]);
ind = (surf_in == 1) & (surf_out == 0);
A21 = accumarray([vtx_in(ind)', vtx_out(ind)'], 1, [nndsR, nndsL]);
A12 = A12 + A21';
clear A21;

fprintf('  A11 total=%.0f  A22 total=%.0f  A12 total=%.0f\n', ...
        sum(A11(:)), sum(A22(:)), sum(A12(:)));

fprintf('smoothing\n');
t = tic;
IM11 = KM_L * A11 * KM_L;
IM22 = KM_R * A22 * KM_R;
IM12 = KM_L * A12 * KM_R;
fprintf('  K A K in %.1fs\n', toc(t));

total_IM = sum(IM11(IM11 > 0)) + sum(IM22(IM22 > 0)) + (2 * sum(IM12(IM12 > 0)));
DM11 = (IM11 ./ total_IM) .* (IM11 > 0);
DM12 = (IM12 ./ total_IM) .* (IM12 > 0);
DM22 = (IM22 ./ total_IM) .* (IM22 > 0);
clear IM11 IM12 IM22;

DM = zeros(nnds, nnds);
DM(1:nndsL, 1:nndsL) = DM11;
DM(nndsL+1:end, nndsL+1:end) = DM22;
DM(1:nndsL, nndsL+1:end) = DM12;
DM(nndsL+1:end, 1:nndsL) = DM12';

fprintf('  DM sum = %.15g\n', sum(DM(:)));

% Save the kernel separately: comparing it alone isolates the kernel maths
% from the adjacency construction if the two ever disagree.
fprintf('saving\n');
save(fullfile(OUT, 'reference_kernel_L.mat'), 'KM_L', 'kappa', '-v7.3');
save(fullfile(OUT, 'reference_adjacency.mat'), 'A11', 'A22', 'A12', '-v7.3');
save(fullfile(OUT, 'reference_density.mat'), 'DM', 'kappa', '-v7.3');
fprintf('done\n');
