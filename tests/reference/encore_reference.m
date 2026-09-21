% Reference run of ENCORE, dumping every intermediate so a Python port can be
% checked stage by stage rather than only end to end.
%
% A level-2 icosphere (162 vertices per hemisphere) keeps the product mesh at
% 324x324, small enough to save whole and large enough to catch index errors.
% The right hemisphere is rotated so the two are not interchangeable.

R = '/work/users/x/y/xya/sbci-reference/ConCon_Alignment/scripts';
addpath(R, fullfile(R,'utils'), fullfile(R,'tangent_basis'), ...
        fullfile(R,'kde'), fullfile(R,'interpolation'));
OUT = '/work/users/x/y/xya/encore-ref';

L = 3;              % spherical harmonic order for the tangent basis
DELTA = 1e-10;      % finite-difference step used by Concon/SphericalWarp
rng(20260920);

%% ---- meshes -------------------------------------------------------------
ico = icosphere(2);
lh_mesh.V = ico.V;
lh_mesh.T = ico.T;

% rotate the right hemisphere so lh and rh are distinguishable
a = 0.37;
Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];
rh_mesh.V = (Rz * ico.V.').';
rh_mesh.T = ico.T;

fprintf('mesh: %d vertices, %d faces per hemisphere\n', size(ico.V,1), size(ico.T,1));

%% ---- SphericalGrid ------------------------------------------------------
lh_grid = SphericalGrid(lh_mesh, L);
rh_grid = SphericalGrid(rh_mesh, L);
P = size(lh_grid.V,1);

grid_lh_V = lh_grid.V;      grid_lh_T = lh_grid.T;
grid_lh_theta = lh_grid.theta;  grid_lh_phi = lh_grid.phi;
grid_lh_A = lh_grid.A;      grid_lh_basis = lh_grid.basis;
grid_lh_lap = lh_grid.laplacian;
grid_lh_e1 = lh_grid.e1;    grid_lh_e2 = lh_grid.e2;
grid_rh_V = rh_grid.V;      grid_rh_A = rh_grid.A;
grid_rh_basis = rh_grid.basis;  grid_rh_lap = rh_grid.laplacian;
grid_rh_e1 = rh_grid.e1;    grid_rh_e2 = rh_grid.e2;
fprintf('grid: A sums to %.12f (4*pi = %.12f), basis %s\n', ...
        sum(lh_grid.A), 4*pi, mat2str(size(lh_grid.basis)));

%% ---- exp / log maps -----------------------------------------------------
gamma_test = 0.3 * (lh_grid.e1 .* sin(lh_grid.theta) + lh_grid.e2 .* cos(lh_grid.phi));
exp_test = sphere_exp_map(lh_grid.V, gamma_test);
log_test = sphere_log_map(lh_grid.V, exp_test);

%% ---- AABB barycentric query --------------------------------------------
query_pts = normr(lh_grid.V + 0.05 * gamma_test);
aabb = AABBtree(lh_grid);
[bary_V_false, bary_T_false] = aabb.get_barycentric_data(query_pts, false);
[bary_V_true,  bary_T_true ] = aabb.get_barycentric_data(query_pts, true);

%% ---- bary_interp_2D -----------------------------------------------------
F_test = rand(2*P, 2*P);
F_test = (F_test + F_test.') / 2;
F_test = F_test - diag(diag(F_test));

concon = Concon(lh_grid, rh_grid, DELTA);
[dQe1, dQe2] = concon.get_derivative(F_test);

%% ---- SphericalWarp ------------------------------------------------------
warp = SphericalWarp(lh_grid, DELTA);
warp_J_identity = warp.J;
disp_test = 0.02 * [sin(3*lh_grid.theta), cos(2*lh_grid.phi)];
warp2 = warp.compose_warp(disp_test);
warp_V_composed = warp2.V;
warp_J_composed = warp2.J;

rh_warp = SphericalWarp(rh_grid, DELTA);
rh_warp2 = rh_warp.compose_warp(0.02 * [cos(2*rh_grid.theta), sin(3*rh_grid.phi)]);

%% ---- Concon evaluate ----------------------------------------------------
A_prod = [lh_grid.A; rh_grid.A] * [lh_grid.A; rh_grid.A].';
Q_test = sqrt(F_test / sum(F_test(:) .* A_prod(:)));
evalQ_test = concon.evaluate_Q(Q_test, warp2, rh_warp2);
evalF_test = concon.evaluate(F_test, warp2, rh_warp2);

%% ---- Encore: register and template --------------------------------------
encore = Encore(lh_mesh, rh_mesh, L, 0.05, 15, 1e-8);

F1 = rand(2*P, 2*P); F1 = (F1 + F1.')/2; F1 = F1 - diag(diag(F1));
F2 = rand(2*P, 2*P); F2 = (F2 + F2.')/2; F2 = F2 - diag(diag(F2));
F3 = rand(2*P, 2*P); F3 = (F3 + F3.')/2; F3 = F3 - diag(diag(F3));

[reg_result, reg_lh_warp, reg_rh_warp, reg_cost] = encore.register(F1, F2);
reg_lh_V = reg_lh_warp.V; reg_lh_J = reg_lh_warp.J;
reg_rh_V = reg_rh_warp.V; reg_rh_J = reg_rh_warp.J;
fprintf('register: final cost %.10g\n', reg_cost);

Fs = cat(3, F1, F2, F3);
template = encore.get_template(Fs, 5);
fprintf('template: sum %.10g\n', sum(template(:)));

%% ---- save ---------------------------------------------------------------
save(fullfile(OUT, 'encore_reference.mat'), ...
     'P','L','DELTA', ...
     'grid_lh_V','grid_lh_T','grid_lh_theta','grid_lh_phi','grid_lh_A', ...
     'grid_lh_basis','grid_lh_lap','grid_lh_e1','grid_lh_e2', ...
     'grid_rh_V','grid_rh_A','grid_rh_basis','grid_rh_lap','grid_rh_e1','grid_rh_e2', ...
     'gamma_test','exp_test','log_test', ...
     'query_pts','bary_V_false','bary_T_false','bary_V_true','bary_T_true', ...
     'F_test','dQe1','dQe2', ...
     'warp_J_identity','disp_test','warp_V_composed','warp_J_composed', ...
     'Q_test','evalQ_test','evalF_test', ...
     'F1','F2','F3','reg_result','reg_cost','reg_lh_V','reg_lh_J','reg_rh_V','reg_rh_J', ...
     'template', '-v7.3');
fprintf('wrote encore_reference.mat\n');
