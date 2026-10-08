% The author's research fork of ConSEAL on one case of PORTING.md item 19, on CPU nodes, set up
% as its gpu/registration.m sets itself up: ico4, order 15, its own heat kernel (0.005, degree
% 30), Encore(grid,grid,15,0.1,1000,1e-6).
%     module load matlab/2024b
%     matlab -batch "case_name='known_007'; run('fork_case.m')"
% Nothing in the fork is changed; the inputs are cases/<case>.mat from fork_cases.py. The fork
% is the author's and is not distributed with the package. mfallback/ holds two stand-ins:
% fork_standin/bary_interp_2D_mex.m, and a copy of the fork's Concon.m whose evaluate returns
% the smoothed connectome it computes, as the committed file and the run of 6 May do (its
% working copy of 19 June returns the raw adjacency).
% The fork's folders by absolute path, and no cd into the fork: MATLAB's current folder outranks
% the path, so from inside the fork its own Concon.m would shadow the stand-in added below.
E = '/users/x/y/xya/Encore';
addpath(E); addpath(fullfile(E, 'interpolation')); addpath(fullfile(E, 'kde'));
addpath(fullfile(E, 'simulations')); addpath(fullfile(E, 'tangent_basis')); addpath(fullfile(E, 'utils'));
cd('/work/users/x/y/xya/conseal-fork-compare/cases')  % holds no .m files
% First on the path, two stand-ins: the compiled bary_interp_2D_mex crashes on these nodes at
% full size, so the same arithmetic in MATLAB (agreeing with the MEX to 1.8e-15 where it runs);
% and Concon.m with evaluate returning the smoothed connectome, as committed and as on 6 May --
% the working copy (19 June) returns the raw adjacency it smooths, one changed line.
addpath('/work/users/x/y/xya/conseal-fork-compare/mfallback');
assert(startsWith(which('Concon'), '/work/users/x/y/xya/conseal-fork-compare/mfallback/'), 'Concon resolves to %s', which('Concon'));
assert(startsWith(which('bary_interp_2D_mex'), '/work/users/x/y/xya/conseal-fork-compare/mfallback/'), 'bary_interp_2D_mex resolves to %s', which('bary_interp_2D_mex'));
fprintf('bary_interp_2D_mex is %s\nConcon is %s\n', which('bary_interp_2D_mex'), which('Concon'));
OUT = '/work/users/x/y/xya/conseal-fork-compare/cases';

% CPU nodes: no gpuDevice; Encore.register takes its CPU path when gpuDeviceCount is 0.
fprintf("GPUs visible: %d\n", gpuDeviceCount);
ICO_RESOLUTION = 4; l = 15;
ico_mesh = icosphere(ICO_RESOLUTION);
grid = SphericalGrid(ico_mesh, l);
kernel = load('/users/x/y/xya/encore_dice_scores_2/heat_kernel_0005_30.mat');
kernel = kernel.kernel;

x = load(fullfile(OUT, [case_name '.mat']));
F1 = Concon(grid,grid,x.fixed_in,x.fixed_out,x.fixed_hin,x.fixed_hout);
F2 = Concon(grid,grid,x.moving_in,x.moving_out,x.moving_hin,x.moving_hout);
fprintf('case %s: fixed %d, moving %d streamlines\n', case_name, size(x.fixed_in,1), size(x.moving_in,1));

encore = Encore(grid,grid,15,0.1,1000,1e-6);
lh_warp = SphericalWarp(grid,1e-10);
rh_warp = SphericalWarp(grid,1e-10);
tic;
[F2r, lh_warp, rh_warp, cost, cost2] = encore.register(F1, F2, kernel, lh_warp, rh_warp, kernel);
elapsed = toc;
fprintf('Total registration time: %.2f seconds\n', elapsed);

[moved_in, moved_out] = F2r.warp_connectome(lh_warp, rh_warp);
grid_V = ico_mesh.V; grid_T = ico_mesh.T; lh_V = lh_warp.V; rh_V = rh_warp.V;
save(fullfile(OUT, [case_name '_fork-cpu.mat']), 'cost', 'cost2', 'elapsed', 'moved_in', 'moved_out', ...
     'grid_V', 'grid_T', 'lh_V', 'rh_V', '-v7.3');
fprintf('FORK CASE %s SAVED\n', case_name);
