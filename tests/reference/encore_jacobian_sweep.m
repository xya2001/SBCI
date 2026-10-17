% Does the reference Jacobian settle as the finite-difference step changes?
R = '/work/users/x/y/xya/sbci-reference/ConCon_Alignment/scripts';
addpath(R, fullfile(R,'utils'), fullfile(R,'tangent_basis'), ...
        fullfile(R,'kde'), fullfile(R,'interpolation'));
OUT = '/work/users/x/y/xya/encore-ref';
load(fullfile(OUT,'encore_reference.mat'), 'L', 'disp_test');

ico = icosphere(2);
lh_mesh.V = ico.V; lh_mesh.T = ico.T;
lh_grid = SphericalGrid(lh_mesh, L);

deltas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3];
J_sweep = zeros(size(lh_grid.V,1), numel(deltas));
V_sweep = zeros(size(lh_grid.V,1), 3, numel(deltas));
for k = 1:numel(deltas)
    w = SphericalWarp(lh_grid, deltas(k));
    w2 = w.compose_warp(disp_test);
    J_sweep(:,k) = w2.J;
    V_sweep(:,:,k) = w2.V;
    fprintf('delta %g: J in [%.6f, %.6f]\n', deltas(k), min(w2.J), max(w2.J));
end
for k = 1:(numel(deltas)-1)
    fprintf('  |J(%g) - J(%g)| max %.6g\n', deltas(k), deltas(k+1), ...
            max(abs(J_sweep(:,k) - J_sweep(:,k+1))));
end
save(fullfile(OUT,'jac_sweep.mat'), 'deltas', 'J_sweep', 'V_sweep', '-v7.3');
