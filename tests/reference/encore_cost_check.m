% Which cost does the reference return -- the accepted step, or the rejected one?
R = '/work/users/x/y/xya/sbci-reference/ConCon_Alignment/scripts';
addpath(R, fullfile(R,'utils'), fullfile(R,'tangent_basis'), ...
        fullfile(R,'kde'), fullfile(R,'interpolation'));
OUT = '/work/users/x/y/xya/encore-ref';
load(fullfile(OUT,'encore_reference.mat'), 'L','F1','F2');

ico = icosphere(2);
lh_mesh.V = ico.V; lh_mesh.T = ico.T;
a = 0.37; Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];
rh_mesh.V = (Rz * ico.V.').'; rh_mesh.T = ico.T;

encore = Encore(lh_mesh, rh_mesh, L, 0.05, 15, 1e-8);
[result, lhw, rhw, cost] = encore.register(F1, F2);

% Recompute the cost that actually belongs to the warps it returned.
lh_grid = SphericalGrid(lh_mesh, L); rh_grid = SphericalGrid(rh_mesh, L);
A = [lh_grid.A; rh_grid.A] * [lh_grid.A; rh_grid.A].';
c = Concon(lh_grid, rh_grid, 1e-10);
Q1 = sqrt(F1 / sum(F1(:) .* A(:)));
Q2 = sqrt(F2 / sum(F2(:) .* A(:)));
img = c.evaluate_Q(Q2, lhw, rhw);
true_cost = sum((Q1(:) - img(:)).^2 .* A(:));

fprintf('returned cost            : %.12f\n', cost);
fprintf('cost of the returned warp: %.12f\n', true_cost);
fprintf('difference               : %.6e\n', abs(cost - true_cost));
