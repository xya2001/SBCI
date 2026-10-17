% Is the reference derivative well conditioned at delta = 1e-10?
R = '/work/users/x/y/xya/sbci-reference/ConCon_Alignment/scripts';
addpath(R, fullfile(R,'utils'), fullfile(R,'tangent_basis'), ...
        fullfile(R,'kde'), fullfile(R,'interpolation'));
OUT = '/work/users/x/y/xya/encore-ref';
load(fullfile(OUT,'encore_reference.mat'), 'F_test', 'L');

ico = icosphere(2);
lh_mesh.V = ico.V; lh_mesh.T = ico.T;
a = 0.37; Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];
rh_mesh.V = (Rz * ico.V.').'; rh_mesh.T = ico.T;

lh_grid = SphericalGrid(lh_mesh, L);
rh_grid = SphericalGrid(rh_mesh, L);

deltas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3];
dQe1_sweep = zeros(size(F_test,1), size(F_test,2), numel(deltas));
dQe2_sweep = zeros(size(F_test,1), size(F_test,2), numel(deltas));
for k = 1:numel(deltas)
    c = Concon(lh_grid, rh_grid, deltas(k));
    [d1, d2] = c.get_derivative(F_test);
    dQe1_sweep(:,:,k) = d1;
    dQe2_sweep(:,:,k) = d2;
    fprintf('delta %g: |dQe1| max %.6g, |dQe2| max %.6g\n', ...
            deltas(k), max(abs(d1(:))), max(abs(d2(:))));
end

% Is the reference derivative itself stable as delta shrinks?
for k = 1:(numel(deltas)-1)
    d = dQe1_sweep(:,:,k) - dQe1_sweep(:,:,k+1);
    fprintf('  |dQe1(%g) - dQe1(%g)| max %.6g\n', deltas(k), deltas(k+1), max(abs(d(:))));
end

save(fullfile(OUT,'delta_sweep.mat'), 'deltas', 'dQe1_sweep', 'dQe2_sweep', '-v7.3');
fprintf('saved\n');
