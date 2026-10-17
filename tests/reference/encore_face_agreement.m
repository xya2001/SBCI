% The faces libigl's tree picks for offset queries at five step sizes, for
% encore_first_checks_probe.py to measure the port's agreement (PORTING.md item 4).
R = '/work/users/x/y/xya/sbci-reference/ConCon_Alignment/scripts';
addpath(R, fullfile(R,'utils'), fullfile(R,'tangent_basis'), ...
        fullfile(R,'kde'), fullfile(R,'interpolation'));
OUT = '/work/users/x/y/xya/encore-ref';
load(fullfile(OUT,'encore_reference.mat'), 'L', 'disp_test');

ico = icosphere(2);
lh_mesh.V = ico.V; lh_mesh.T = ico.T;
grid = SphericalGrid(lh_mesh, L);
P = size(grid.V,1);
aabb = AABBtree(grid);

deltas = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4];
nd = numel(deltas);
pTx = zeros(P,3,nd); pVx = zeros(P,3,nd);
mTx = zeros(P,3,nd); mVx = zeros(P,3,nd);
Jd  = zeros(P,nd);

for k = 1:nd
    d = deltas(k);
    [pv,pt] = aabb.get_barycentric_data(sphere_exp_map(grid.V, grid.e1 *  d), false);
    [mv,mt] = aabb.get_barycentric_data(sphere_exp_map(grid.V, grid.e1 * -d), false);
    pTx(:,:,k) = pt; pVx(:,:,k) = pv;
    mTx(:,:,k) = mt; mVx(:,:,k) = mv;
    w = SphericalWarp(grid, d);
    Jd(:,k) = w.compose_warp(disp_test).J;
    fprintf('delta %g done\n', d);
end
save(fullfile(OUT,'face_agree.mat'), 'deltas','pTx','pVx','mTx','mVx','Jd','-v7.3');
