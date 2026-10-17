% The reference's Jacobian for one composed warp, step by step, so that
% encore_first_checks_probe.py can feed the port the same points (PORTING.md item 4).
R = '/work/users/x/y/xya/sbci-reference/ConCon_Alignment/scripts';
addpath(R, fullfile(R,'utils'), fullfile(R,'tangent_basis'), ...
        fullfile(R,'kde'), fullfile(R,'interpolation'));
OUT = '/work/users/x/y/xya/encore-ref';
load(fullfile(OUT,'encore_reference.mat'), 'L', 'disp_test', 'DELTA');

ico = icosphere(2);
lh_mesh.V = ico.V; lh_mesh.T = ico.T;
grid = SphericalGrid(lh_mesh, L);
P = size(grid.V,1);

w = SphericalWarp(grid, DELTA);
w2 = w.compose_warp(disp_test);
Vw = w2.V;

p_Dx = sphere_exp_map(grid.V, grid.e1 * DELTA);
p_Dy = sphere_exp_map(grid.V, grid.e2 * DELTA);
m_Dx = sphere_exp_map(grid.V, grid.e1 * -DELTA);
m_Dy = sphere_exp_map(grid.V, grid.e2 * -DELTA);
aabb = AABBtree(grid);
[p_Vx,p_Tx] = aabb.get_barycentric_data(p_Dx, false);
[m_Vx,m_Tx] = aabb.get_barycentric_data(m_Dx, false);
[p_Vy,p_Ty] = aabb.get_barycentric_data(p_Dy, false);
[m_Vy,m_Ty] = aabb.get_barycentric_data(m_Dy, false);

pt_px = normr([dot(p_Vx, reshape(Vw(p_Tx,1),P,3),2), dot(p_Vx, reshape(Vw(p_Tx,2),P,3),2), dot(p_Vx, reshape(Vw(p_Tx,3),P,3),2)]);
pt_mx = normr([dot(m_Vx, reshape(Vw(m_Tx,1),P,3),2), dot(m_Vx, reshape(Vw(m_Tx,2),P,3),2), dot(m_Vx, reshape(Vw(m_Tx,3),P,3),2)]);
pt_py = normr([dot(p_Vy, reshape(Vw(p_Ty,1),P,3),2), dot(p_Vy, reshape(Vw(p_Ty,2),P,3),2), dot(p_Vy, reshape(Vw(p_Ty,3),P,3),2)]);
pt_my = normr([dot(m_Vy, reshape(Vw(m_Ty,1),P,3),2), dot(m_Vy, reshape(Vw(m_Ty,2),P,3),2), dot(m_Vy, reshape(Vw(m_Ty,3),P,3),2)]);

[p_Dtt,p_Dpt] = cart_to_sphere(pt_px);
[m_Dtt,m_Dpt] = cart_to_sphere(pt_mx);
[p_Dtp,p_Dpp] = cart_to_sphere(pt_py);
[m_Dtp,m_Dpp] = cart_to_sphere(pt_my);

offset = ((-2:2).*pi);
d = 2*DELTA;
Dtt = min(p_Dtt - m_Dtt + offset,[],2,'ComparisonMethod','abs') ./ d;
Dtp = min(p_Dtp - m_Dtp + offset,[],2,'ComparisonMethod','abs') ./ d;
Dpt = min(p_Dpt - m_Dpt + offset,[],2,'ComparisonMethod','abs') ./ d;
Dpp = min(p_Dpp - m_Dpp + offset,[],2,'ComparisonMethod','abs') ./ d;
Jdet = (Dtt.*Dpp) - (Dtp.*Dpt);
[thw,~] = cart_to_sphere(Vw);
Jfinal = abs(Jdet .* sin(thw));

fprintf('max |Jfinal - w2.J| = %.3e\n', max(abs(Jfinal - w2.J)));
save(fullfile(OUT,'jac_inner.mat'), 'Vw','p_Vx','p_Tx','m_Vx','m_Tx', ...
     'p_Vy','p_Ty','m_Vy','m_Ty','pt_px','pt_mx','pt_py','pt_my', ...
     'Dtt','Dtp','Dpt','Dpp','Jdet','Jfinal','-v7.3');
fprintf('saved\n');
