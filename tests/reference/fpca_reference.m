% Run the published ConConBasis.Fit unmodified, on a mesh basis.
FPCA = '/work/users/x/y/xya/sbci-reference/SBCI_Modeling_FPCA';
addpath('/work/users/x/y/xya/fpca-ref/patched');
addpath('/work/users/x/y/xya/tensor_toolbox');
addpath('/work/users/x/y/xya/fpca-ref');
OUT = '/work/users/x/y/xya/fpca-ref';

load(fullfile(OUT,'fpca_inputs.mat'));
nv = double(nv); N = double(N);
fprintf('inputs: Y %dx%dx%d, nv %d\n', size(Y,1), size(Y,2), size(Y,3), nv);

b0 = GridBasis(zeros(nv,3), J_block, R_block);
b1 = GridBasis(zeros(nv,3), J_block, R_block);

Ycell = {};
for i=1:N
    Ycell{i} = sparse(Y(:,:,i));
end

K = 4;
CCB = ConConBasis(b0, b1, 'K', K);
X0 = zeros(nv,3); X1 = zeros(nv,3);

fprintf('calling the published Fit...\n');
try
    [CCB, Smat, scales, objective_function, times, residual_norms] = ...
        CCB.Fit(Ycell, X0, X1, 'alpha_1', 1e-10, 'MAXITER_OUTER', 30, ...
                'MAXITER_INNER', 30, 'TOL_OUTER', 1e-3, 'TOL_INNER', 1e-3);
    fprintf("Fit returned normally\n");
    C0 = get(CCB,'C0'); C1 = get(CCB,'C1');
    save(fullfile(OUT,'fpca_reference.mat'), 'C0','C1','Smat','scales','V0_RECORD','CTILDES_RECORD', ...
         'objective_function','residual_norms','K','-v7.3');
    fprintf('scales: %s\n', mat2str(scales', 8));
    fprintf('residual norms: %s\n', mat2str(residual_norms', 8));
catch err
    fprintf('Fit FAILED: %s\n', err.message);
    fprintf('  at %s line %d\n', err.stack(1).name, err.stack(1).line);
end
