function out = bary_interp_2D_mex(lh_points, rh_points, lh_faces, rh_faces, data)
% The fork's bary_interp_2D_mex.cpp in MATLAB, for Longleaf's CPU and V100 nodes, where the
% compiled MEX (shipped, or rebuilt from the same source) crashes on valid input. The C++ loop
% writes new_data[i * lh_n + j], column-major, as
%     out(j, i) = sum_ab lh_points(a, i) rh_points(b, j) data(rh_faces(b, j) + 1, lh_faces(a, i) + 1)
% with zero-based int64 faces, which is W2 * data * W1' for the barycentric matrices W1 and W2.
n_lh = size(lh_points, 2);
n_rh = size(rh_points, 2);
W1 = sparse(repmat(1:n_lh, 3, 1), double(lh_faces) + 1, lh_points, n_lh, size(data, 2));
W2 = sparse(repmat(1:n_rh, 3, 1), double(rh_faces) + 1, rh_points, n_rh, size(data, 1));
out = full(W2 * data * W1.');
end
