function flipped = fold_flip(V, T, V0)
% Triangles of the warped vertices V whose orientation on the sphere differs from the
% unwarped mesh V0's: the folded ones. T is the one-based triangulation both share. Used by
% the guarded copy of the fork's Encore.register (fork_case.m, guard = true), which refuses a
% hemisphere's step that would fold, as the package's velocity field does.
flipped = sign(orientation(V, T)) ~= sign(orientation(V0, T));
end

function s = orientation(V, T)
a = V(T(:, 1), :); b = V(T(:, 2), :); c = V(T(:, 3), :);
s = sum(cross(b - a, c - a, 2) .* (a + b + c), 2);
end
