classdef GridBasis < matlab.mixin.SetGet
    %GridBasis: the identity basis on a mesh, with supplied J and R.
    %
    % ConConBasis.Fit only ever asks a basis for Evaluate, InnerProduct,
    % QVInnerProduct and the properties M, J, R. The spherical spline basis
    % that the reference uses needs splinepak and Lebedev quadrature to build
    % J and R; on a mesh they are already known -- the Voronoi vertex areas
    % and the cotangent stiffness matrix -- so no such dependency is needed.
    properties
        M
        J
        R
        Verts
    end
    methods
        function obj = GridBasis(verts, J, R)
            obj.Verts = verts;
            obj.M = size(verts,1);
            obj.J = J;
            obj.R = R;
        end
        function Phi = Evaluate(obj, X)
            Phi = eye(obj.M);
        end
        function obj = InnerProduct(obj, varargin)
        end
        function obj = QVInnerProduct(obj, varargin)
        end
    end
end
