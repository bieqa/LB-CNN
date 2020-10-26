function L=LB_operator(surf)
%--------------------------------------------------------------------------
% Discretization of Laplace-Beltrami operator
%
% surf :    surf mesh
% L :       LB-operator (sparse matrix)
%
%
% The Laplace-Beltrami (LB) operator is computed by the Spectral Laplace-Beltrami 
% Wavelets [1] toolbox (http://www.bioeng.nus.edu.sg/cfa/spectrum_LBW.html)
%
% Reference:
% [1] Tan, M., Qiu, A.: Spectral Laplace-Beltrami wavelets with applications 
% in medical images. IEEE Transactions on Medical Imaging 34, 1005-1017, 2015
%
% 
% (C) 2019 Shih-Gu Huang shihgu@gmail.com
%          Moo K. Chung  mkchung@wisc.edu
%          Universtiy of Wisconsin-Madison
%
% Update history:
%     Aug. 18, 2019 created by Huang
%--------------------------------------------------------------------------

% Compute cotangent matrix
cotmtx = cotangleMesh2(surf);

% Compute mixed area
mixed = surf_mixedarea2(surf);

% Compute Laplace-Beltrami operator
L = diag(sum(cotmtx,2)./(2*mixed));
[I,J,scot] = find(cotmtx);
L = L - sparse(I,J,scot./(2*mixed(I))); 

