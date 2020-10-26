% This example generates Laplace-Beltrami operator and graph Laplacian 
% on a hippocampus surface mesh.
% The discetization of LB-opertor is provided by [1,2]. 
%
% Only the adjacency matrices are saved (adjacency_LB.mat and adjacency_gL.mat).
%
% References:
% [1] Huang, S.-G., Lyu, I., Qiu, A., & Chung, M.K.: Fast Polynomial Approximation 
% of Heat Kernel Convolution on Manifolds and Its Application to Brain Sulcal and 
% Gyral Graph Pattern Analysis. IEEE Transactions on Medical Imaging, 39(6), 2201-2212, 2020.
%
% [2] Tan, M., Qiu, A.: Spectral Laplace-Beltrami wavelets with applications
% in medical images. IEEE Transactions on Medical Imaging 34, 1005-1017, 2015
%
%
% Update history:
%     Oct. 20, 2020  Created by Shih-Gu Huang
%--------------------------------------------------------------------------

%% Load hippocampus surface mesh.
load('hippocampus_l.mat')           % left hippocampus surface
% nvertex=size(surf.vertices, 1)       % number of vertices


%% Discretization of Laplace-Beltrami (LB) operator
addpath('./LB/')
% folder containing some functions modified from Spectral Laplace-Beltrami Wavelets [2] for computing LB-operator
L=LB_operator(surf);

W=diag(diag(L))-L;

save('adjacency_LB.mat','W','-v7.3')


%% Graph Laplacian
W(W~=0)=1;
save('adjacency_gL.mat','W','-v7.3')
% tempi=surf.faces;
% tempj=surf.faces(:,[2 3 1]);
% Nf=size(surf.faces,1);          % number of faces (triangles)
% W = sparse(tempi,tempj,ones(3*Nf,1));



