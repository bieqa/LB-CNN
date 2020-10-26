 function edgeAdjface = surf_edgeAdjface2(surf)
%--------------------------------------------------------------------------
% Compute the adjacent faces to edge(i,j). For every edge,
% there are 2 adjacent faces, one will be in edgeAdjface(i,j), the other
% will be in edgeAdjface(j,i). 
% Important: This function works only if the surface mesh is indexed in a
% consistent orientation for all triangles.
%
% surf :        surface mesh (surf.vertices and surf.faces)
% edgeAdjface:  Nv x Nv sparse matrix (Nv: number of vertices). 
%               Each element is the index of a face.
%
%--------------------------------------------------------------------------
% Based on 'surf_edgeAdjface.m' written by Ming Zhen on 2013-07-17
%   as part of the Spectral Laplace-Beltrami Wavelets [1] toolbox
%   (http://www.bioeng.nus.edu.sg/cfa/spectrum_LBW.html).
%
% Modified by Shih-Gu Huang on 2019-08-18
%   to replaced byu file format by surface structure data.
%
%
% Reference:
% [1] Tan, M., Qiu, A.: Spectral Laplace-Beltrami wavelets with applications 
% in medical images. IEEE Transactions on Medical Imaging 34, 1005-1017, 2015
%--------------------------------------------------------------------------


%% Load Surface Parameters
% if ischar(byufile)
%     [nvertex, ntris, nconns, triloc, tris] = loadbyu(byufile);
% else
%     tris = byufile;
%     ntris = size(tris,1);
% end

%% Compute index of faces adjacent to each edge (sparse)
% Works only if surface mesh is index in a consistent orientation for all triangles

% tempi = [tris(:,1);tris(:,2);tris(:,3)];
% tempj = [tris(:,2);tris(:,3);tris(:,1)];
% temps = [1:ntris 1:ntris 1:ntris]; temps = temps';

tempi=surf.faces;
tempj=surf.faces(:,[2 3 1]);
Nf=size(surf.faces,1);          % number of faces (triangles)
temps = [1:Nf 1:Nf 1:Nf]';

edgeAdjface = sparse(tempi,tempj,temps);


 end