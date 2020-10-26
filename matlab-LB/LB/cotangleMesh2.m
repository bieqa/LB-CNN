function cot= cotangleMesh2(surf)
%--------------------------------------------------------------------------
% Compute the sum of cotangent angles for the faces adjacent to each edge i,j
%
% surf :    surface mesh (surf.vertices and surf.faces)
% cot :     cotangent angles, Nv x Nv sparse matrix (Nv: number of vertices)
%
%--------------------------------------------------------------------------
% Based on 'cotangleMesh.m' written by Ming Zhen on 2013-05-07
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


%% Obtain parameters from byufile
% [nvertex, ntris, nconns, triloc, tris] = loadbyu(byufile);

%% Compute area of each face
% area = surf_areaTris(byufile);
area = surf_areaTris2(surf);


%% Compute index of faces adjacent to each edge (sparse)
% Works only if surface mesh is index in a consistent orientation for all triangles
% edgeAdjface = surf_edgeAdjface(byufile);
edgeAdjface = surf_edgeAdjface2(surf);


%% Compute sum of cotangent angles for each edge i,j (sparse)
[row_i, column_i, face] = find(edgeAdjface);
other = sum(surf.faces(face,:),2) - row_i - column_i; %find the index of third point

% Find the vector from third point to other 2 points
% e21 = triloc(row_i,:) - triloc(other,:);
% e23 = triloc(column_i,:) - triloc(other,:);
e21 = surf.vertices(row_i,:) - surf.vertices(other,:);
e23 = surf.vertices(column_i,:) - surf.vertices(other,:);

% Compute cot
cot1 = 0.5 * sum(e21.*e23,2) ./ area(face);
cot = sparse(row_i,column_i,cot1);
cot = cot + cot';

end
    
    
    
    
    
