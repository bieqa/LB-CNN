function mixed = surf_mixedarea2(surf)
%--------------------------------------------------------------------------
% Computes mixed area of a point on an arbitrary surface
%
% surf :        surface mesh (surf.vertices and surf.faces)
% mixed:        mixed area, Nv x 1 vector (Nv: number of vertices)
%
%--------------------------------------------------------------------------
% Based on 'surf_mixedarea.m' written by Ming Zhen on 2013-05-10
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


%% Load parameters from byufile
% [nvertex, ntris, nconns, triloc, tris] = loadbyu(byufile);

%% Find faces in a 1-ring about a point
% I = [tris(:,1) ; tris(:,2) ; tris(:,3)];
% J = [(1:ntris)' ; (1:ntris)' ; (1:ntris)'];
Nf=size(surf.faces,1);
Nv=size(surf.vertices,1);
I=reshape(surf.faces,[],1);
J = [1:Nf  1:Nf  1:Nf ]';

%% Find voronoi area of a triangle w.r.t. a point
% area = surf_areaTris(byufile);
area = surf_areaTris2(surf);

% For each point in a triangle, find the opposite edge length
% II = [tris(:,1),tris(:,2),tris(:,3);
%       tris(:,2),tris(:,1),tris(:,3);
%       tris(:,3),tris(:,1),tris(:,2)];
% oppedge = triloc(II(:,2),:) - triloc(II(:,3),:);
II = [surf.faces;  surf.faces(:,[2 1 3]); surf.faces(:,[3 1 2])];
oppedge = surf.vertices(II(:,2),:) - surf.vertices(II(:,3),:);
oppedge = sum(oppedge.*oppedge,2);
oppedge = reshape(oppedge,Nf,3);

% For each point in a triangle, find the cotangle
% tmp1 = triloc(II(:,2),:) - triloc(II(:,1),:);
% tmp2 = triloc(II(:,3),:) - triloc(II(:,1),:);
tmp1 = surf.vertices(II(:,2),:) - surf.vertices(II(:,1),:);
tmp2 = surf.vertices(II(:,3),:) - surf.vertices(II(:,1),:);
cotangle = 0.5 * sum(tmp1.*tmp2,2) ./ area(J);
cotangle = reshape(cotangle,Nf,3);

% For each point, find the voronoi area (sparse matrix)
tmp1 = oppedge(:,2) .* cotangle(:,2) + oppedge(:,3) .* cotangle(:,3);
tmp2 = oppedge(:,1) .* cotangle(:,1) + oppedge(:,3) .* cotangle(:,3);
tmp3 = oppedge(:,2) .* cotangle(:,2) + oppedge(:,1) .* cotangle(:,1);
voronoi = [tmp1;tmp2;tmp3]; voronoi = voronoi ./ 8;


%% Find mixed area of every point
% obtuse = surf_ObtuseFace(byufile); 
obtuse = surf_ObtuseFace2(surf); 
obtuse = repmat(obtuse,3,1);
obtusetrue = obtuse(:,1);
obtusefalse = -obtusetrue + 1;

voronoi = voronoi.*obtusefalse;

triObtuseButNotPoint = I.*obtusetrue - obtuse(:,2); 
triObtuseButNotPoint = triObtuseButNotPoint~=0;
notvoronoi = area(J).*obtusetrue./2 - area(J).*triObtuseButNotPoint./4;

mixed = voronoi + notvoronoi;
% mixed = sparse(I,J,mixed,nvertex,ntris,length(I));
mixed = sparse(I,J,mixed,Nv,Nf,length(I));
mixed = sum(mixed,2); mixed = full(mixed);


end

