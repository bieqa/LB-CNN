function area = surf_areaTris2(surf)
%--------------------------------------------------------------------------
% Calculate the areas of the faces (triangles) using Heron's formula.
%
% surf :    surface mesh (surf.vertices and surf.faces)
% area :	areas of faces, Nf x 1 vector (Nf: numver of faces)
%
%--------------------------------------------------------------------------
% Based on 'surf_areaTris.m' written by Ming Zhen on 2013-05-07
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


%% Heron's Formula
% [nvertex, ntris, nconns, triloc, tris] = loadbyu(byufile);
% A = triloc(tris(:,1),:);
% B = triloc(tris(:,2),:);
% C = triloc(tris(:,3),:);

A = surf.vertices(surf.faces(:,1),:);
B = surf.vertices(surf.faces(:,2),:);
C = surf.vertices(surf.faces(:,3),:);

a = B - C; a = sum(a.*a,2); a = sqrt(a);
b = A - C; b = sum(b.*b,2); b = sqrt(b);
c = B - A; c = sum(c.*c,2); c = sqrt(c);
s = 0.5*(a+b+c);
area = s.*(s-a).*(s-b).*(s-c);
area = sqrt(area);

end
