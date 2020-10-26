function obtuse = surf_ObtuseFace2(surf)
%--------------------------------------------------------------------------
% Find faces with obtuse angles
%
% surf :        surface mesh (surf.vertices and surf.faces)
% obtuse:       Nf x 2 matrix (Nf: number of faces)
%               1st column: 1 if obtuse, 0 otherwise
%               2nd column: indexes of vertices with obtuse angle if obtuse, 0 otherwise
%
%--------------------------------------------------------------------------
% Based on 'surf_ObtuseFace.m' written by Ming Zhen on 2013-05-08
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


%% Compute Obtuse angles
% [nvertex, ntris, nconns, triloc, tris] = loadbyu(byufile);
% A = zeros(ntris,3); B = A; C = A;
% for i = 1:ntris
%     A(i,:) = triloc(tris(i,1),:);
%     B(i,:) = triloc(tris(i,2),:);
%     C(i,:) = triloc(tris(i,3),:);
% end
% obtuse = zeros(ntris,2);

Nf=size(surf.faces,1);
A = zeros(Nf,3); B = A; C = A;
for i = 1:Nf
    A(i,:) = surf.vertices(surf.faces(i,1),:);
    B(i,:) = surf.vertices(surf.faces(i,2),:);
    C(i,:) = surf.vertices(surf.faces(i,3),:);
end
obtuse = zeros(Nf,2);
obtuse(:,1) = false;

% Check point A
AB = B-A;
AC = C-A;
a = sum(AB.*AC,2); 
for i = 1:length(a)
    if a(i) < 0
        obtuse(i,1) = true;
%         obtuse(i,2) = tris(i,1);       
        obtuse(i,2) = surf.faces(i,1);
    end
end

% Check point B
BA = A-B;
BC = C-B;
b = sum(BA.*BC,2); 
for i = 1:length(b)
    if b(i) < 0
        obtuse(i,1) = true;
%         obtuse(i,2) = tris(i,2);        
        obtuse(i,2) = surf.faces(i,2);
    end
end

% Check point C
CB = B-C;
CA = A-C;
c = sum(CB.*CA,2); 
for i = 1:length(c)
    if c(i) < 0
        obtuse(i,1) = true;
%         obtuse(i,2) = tris(i,3);
        obtuse(i,2) = surf.faces(i,3);
    end
end


end