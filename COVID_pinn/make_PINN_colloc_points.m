function [boundaryPts, initPts, collocPts] = make_PINN_colloc_points(boundaryMap, Nx, Ny, Nt)
    %Identify boundary and interior cells 
    [bi, bj] = find(boundaryMap); 
    filledRegion = imfill(boundaryMap, 'holes');
    interiorMask = filledRegion & ~boundaryMap;
    [ii, ij] = find(interiorMask);

    nb = numel(bi);
    ni = numel(ii);

    % Normalize indices to [0,1]
    normX = @(idx) (idx - 1) / (Nx - 1);
    normY = @(idx) (idx - 1) / (Ny - 1);
    normT = @(idx) (idx - 1) / (Nt - 1);

    % Boundary points (all timesteps)
    tvals = repelem((1:Nt)', nb, 1);
    boundaryPts = [repmat(bi, Nt, 1), repmat(bj, Nt, 1), tvals];
    boundaryPts = [normX(boundaryPts(:,1)), normY(boundaryPts(:,2)), normT(boundaryPts(:,3))];

    %initial condition points (t = 1)
    initPts = [normX(ii), normY(ij), zeros(ni,1)];
    %Interior collocation points (t = 2:Nt)
    tcol = repelem((2:Nt)', ni, 1);
    collocPts = repmat([ii, ij], Nt-1, 1);
    collocPts = [normX(collocPts(:,1)), normY(collocPts(:,2)), normT(tcol)];

    fprintf('Boundary points: %d (×%d times = %d total)\n', nb, Nt, size(boundaryPts,1));
    fprintf('Initial interior points: %d\n', size(initPts,1));
    fprintf('Interior collocation points: %d\n', size(collocPts,1));
end
