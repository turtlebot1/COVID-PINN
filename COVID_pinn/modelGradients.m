function [loss, parts, grads] = modelGradients(net, p, IPmask, ICmask, BCmask, S0, I0, R0,  S_arr, I_arr, R_arr, w_pde, w_ic, w_data, w_bc)

    % % ===== Forward pass at collocation points =====
    % Extract x, y, t and convert *inside the function*
    x = dlarray(single(IPmask(:,1)'), 'CB');
    y = dlarray(single(IPmask(:,2)'), 'CB');
    t = dlarray(single(IPmask(:,3)'), 'CB');

    % Stack them
    Xc = [x; y; t];
    % Xc = dlarray(IPmask','CB');              % [3 × N]
    Yc = forward(net, Xc);                    % [4 × N] → [S L I R]
    S = Yc(1,:); L = Yc(2,:); I = Yc(3,:); R = Yc(4,:);

    % ===== Time derivatives =====
    St = dlgradient(sum(S), t);
    Lt = dlgradient(sum(L), t);
    It = dlgradient(sum(I), t);
    Rt = dlgradient(sum(R), t);

    % ===== Spatial derivatives =====
    Sx = dlgradient(sum(S), x, 'EnableHigherDerivatives', true);  
    Sy = dlgradient(sum(S), y, 'EnableHigherDerivatives', true);
    Lx = dlgradient(sum(L), x, 'EnableHigherDerivatives', true);  
    Ly = dlgradient(sum(L), y, 'EnableHigherDerivatives', true);
    Ix = dlgradient(sum(I), x, 'EnableHigherDerivatives', true);  
    Iy = dlgradient(sum(I), y, 'EnableHigherDerivatives', true);
    Rx = dlgradient(sum(R), x, 'EnableHigherDerivatives', true);  
    Ry = dlgradient(sum(R), y, 'EnableHigherDerivatives', true);

    % ===== Second derivatives (Laplacians) =====
    Sxx = dlgradient(sum(Sx), x);  Syy = dlgradient(sum(Sy), y);
    Lxx = dlgradient(sum(Lx), x);  Lyy = dlgradient(sum(Ly), y);
    Ixx = dlgradient(sum(Ix), x);  Iyy = dlgradient(sum(Iy), y);
    Rxx = dlgradient(sum(Rx), x);  Ryy = dlgradient(sum(Ry), y);

    lapS = Sxx + Syy;
    lapL = Lxx + Lyy;
    lapI = Ixx + Iyy;
    lapR = Rxx + Ryy;

    % ===== PDE residuals =====
    fS = St - ( p.eta_s.*lapS - p.theta.*S - p.phi.*S.*I );
    fL = Lt - ( p.eta_Lat.*lapL + p.phi.*S.*I - p.e.*L );
    fI = It - ( p.eta_l.*lapI + p.e.*L - p.delta.*I );
    fR = Rt - ( p.eta_r.*lapR + p.omega.*I - p.theta.*R);
    L_pde = mean(fS.^2 + fL.^2 + fI.^2 + fR.^2, 'all');

    % ===== Initial condition loss (S,I,R only; L excluded) =====
    X0 = dlarray(ICmask','CB');
    Y0 = forward(net, X0);
    S0_hat = Y0(1,:);
    % L0_hat = Y0(2,:);
    I0_hat = Y0(3,:);
    R0_hat = Y0(4,:);
    
    sumS = sum(S0_hat);
    sumI = sum(I0_hat);
    sumR = sum(R0_hat);
    L_ic = (sumS - S0).^2 + (sumI - I0).^2 + (sumR - R0).^2;

    % ===== boundary loss =============
    Xb = dlarray(BCmask', 'CB');
        Yb = forward(net, Xb);   % [3 x Nb]

    S_bc_target = dlarray(ones(size(Yb(1,:)), 'like', Yb));
    I_bc_target = dlarray(zeros(size(Yb(3,:)), 'like', Yb));
    R_bc_target = dlarray(zeros(size(Yb(4,:)), 'like', Yb));

    % Boundary loss
    L_bc = mean((Yb(1,:) - S_bc_target).^2 + ...
                (Yb(3,:) - I_bc_target).^2 + ...
                (Yb(4,:) - R_bc_target).^2, 'all');


    % ===== Supervised data loss for S,I,R only =====
    % L_data = dlarray(0,'CB');
    % if ~isempty(tIdx)
    %     for kk = 1:numel(tIdx)
    %         k = tIdx(kk);
    %         tk = single((k-1)/(size(S_arr,3)-1));
    %         Xd = dlarray([xData(:) yData(:) tk*ones(numel(xData),1,'single')]','CB');
    %         Yd = forward(net, Xd);  % [4 × M]
    %         Sd = Yd(1,:); Id = Yd(3,:); Rd = Yd(4,:);
    % 
    %         % nearest-neighbor extract from true arrays
    %         ix = max(1, min(Nx, round(xData*(Nx-1)+1)));
    %         iy = max(1, min(Ny, round(yData*(Ny-1)+1)));
    %         lin = sub2ind([Nx,Ny], ix, iy);
    % 
    %         Sg = dlarray(single(S_arr(:,:,k))); Sg = Sg(lin);
    %         Ig = dlarray(single(I_arr(:,:,k))); Ig = Ig(lin);
    %         Rg = dlarray(single(R_arr(:,:,k))); Rg = Rg(lin);
    % 
    %         L_data = L_data + mean((Sd - Sg).^2 + (Id - Ig).^2 + (Rd - Rg).^2);
    %     end
    %     L_data = L_data / numel(tIdx);
    % end

    % ===== Total loss =====
    % loss = w_pde*L_pde + w_ic*L_ic + w_data*L_data;
    loss = w_pde*L_pde + w_ic*L_ic + w_bc*L_bc;

    % ===== Gradients =====
    grads = dlgradient(loss, net.Learnables);

    % ===== Outputs for logging =====
    parts.pde  = double(gather(extractdata(L_pde)));
    parts.ic   = double(gather(extractdata(L_ic)));
    % parts.data = double(gather(extractdata(L_data)));
end
