function [S_ini, I_ini, R_ini, S_filled, I_filled, R_filled] = fillInsideBoundary(env, Nx, Ny, Time, boundaryMap)
    envData = env.env.env;
    S_array = zeros(Nx, Ny, Time); 
    I_array = zeros(Nx, Ny, Time);
    R_array = zeros(Nx, Ny, Time);
    S_ini = 0;
    I_ini = 0;
    R_ini = 0;
    %% Fill arrays
    for i = 1:Nx
        for j = 1:Ny
            % Accumulate scalar initial values
            S_ini = S_ini + env.env.env(i,j).S;
            I_ini = I_ini + env.env.env(i,j).I;
            R_ini = R_ini + env.env.env(i,j).R;
    
            % --- Extract and convert S, I, R time series ---
            S_cell = envData(i,j).S_T_loop;
            I_cell = envData(i,j).I_T_loop;
            R_cell = envData(i,j).R_T_loop;
    
            S_vec = zeros(Time,1);
            I_vec = zeros(Time,1);
            R_vec = zeros(Time,1);
    
            for k = 1:Time
                % S
                if isempty(S_cell{k})
                    S_vec(k) = 0;   
                else
                    val = S_cell{k};
                    if numel(val) == 1
                        S_vec(k) = val;
                    else
                        S_vec(k) = val(1);
                    end
                end
                
                % I
                if isempty(I_cell{k})
                    I_vec(k) = 0;
                else
                    val = I_cell{k};
                    if numel(val) == 1
                        I_vec(k) = val;
                    else
                        I_vec(k) = val(1);
                    end
                end
                
                % R
                if isempty(R_cell{k})
                    R_vec(k) = 0;
                else
                    val = R_cell{k};
                    if numel(val) == 1
                        R_vec(k) = val;
                    else
                        R_vec(k) = val(1);
                    end
                end
            end
           % --- Store into 3D arrays ---
            S_array(i,j,:) = S_vec;
            I_array(i,j,:) = I_vec;
            R_array(i,j,:) = R_vec;
    
        end
    end

    [Nx, Ny, Time] = size(S_array);

    % Initialize outputs
    S_filled = S_array;
    I_filled = I_array;
    R_filled = R_array;

    % 4-connected neighbor kernel (up, down, left, right)
    kernel = [0 1 0; 1 0 1; 0 1 0];
    epsilon = 1e-10;

    % Mask of interior cells (not boundary)
    interiorMask = (boundaryMap == 0);

    for t = 1:Time
        % Slice for this time step
        S_slice = S_filled(:,:,t);
        I_slice = I_filled(:,:,t);
        R_slice = R_filled(:,:,t);

        % Mask of zeros in interior
        zeroMask = (S_slice == 0) & interiorMask;

        % Repeat a few passes to propagate values (usually 2-3 is enough)
        numPasses = 5;
        for pass = 1:numPasses
            % Sum of neighbors
            S_neighbors_sum = conv2(S_slice, kernel, 'same');
            I_neighbors_sum = conv2(I_slice, kernel, 'same');
            R_neighbors_sum = conv2(R_slice, kernel, 'same');

            % Count of nonzero neighbors
            S_neighbors_count = conv2((S_slice~=0).*interiorMask, kernel, 'same');
            I_neighbors_count = conv2((I_slice~=0).*interiorMask, kernel, 'same');
            R_neighbors_count = conv2((R_slice~=0).*interiorMask, kernel, 'same');

            % Only fill zeros with at least one neighbor
            fillMask = zeroMask & (S_neighbors_count > 0);

            % Fill using neighbor average
            S_slice(fillMask) = S_neighbors_sum(fillMask) ./ (S_neighbors_count(fillMask) + epsilon);
            I_slice(fillMask) = I_neighbors_sum(fillMask) ./ (I_neighbors_count(fillMask) + epsilon);
            R_slice(fillMask) = R_neighbors_sum(fillMask) ./ (R_neighbors_count(fillMask) + epsilon);

            % Stop early if nothing to fill
            if ~any(zeroMask(:))
                break
            end
        end

        % Save back
        S_filled(:,:,t) = S_slice;
        I_filled(:,:,t) = I_slice;
        R_filled(:,:,t) = R_slice;
    end
end
