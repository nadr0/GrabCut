% Meyer, Nadro, Kuck 2015
% CS445 Computational Photography
function [ mask ] = grabcut( image, box, pos )
%   grabcut Iterative Image Segmentation.
%   Iteratively improves segmentation given simple box mask provided by
%   user, which explains what isn't foreground.

    NUM_ITERATIONS = 12;

    energy = zeros(1,5);
    
    beta  = calculate_beta(image);
    image = double(image);
    
    %h = pos(3); %not working bc probably not exact!
    %disp(h);
    %vs = find(sum(box,1));
    %h = vs(end) - vs(1) + 1;
    %disp(h);
    
    
    NUM_LABELS = 2;
    LAB_FORE = 1;
    LAB_BACK = 0;
    K = 5;
    [height, width, dimension] = size(image);
    mask = uint8(box);

%  ) ===============================================================
% 1) Color estimation using GMMs. Full covariance, K = 5 components.
%  ) ===============================================================

    %fore_X = zeros(numel(find(box)),dimension);     %each N x 3
    %back_X = zeros(numel(find(~box)),dimension);    %List of pixels.
    fore_X = {};
    back_X = {};
    for d = 1:dimension
        split = image(:,:,d);
        fore_X{d} = split(box);
        back_X{d} = split(~box);
    end
    fore_X = cell2mat(fore_X);
    back_X = cell2mat(back_X);                      % can switch back to preallocation.
    
    disp('Creating Initial GMM...');
    disp(size(fore_X));
    disp(size(back_X));
    fore_GMM = fitgmdist(fore_X,K,'RegularizationValue',0.01); %don't know if need extra value.
    back_GMM = fitgmdist(back_X,K,'RegularizationValue',0.01);
    
    disp('Created Initial GMM.');
    
%  ) ===============================================================
% 2) Setup intial values for iteration.
%  ) ===============================================================

    %k_U = zeros(height,width);          % The Cluster Number of each unknown pixel.
    k_U = zeros(numel(find(box)),1);    % Cluster number of each unknown pixel.
    alpha_U = ones(numel(find(box)),1); % The alpha values of each Unknown pixel.
    tri_U = fore_X;                     % The list of pixels of the user.
    past_U = alpha_U;                   % the previous set of mask.
   
    
%  ) ===============================================================
%  ) Iterative Minimization. 5 Stages.
%  ) ===============================================================
    converged = false;
    n = 0;
    while ~converged
        n = n + 1;
        disp('Iterating...');
        %  ) ===============================================================
        % 1) Assign GMM components to pixels. For each pixel in T_U, give
        % it a cluster center.
        %  ) ===============================================================
        k_U = assign_components(alpha_U,tri_U,fore_GMM,back_GMM);
        
        %  ) ===============================================================
        % 2) Learn GMM parameters from data. Update sigma, Mu, and Pi
        %  ) ===============================================================

        [fore_GMM, back_GMM] = learn_parameters(back_X,alpha_U,tri_U,k_U,fore_GMM,back_GMM);
        
        %  ) ===============================================================
        % 3) Create the graph and segment it.
        %  ) ===============================================================
        segclass = create_segclass(alpha_U);
        unary = create_unary(tri_U,fore_GMM,back_GMM);
        pairwise = create_pairwise(pos(4),pos(3),tri_U,beta);
        labelcost = create_labelcost();
        [labels, E, Eafter] = GCMex(segclass, single(unary), pairwise, single(labelcost),0);
        
        energy(n) = Eafter;
        %  ) ===============================================================
        % 4) Store the result and check for convergence.
        %  ) ===============================================================
        %mask(box) = labels(:);                %set the mask.
        jaja = double(~labels);
        alpha_U = jaja;              %set the unknown alpha value trimap.
        
        mask(box) = alpha_U; %remakes mask.
        figure(1), imagesc(mask); %for the shits n giggles.
        
        %  ) ===============================================================
        % 5) Display Mask.
        %  ) ===============================================================
        
        if n >= NUM_ITERATIONS
            converged = true;
        end
    end
    disp(n);
    figure(7), plot(energy);
end

% alpha_U column vector [0,1] foreground/background
% tri_U column vector of all pixels in the unknown 
function [k] = assign_components(alpha_U, tri_U, fore_GMM,back_GMM)
% return a list of the k's assigned to the U.
    %fore_K = create_nk_matrix(tri_U, fore_GMM);
    %back_K = create_nk_matrix(tri_U, back_GMM);
    
    %[~, idx_fore] = min(fore_K,[],2); %should be Nx1
    %[~, idx_back] = min(back_K,[],2); %should be Nx1
    idx_fore = cluster(fore_GMM,tri_U); %EDIT 
    idx_back = cluster(back_GMM,tri_U); %EDIT
    
    k = alpha_U .* idx_fore + (~alpha_U) .* idx_back;
end

function [distance_matrix] = create_nk_matrix(tri_U, GMM)
% this function creates a nxk distance matrix for each pixel in tri_U for
% the specified gaussian mixture model.

[N, ~, ~] = size(tri_U); % Compute size
K = 5;

A = repmat(GMM.ComponentProportion, N,1); % Store the -log(pi) value
% A = -log(A); % Compute the -log of the weighting coeff


calculated_logdetSigma = zeros(1, K); % calculate 1/2 * log(det(sigma)) into an array [ , , , ]
for i = 1:K
    calculated_logdetSigma(1,i) = 1/sqrt(det(GMM.Sigma(:,:,i)));
end
B = repmat(calculated_logdetSigma, N, 1);

% Create cell data structure to store the normalized pixel value
% Column vector that stores an array at the index

% All the pixels from tri_U in cell format
% indexing will return an array in the form [R G B]
pixel_cells = cell(N, 1);

for i = 1:N
   pixel_cells(i,1) = {tri_U(i,1:3)}; % Populate with pixel values
end

% Make it wider
% Repmat for there are 5 pixels for each cluster
pixel_cells = repmat(pixel_cells,1,K);

% one cell array line to be repmat
mean_one_line = cell(1, K);
for i = 1:K
    mean_one_line(1,i) = {GMM.mu(i,1:3)};
end

% Repmat the height to make N 
mean_pixel_cells = repmat(mean_one_line, N, 1);

normalized_pixel_cells = cell(N,1);

% Perform the subtraction to normalize the pixels
% Pixels are in row major
for i = 1:N
    for j = 1:K
        normalized_pixel_cells(i,j) = {pixel_cells{i,j} - mean_pixel_cells{i,j}};
    end
end

inverse_sigma_one_line = cell(1, K); % Calculate the inverse sigma
for i = 1:K
    % Compute the inverse sigma for each cluser
    inverse_sigma_one_line(1,i) = {inv(GMM.Sigma(:,:,i))};
end

% Stores the inverse Sigmas
inverse_sigma_cells = repmat(inverse_sigma_one_line,N,1);

% Stores the last term in the D equation  (A + B + C)
C = zeros(N,K);

% Compute C
for i = 1:N
    for j = 1:K
        C(i,j) = exp(-  (0.5) * normalized_pixel_cells{i,j} * inverse_sigma_cells{i,j} * normalized_pixel_cells{i,j}');
    end
end

% Sum up all the weights for each cluster
distance_matrix = A .* B .* C;

end

function [fore_GMM, back_GMM] = learn_parameters(back_X,alpha_U,tri_U,k_U,fore_GMM,back_GMM)
    K = 5;

    %nk = create_nk_matrix(back_X,back_GMM); %get the k's
    %[~,back_k] = min(nk,[],2);
    back_k = cluster(back_GMM,back_X);
    tot_U = cat(1,back_X,tri_U(alpha_U == 0,:));
    tot_k = cat(1,back_k,k_U(alpha_U == 0));
    %relearn the back GMM here according to tot_U and tot_k
    back_GMM = fitgmdist(tot_U,K,'Start',tot_k,'RegularizationValue',0.01);
    
    tot_U = tri_U(alpha_U == 1,:);
    tot_k = k_U(alpha_U == 1);
    %relearn the back GMM here according to tot_U and tot_k
    fore_GMM = fitgmdist(tot_U,K,'Start',tot_k,'RegularizationValue',0.01);

end

function pairwise = create_pairwise(h,w,tri_U,beta)
%copy from graph cut and then optimize using preallocated arrays of size
%4*n. See sparse(i,j,v,m,n) in documentation for details to speed up
%process.

    %h,w are height and width of box.
    h = floor(h);
    w = floor(w);
    
    [n,~] = size(tri_U); %n is size of graph.
    all = 1:n;
    east = all - 1;
    west = all + 1;
    north= all + h; % use width or height, idk. CHECk THIS OUT...
    south= all - h;
    
    all = repmat(1:n,1,4); %four times.
    neighbors = cat(2,east,west,north,south);
    
    true_all = all(neighbors > 0 & neighbors <= n);
    true_neighbors = neighbors(neighbors > 0 & neighbors <= n);
    values = tri_U(true_all,:) - tri_U(true_neighbors,:);
    values = sum(values .^2, 2); %this is now ssd of 3 components. might fail.
    values = sqrt(values);
    values = 50.0 * exp( - beta * values );
    pairwise = sparse(true_all,true_neighbors,values,n,n);
end

% Takes in tri_U, fore_GMM, back_GMM
% Returns two column vectors of the weights for each pixel
function [weights] = create_unary(tri_U,fore_GMM,back_GMM)

    weights = zeros(2,size(tri_U,1));
    weights(1,:) = -log(compute_unary(tri_U, fore_GMM)); % Set the foreground weights
    weights(2,:) = -log(compute_unary(tri_U, back_GMM)); % Set the background weights
    
end

function [final_weights] = compute_unary(tri_U,GMM)
% This function returns a column vector final_weights 
% with the weights for each pixel
distance_matrix = create_nk_matrix(tri_U,GMM);
final_weights = sum(distance_matrix,2);

end

function segclass = create_segclass(alpha_U)
    segclass = alpha_U;
end

function labelcost = create_labelcost()
    labelcost = ~eye(2);
end