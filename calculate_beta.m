% This will calculate the beta constant as seen in the Grabcut paper.
function [beta] = calculate_beta(image)

[height, width, dim] = size(image);
n = height * width;

tri_U = zeros(n,3);
for d=1:dim
    s = image(:,:,d);
    tri_U(:,d) = s(:);
end

all = 1:n;
east = all - 1;
west = all + 1;
north= all + height;
south= all - height;
  
all = repmat(1:n,1,4); %four times.
neighbors = cat(2,east,west,north,south);
  
true_all = all(neighbors > 0 & neighbors <= n);
true_neighbors = neighbors(neighbors > 0 & neighbors <= n);

values = tri_U(true_all,:) - tri_U(true_neighbors,:); %zm - zn
values = sum(values .^ 2,2); % (zm - zn) ^ 2
values = sqrt(values);
avg = mean(values); % 1x3 vector. 

beta = 1 / (2 * avg );
disp('Made beta:');
disp(beta);

end

