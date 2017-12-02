E = csvread('./example1.dat');
col1 = E(:,1);
col2 = E(:,2);

max_ids = max(max(col1,col2));
As= sparse(col1, col2, 1, max_ids, max_ids); 
%So A is our affinity matrix that is described in step 1
A = full(As);

%D is the diagonal matrix, ith element on the diagonal is the sum of ith row of A
D = diag(sum(A, 1));
D_square_rooted = D ** (-1/2);

%Now we can calculate the Laplacian, as described in step 2
L = D_square_rooted * A * D_square_rooted;

%Find the eigenvalues (v) and the eigenvectors (S)
[v, S] = eig(L);
[sorted_eigenvalues, sorted_eigenvectors] = sort(abs(diag(S)), 'descend');

%Parameter k that corresponds to number of clusters
k = 4;
[n, _] = size(A); % Also save the size of input in a variable

% Now we take the K largest eigenvectors, as described in step 3
top_k_eigenvectors = v(:, sorted_eigenvectors(2:k+1));

% We need to normalize it before we do K-means, step 4
Y = top_k_eigenvectors ./ sqrt(sum(top_k_eigenvectors .* top_k_eigenvectors, 2));

% Now we do K-means clustering, trearing Y as point in k-dimensional space
[idx, centers] = kmeans(Y, k);

 % Plot the result for 3 clusters

 %{
 figure;
 plot (Y (idx==1, 1), Y (idx==1, 2), 'ro');
 hold on;
 plot (Y (idx==2, 1), Y (idx==2, 2), 'bs');
 hold on;
 plot (Y (idx==3, 1), Y (idx==3, 2), 'go');
 plot (centers (:, 1), centers (:, 2), 'kv', 'markersize', 10);

  % Plot the result for 2 clusters

 figure;
 plot (Y (idx==1, 1), Y (idx==1, 2), 'ro');
 hold on;
 plot (Y (idx==2, 1), Y (idx==2, 2), 'bs');
 plot (centers (:, 1), centers (:, 2), 'kv', 'markersize', 10);
  %}

% Plot the result for 4 clusters

 figure;
 plot (Y (idx==1, 1), Y (idx==1, 2), 'ro');
 hold on;
 plot (Y (idx==2, 1), Y (idx==2, 2), 'bs');
 hold on;
 plot (Y (idx==3, 1), Y (idx==3, 2), 'go');
 hold on;
 plot (Y (idx==4, 1), Y (idx==4, 2), 'ys');
 plot (centers (:, 1), centers (:, 2), 'kv', 'markersize', 10);

 
 % Finally, we assign a node to the same cluster as was assigned to the corresponding row 
output = zeros(n, k);
for i=1:k
    i_idx = idx == i;
    output(i_idx, i) = 1;
end
output;