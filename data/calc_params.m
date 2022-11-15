mat = readmatrix('real_data/binary_sequences_all.csv');
mat = mat(:,1:find(max(mat)==1,1,'last'));
id = readmatrix('real_data/flash_data_all.csv');
id = id(:,2);
%%
params = zeros(size(mat,1),4);
for i = 1:size(mat,1)
    seq = mat(i,1:find(~isnan(mat(i,:)),1,'last'));
    ac = accumarray(nonzeros((cumsum(~seq)+1).*seq),1);
    nac = nonzeros(ac); % list of lengths of consecutive ones
    flash_length = mean(nac); 
    num_flashes = numel(nac);
    seq = ~seq;
    zac = accumarray(nonzeros((cumsum(~seq)+1).*seq),1); % list of lengths of consecutive zeros
    gap = mean(nonzeros(zac));
    if isnan(gap)
        gap = 0;
    end
    params(i,:) = [num_flashes flash_length gap id(i)];
end
params(:,2:3) = params(:,2:3)/30;
%%
writematrix(params,'params_5species.csv');