% Read in original csvs
seqs = readmatrix('binary_sequences_all.csv');
labels = readmatrix('flash_data_all.csv','range',[2 2]);
% Remove all pyralis data and relabel Bw as 3 not 4
seqs(labels==3,:) = [];
labels(labels==3,:) = [];
labels(labels==4,:) = 3;
% write binary sequences
fid = fopen('binary_sequences_4.csv', 'at');
for k = 1 : size(seqs,1)
    
    seq = seqs(k,:);
    if isnan(sum(seq))
        seq = seq(1:(find(isnan(seq),1,'first')-1));
    end
    if length(seq)>1
        fprintf(fid, '%.1f,', seq(1:end-1));
        fprintf(fid, '%.1f\n', seq(end) );
    else
        fprintf(fid,'%.1f\n',seq);
    end
end
fclose(fid);
% write labels
fid = fopen('flash_data_4.csv','at');
for k = 1 : size(seqs,1)
    if labels(k) == 0
        fprintf(fid, '%s,', 'P. knulli');
        fprintf(fid, '%d\n', 0 );
    elseif labels(k) == 1
        fprintf(fid, '%s,', 'P. frontalis');
        fprintf(fid, '%d\n', 1 );
    elseif labels(k) == 2
        fprintf(fid, '%s,', 'P. carolinus');
        fprintf(fid, '%d\n', 2 );
    elseif labels(k) == 3
        fprintf(fid, '%s,', 'Bw');
        fprintf(fid, '%d\n', 3 );
    end
end
fclose(fid);