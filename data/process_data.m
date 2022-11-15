trajs = readmatrix('xyztj.csv');
streaks = readmatrix('xyztk.csv');
trajs = horzcat(trajs,streaks(:,end));
%%
num_traj = max(trajs(:,5));
seqs = NaN*zeros(num_traj,1000);
% masterseq = zeros(1,1000000);
for i = 1:num_traj
    traj = trajs(trajs(:,5)==i,4:6);
    frame = traj(:,1);
    traj(:,3) = traj(:,3)-min(traj(:,3))+1;
    traj(:,1) = traj(:,1)-min(traj(:,1))+1;
    seq = zeros(1,traj(end,1));
    seq(traj(:,1))=1;
    seqs(i,1:length(seq)) = seq;
%     masterseq(frame(1):frame(end)) = masterseq(frame(1):frame(end))+seq;
end
% masterseq = masterseq(find(masterseq==1,1,'first'):find(masterseq==1,1,'last'));
%% drop single-bit sequences
to_drop = [];
for i = 1:size(seqs,1)
    seq = seqs(i,:);
    if sum(~isnan(seq)) == 1
        to_drop = [to_drop, i];
    end
end
seqs(to_drop,:) = [];


%% write data
fid = fopen('binary_sequences_Pknulli_Pcarolinus_Pfrontalis_Ppyralis_bw.csv', 'at');
% fprintf(fid, '\n' );
for k = 1 : size(seqs,1)
   seq = seqs(k,:);
   seq = seq(1:(find(isnan(seq),1,'first')-1));
   if length(seq)>1
       fprintf(fid, '%.1f,', seq(1:end-1));
       fprintf(fid, '%.1f\n', seq(end) );
   else
       fprintf(fid,'%.1f\n',seq);
   end
end
fclose(fid);
%%
fid = fopen('flash_data_Pknulli_Pcarolinus_Pfrontalis_Ppyralis_bw.csv','at');
% fprintf(fid, '\n' );

for k = 1 : size(seqs,1)
   fprintf(fid, '%s,', 'Bw');
   fprintf(fid, '%d\n', 4 );
end
fclose(fid);