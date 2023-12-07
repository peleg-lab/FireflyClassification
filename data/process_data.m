%% Convert trajectory coordinates (xyztkj) to timeseries
% Creates .csv file containing time series, species, date, dataset label,
% temperature, and flash params

% Import list of dataset labels
opts = detectImportOptions('dataset_list.csv');
opts.DataLines = 2;
opts.VariableTypes{4} = 'char';
datasets = readtable('dataset_list.csv',opts);

dataset_folder = 'datasets/'; % directory containing datasets
addpath(dataset_folder);

fps = 30; % frame rate

species = datasets.Species;
species_label = zeros(size(species));
for sp_i = 1:length(species)
    switch species{sp_i}
        case 'B. wickershamorum'
            species_label(sp_i) = 0;
        case 'P. bethaniensis'
            species_label(sp_i) = 1;
        case 'P. carolinus'
            species_label(sp_i) = 2;
        case 'P. forresti'
            species_label(sp_i) = 3;
        case 'P. frontalis'
            species_label(sp_i) = 4;
        case 'P. knulli'
            species_label(sp_i) = 5;
        case 'P. obscurellus'
            species_label(sp_i) = 6;
    end
end
datasets.Species_label = species_label;

all_seqs = [];

% Generate cleaned timeseries from trajectories
for dataset_i = 1:length(datasets.ID)
    dataset_ID = datasets.ID{dataset_i};
    dataset_path = [dataset_folder 'xyztkj_' dataset_ID(2:end) '.csv'];
    xyztkj = readmatrix(dataset_path); % matrix containing xyz coordinates, time (seconds), streak #, trajectory #
    num_traj = max(xyztkj(:,6)); % number of trajectories
    seqs = NaN*zeros(num_traj,2000);
    for traj_i = 1:num_traj
        % extract single trajectory to convert to timeseries
        traj = xyztkj(xyztkj(:,6)==traj_i, 4:6); % columns are time (s), streak #, trajectory #
        traj(:,1) = round(traj(:,1)*fps); % convert time to frame
        traj(:,1) = traj(:,1) - min(traj(:,1)) + 1; % start from 1
        seq = zeros(1,traj(end,1));
        seq(traj(:,1)) = 1;
        if sum(seq) > 1 % only keep sequences longer than 1 bit
            
            % here we combine any flashes split by one 0 to a single flash
            
            % first convert the sequence into a string
            seq_str = sprintf('%d', seq);
            
            % then we string match occurrences of '1 0 1'
            split_flashes = strfind(seq_str, sprintf('%d', [1 0 1]));
            if ~isempty(split_flashes)
                for j = 1:length(split_flashes)
                    seq(split_flashes(j):split_flashes(j)+2) = [1 1 1];
                end
            end
            
            % first convert the sequence into a string
            seq_str = sprintf('%d', seq);
            
            % then we string match occurrences of '1 0 0 1'
            split_flashes = strfind(seq_str, sprintf('%d', [1 0 0 1]));
            if ~isempty(split_flashes)
                for j = 1:length(split_flashes)
                    seq(split_flashes(j):split_flashes(j)+3) = [1 1 1 1];
                end
            end
            
            if sum(seq) ~= length(seq) % skip single flashers
                seqs(traj_i, 1:length(seq)) = seq;
            end
        end
    end
    seqs(find(all(isnan(seqs),2)),:) = []; % remove all-nan rows
    seqs = [dataset_i*ones(size(seqs,1),1), seqs]; % append dataset ID to the left
    all_seqs = [all_seqs; seqs];
end

%% write data
fid = fopen('flash_sequence_data.csv', 'at');
% fprintf(fid, '\n' );
fprintf(fid, '%s\n', 'Dataset,species,species_label,start_time,start_temp_F,num_flashes,flash_duration,ifi,timeseries');
all_params = [];
for k = 1 : size(all_seqs,1)
    seq = all_seqs(k,2:end);
    seq = seq(1:(find(~isnan(seq),1,'last')));
    metadata = table2cell(datasets(all_seqs(k,1),1:5));
    fprintf(fid, '%s,', metadata{1});
    fprintf(fid, '%s,', metadata{2});
    fprintf(fid, '%i,', metadata{5});
    fprintf(fid, '%s,', metadata{3});
    fprintf(fid, '%s,', metadata{4});
    
    % extract params: num_flashes, flash duration, inter-flash gap
    ac = accumarray(nonzeros((cumsum(~seq)+1).*seq),1);
    nac = nonzeros(ac); % list of lengths of consecutive ones
    flash_length = mean(nac);
    num_flashes = numel(nac);
    tseq = ~seq;
    zac = accumarray(nonzeros((cumsum(~tseq)+1).*tseq),1); % list of lengths of consecutive zeros
    gap = mean(nonzeros(zac));
    if isnan(gap)
        gap = 0;
    end
    params = [num_flashes flash_length/fps gap/fps];
    all_params = [all_params; params];
    fprintf(fid, '%i,%.8f,%.8f,',params);
    fprintf(fid,'"');
    if length(seq)>1
        fprintf(fid, '%.1f,', seq(1:end-1));
        fprintf(fid, '%.1f', seq(end) );
    else
        fprintf(fid,'%.1f',seq);
    end
    fprintf(fid,'"\n');
end
fclose(fid);