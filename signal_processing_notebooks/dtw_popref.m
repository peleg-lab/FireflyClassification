%% Classifies flash pattern time series against population references using dynamic time warping
% Saves predictions to csv files that are then analyzed in the DTW.ipynb
% notebook
% Requires the dtw_c.m function available at: https://www.mathworks.com/matlabcentral/fileexchange/43156-dynamic-time-warping-dtw

% !!! Before running, make sure dtw_c.m function is in path !!!

%% Load data
data = readtable('../data/real_data/flash_sequence_data.csv');
% Unpack timeseries
timeseries = data.timeseries;
for i = 1:size(timeseries,1)
    timeseries{i} = str2num(timeseries{i});
end
seqs = NaN*zeros(size(timeseries,1),max(cellfun('length',timeseries)));
for i = 1:size(timeseries,1)
    seqs(i,1:length(timeseries{i})) = timeseries{i};
end

labels = data.species_label;
params = data{:,{'num_flashes','flash_duration','ifi'}};
% Exclude sequences with only 1 flash
seqs(params(:,1)<=1,:) = [];
labels(params(:,1)<=1,:) = [];

% Load train and test indices
train_indices = csvread('train_indices.csv');
test_indices = csvread('test_indices.csv');
reshuff_indices = csvread('reshuff_indices.csv');
%% Compute DTW and classify
num_species = length(unique(labels)); % Number of species
train_split = 0.8; % Training split
test_split = 1-train_split; % Test split
    
y_trues = []; % True classes
y_preds = []; % Predicted classes
y_scores = []; % Scores

C = zeros(num_species,num_species); % Confusion matrix
num_folds = size(train_indices,1); % Number of folds
w = 100; % dtw window

seed = 1; % random seed
% reshuffle dataset
seqs = seqs(reshuff_indices,:);
labels = labels(reshuff_indices);

tic;
for fold = 1:num_folds
    train_ind = train_indices(fold,:);
    if train_ind(end) == 0
        train_ind = train_ind(1:(find(train_ind==0,1,'first')-1)); % Each training set is a slightly different size, padded with zeros
    end
    test_ind = test_indices(fold,:);
    if test_ind(end) == 0
        test_ind = test_ind(1:(find(test_ind==0,1,'first')-1));
    end
    % Build population reference set
    ref_seqs = zeros(num_species,size(seqs,2)); 
    for i = 1:num_species
        inds = train_ind(labels(train_ind)==(i-1));
        curr_spec_train = seqs(inds,:);
        ref_seqs(i,:) = nansum(curr_spec_train,1)./size(curr_spec_train,1);
    end

    % Classify
    predicts = [];
    scores = [];
    test_size = zeros(num_species,1);
    for i = 1:num_species
        prediction = [];
        spec_scores = [];
        inds = test_ind(labels(test_ind)==(i-1));%find(temp_labels==(i-1));
        curr_spec_test = seqs(inds,:);
        test_size(i) = size(curr_spec_test,1);
        for j = 1:size(curr_spec_test,1)
            seq = curr_spec_test(j,:);
            if isnan(sum(seq))
                seq = seq(1:(find(isnan(seq),1,'first')-1));
            end
            if length(seq) ~= size(ref_seqs,2)
                seq = [seq, zeros(1,size(ref_seqs,2)-length(seq))];
            end
            dtws = zeros(size(ref_seqs,1),1);
            for k = 1:size(ref_seqs,1)
                dtws(k) = dtw_c(seq,ref_seqs(k,:),w);
            end
            score = exp(-1*dtws)/sum(exp(-1*dtws));
            if length(unique(score))==1
                pred_class = randi(num_species) - 1;
            else
                [~,I]=max(score);
                pred_class = I-1;
            end
            prediction = [prediction; pred_class];
            spec_scores = [spec_scores;score'];
        end
        predicts = [predicts;prediction];
        scores = [scores;spec_scores];
    end
    y_true = [];
    for i = 1:num_species
        y_true = vertcat(y_true,(i-1).*ones(test_size(i),1));
    end

    y_true = y_true(:);
    y_pred = predicts;
    y_score = scores;
    cc = confusionmat(y_true,y_pred)./(repmat(sum(confusionmat(y_true,y_pred),2),1,num_species));
    C = C + cc;
    y_trues = [y_trues;y_true];
    y_preds = [y_preds;y_pred];
    y_scores = [y_scores;y_score];
end
C = C/num_folds;
toc;

writematrix(y_trues,'dtw_y_true.csv');
writematrix(y_preds,'dtw_y_pred.csv');
writematrix(y_scores,'dtw_y_score.csv');
