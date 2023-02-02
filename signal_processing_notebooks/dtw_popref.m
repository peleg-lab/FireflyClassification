%% Classifies flash pattern time series against population references using dynamic time warping
% Saves predictions to csv files that are then analyzed in the DTW.ipynb
% notebook
% Requires the dtw_c.m function available at: https://www.mathworks.com/matlabcentral/fileexchange/43156-dynamic-time-warping-dtw

% !!! Before running, make sure dtw_c.m function is in path !!!

%% Load data
seqs = readmatrix('../data/real_data/binary_sequences_7.csv');
labels = readmatrix('../data/real_data/flash_data_7.csv','range',[2 2]);
params = readmatrix('../data/params_7species.csv');
% Exclude sequences with only 1 flash
seqs(params(:,1)<=1,:) = [];
labels(params(:,1)<=1,:) = [];

%% Compute DTW and classify
num_species = length(unique(labels)); % Number of species
train_split = 0.8; % Training split
test_split = 1-train_split; % Test split
    
y_trues = []; % True classes
y_preds = []; % Predicted classes
y_scores = []; % Scores

C = zeros(num_species,num_species); % Confusion matrix
num_iter = 100; % Number of iterations with reshuffled data
w = 100; % dtw window

seed = 1; % random seed

tic;
for iter = 1:num_iter
    rng(seed+1);
    
    % Build population reference set
    reshuff = randperm(size(seqs,1));
    temp_seq = seqs(reshuff,:);
    temp_labels = labels(reshuff,:);
    freqs = [sum(temp_labels==0),sum(temp_labels==1),sum(temp_labels==2),sum(temp_labels==3)];
    smallest_size = min(freqs); % size of smallest species dataset
    ref_seqs = zeros(num_species,size(temp_seq,2));
    for i = 1:num_species
        inds = find(temp_labels==(i-1));
        curr_spec = temp_seq(inds(1:smallest_size),:);
        curr_spec_train = curr_spec(1:round(train_split*smallest_size),:);
        ref_seqs(i,:) = nansum(curr_spec_train,1)./size(curr_spec_train,1);
    end
    
    % Classify
    predicts = [];
    scores = [];
    test_size = zeros(num_species,1);
    for i = 1:num_species
        prediction = [];
        spec_scores = [];
        inds = find(temp_labels==(i-1));
        curr_spec = temp_seq(inds,:);
        curr_spec_test = curr_spec(round(train_split*smallest_size)+1:end,:);
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
C = C/num_iter;
toc;

writematrix(y_trues,'dtw_y_true.csv');
writematrix(y_preds,'dtw_y_pred.csv');
writematrix(y_scores,'dtw_y_score.csv');
