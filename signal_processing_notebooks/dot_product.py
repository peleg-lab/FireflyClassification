import numpy as np
from numpy.random import randint
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, classification_report

def dotProductLit(df, literature_sequences):
    num_species = len(literature_sequences)
    df = df.sample(frac=1).reset_index(drop=True)
    ref_seqs = np.zeros((num_species,df.shape[1]-2))
    for species in range(len(literature_sequences)):
        ref_seqs[species,:len(literature_sequences[species])] = literature_sequences[species]

    # Compute predictions for each test set
    predicts = []
    scores = []

    for species in range(num_species):
        curr_spec_test = df[df['species_label'] == species]
        curr_spec_test = curr_spec_test.iloc[:,2:].to_numpy()
        prediction = []
        spec_scores = []
        for i in range(curr_spec_test.shape[0]):
            seq = np.nan_to_num(curr_spec_test[i],nan=0.0)
            areas = []
            for j in range(ref_seqs.shape[0]):
                areas.append(sum(seq*ref_seqs[j]))
            score = softmax(areas)
            if np.all(np.isclose(score, score[0])):
                pred_class = randint(0,high=num_species)
            else:
                pred_class = np.argmax(score)
            prediction.append(pred_class)
            spec_scores.append(score)
        predicts.append(prediction)
        scores.append(spec_scores)

    y_true = []
    for species in range(num_species):
        y_true.append(species*np.ones(df['species_label'].value_counts()[species],dtype=np.int8))
    y_true = [i for j in y_true for i in j]
    y_pred = [i for j in predicts for i in j]
    y_score = [i for j in scores for i in j]
    metrics = classification_report(y_true,y_pred,output_dict=True)
    conf_mat = confusion_matrix(y_true,y_pred,normalize='true')
    acc = metrics['accuracy']
    prec = metrics['macro avg']['precision']
    rec = metrics['macro avg']['recall']

    return acc, prec, rec, conf_mat, y_true, y_pred, np.stack(y_score,axis=0), metrics


def dotProductPop(df, num_iter, train_split):
    num_species = len(np.unique(df['species_label']))
    accs = []
    precs = []
    recs = []
    conf_mat = np.zeros((num_species,num_species))
    y_trues = []
    y_preds = []
    y_scores = []
    precs_sp = np.zeros((num_species, num_iter))
    recs_sp = np.zeros((num_species,num_iter))
    for iter in range(num_iter):
        df = df.sample(frac=1).reset_index(drop=True)
        test_df = df.copy()
        Y = df['species_label'].to_numpy()
        # Generate reference sequences from training sets
        ref_seqs = np.zeros((num_species,df.shape[1]-2))
        for species in np.unique(df['species_label']):
            #downsample
            inds = np.where(Y==species)[0][:np.min(df['species_label'].value_counts())]
            inds_train = inds[:int(train_split*len(inds))]
            curr_spec_train = df.iloc[inds_train,:]
            curr_spec_train = curr_spec_train.iloc[:,2:].to_numpy()
            test_df = test_df.drop(inds_train)
            ref_seqs[species,:] = np.nansum(curr_spec_train,axis=0)/curr_spec_train.shape[0]

        # Compute predictions for each test set
        predicts = []
        scores = []
        for species in np.unique(df['species_label']):
            curr_spec_test = test_df[test_df['species_label'] == species]
            curr_spec_test = curr_spec_test.iloc[:,2:].to_numpy()
            prediction = []
            spec_scores = []
            for i in range(curr_spec_test.shape[0]):
                seq = np.nan_to_num(curr_spec_test[i],nan=0.0)
                areas = []
                for j in range(ref_seqs.shape[0]):
                    areas.append(sum(seq*ref_seqs[j]))
                score = softmax(areas)
                if np.all(np.isclose(score, score[0])):
                    pred_class = randint(0,high=num_species)
                else:
                    pred_class = np.argmax(score)
                prediction.append(pred_class)
                spec_scores.append(score)
            predicts.append(prediction)
            scores.append(spec_scores)


        y_true = []
        for species in np.unique(df['species_label']):
            y_true.append(species*np.ones(test_df['species_label'].value_counts()[species],dtype=np.int8))
        y_true = [i for j in y_true for i in j]
        y_pred = [i for j in predicts for i in j]
        y_score = [i for j in scores for i in j]
        metrics = classification_report(y_true,y_pred,output_dict=True)
        conf_mat += confusion_matrix(y_true,y_pred,normalize='true')
        acc = metrics['accuracy']
        prec = metrics['macro avg']['precision']
        rec = metrics['macro avg']['recall']
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        y_trues.append(y_true)
        y_preds.append(y_pred)
        y_scores.append(y_score)
        for sp in range(num_species):
            precs_sp[sp,iter] = metrics[str(sp)]['precision']
            recs_sp[sp,iter] = metrics[str(sp)]['recall']

    return accs, precs, recs, conf_mat/num_iter, [i for j in y_trues for i in j], [i for j in y_preds for i in j], np.stack(y_scores,axis=0), precs_sp, recs_sp
