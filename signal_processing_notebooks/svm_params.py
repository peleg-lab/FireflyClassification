import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

def svmLit(df, params_lit):
    num_species = len(params_lit)

    df = df.sample(frac=1).reset_index(drop=True)
    X_test = df.iloc[:,2:5].to_numpy()
    y_test = df['species_label'].to_numpy()

    X_train = params_lit.to_numpy()
    y_train = np.array([0,1,2])

    rbf = OneVsRestClassifier(svm.SVC(kernel="rbf", gamma=1, C=1)).fit(X_train, y_train)
    y_score = rbf.decision_function(X_test)

    rbf_pred = rbf.predict(X_test)
    metrics = classification_report(y_test,rbf_pred,output_dict=True)
    conf_mat = confusion_matrix(y_test,rbf_pred,normalize='true')
    acc = metrics['accuracy']
    prec = metrics['macro avg']['precision']
    rec = metrics['macro avg']['recall']

    y_score = np.stack(y_score,axis=0)
    return acc, prec, rec, conf_mat, y_test, rbf_pred, y_score, metrics

def svmPop(df, num_iter, train_split):
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
        X = df.iloc[:,2:5].to_numpy()
        Y = df['species_label'].to_numpy()
        smallest_set = np.min(df['species_label'].value_counts())
        train_inds = np.hstack((np.where(Y==0)[0][:int(train_split*smallest_set)],
               np.where(Y==1)[0][:int(train_split*smallest_set)],
               np.where(Y==2)[0][:int(train_split*smallest_set)],
               np.where(Y==3)[0][:int(train_split*smallest_set)]))
        X_train = X[train_inds]
        y_train = Y[train_inds]
        X_test = np.delete(X,train_inds,0)
        y_test = np.delete(Y,train_inds,0)

        rbf = OneVsRestClassifier(svm.SVC(kernel="rbf", gamma=1, C=1)).fit(X_train, y_train)
        y_score = rbf.decision_function(X_test)

        rbf_pred = rbf.predict(X_test)
        metrics = classification_report(y_test,rbf_pred,output_dict=True)
        conf_mat += confusion_matrix(y_test,rbf_pred,normalize='true')
        acc = metrics['accuracy']
        prec = metrics['macro avg']['precision']
        rec = metrics['macro avg']['recall']
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        y_trues.append(y_test)
        y_preds.append(rbf_pred)
        y_scores.append(y_score)
        for sp in range(num_species):
            precs_sp[sp,iter] = metrics[str(sp)]['precision']
            recs_sp[sp,iter] = metrics[str(sp)]['recall']

    return accs, precs, recs, conf_mat/num_iter, [i for j in y_trues for i in j], [i for j in y_preds for i in j], np.stack(y_scores,axis=0), precs_sp, recs_sp
