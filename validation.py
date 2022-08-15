import csv
from fileinput import filename
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.metrics import accuracy_score, f1_score,average_precision_score,roc_auc_score,precision_recall_curve,confusion_matrix,roc_curve,auc
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.utils import resample
from sklearn import preprocessing,svm
from imblearn.under_sampling import RandomUnderSampler

netID = '2'
method = 'n2v'
# emb = 'net1_500_f3.txt'
# emb = 'net1_2000_2000_1e-3_1e-4_0.05_0.95.txt'
# emb = 'net1_1700_2000_1e-3_1e-4_0.1_0.9.txt'
# emb = 'deepNF_MDA_arch_net13_1600-500-1600_features_Net13.pckl'
# emb = 'N2V_embedding13_500.txt'
# emb = 'net1_1700_4000_1e-3_1e-4_0.1_0.9.txt'
# emb = 'net1_2000_2000_1e-3_1e-4_0.05_0.95.txt'
# 2
# emb = 'net1_1700_2000_1e-3_1e-4_0.1_0.9.txt'
# emb = 'net1_2000_2000_1e-3_1e-4_0.05_0.95.txt'
# emb = 'Combine_Net13_f1\'+f2\'.txt'#
emb = 'deepNF_MDA_arch_1000-500-1000_features.pckl'
emb = 'deepNF_MDA_arch_1000-500-10001_features.pckl'
# emb = 'N2V_embedding12_500.txt'
# emb = 'Mashup_Net13_500.mat'
# filePath = '../data/y1'+str(netID)+'/' 
filePath = './combined_embedding/'
# filePath = './Net1'+str(netID)+'/'
fileName = './validation/PPI_eQTL_Cortex_deepnf.csv'


def readNetworks(filepath):
    network = []
    for line in open(filepath):
        line = line.strip()
        temp = list(map(float,line.split("\t")))
        network.append(temp)
    return np.array(network)


def readTargets(id):
    filepath = '../data/y1'+str(netID)+'/y_'+str(id)+".txt"
    y_targets = []
    for line in open(filepath).readlines():
        line = line.strip()
        y_targets.append(int(line))
    
    return(np.array(y_targets))


def scale_emb(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    emb_scaled = min_max_scaler.fit_transform(data)
    return emb_scaled


def load_embedding(emb):
    # merge_emb = readNetworks(filePath + emb)
    # merge_emb = scale_emb(merge_emb)

    data = pickle.load(open(filePath + emb,'rb'))
    merge_emb = np.array([[float(v) for v in line] for line in data])
    
    # data = sio.loadmat(filePath + emb)
    # merge_emb = np.array(np.mat(data['x']).T)
    print(merge_emb.shape)
    return merge_emb


def ml_split(y):
    """Split annotations"""
    kf = KFold(n_splits=5, shuffle=True)
    splits = []
    for t_idx, v_idx in kf.split(y):
        splits.append((t_idx, v_idx))

    return splits


def evaluate_performance(y_true, y_prob, y_pred):
    """Evaluate performance"""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    f1 = 2*precision*recall / (precision+recall)
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    print('tn = {}, fp = {}'.format(tn, fp))
    print('fn = {}, tp = {}'.format(fn, tp))
    print('acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}'.format(accuracy, precision, recall, f1, roc_auc, aupr))

    perf = dict()    
    perf["aupr"] = aupr
    # Computes RECALL
    perf["recall"] = recall
    # Computes PRECISION
    perf["precision"] = precision
    # Computes accuracy
    perf['acc'] = accuracy
    # Computes F1-score
    perf["f1"] = f1
    # Computes AUC
    perf["auc"] = roc_auc
    return perf


def Sample(label,symbol_all):
    pos_index, neg_index = [i for i, j in enumerate(label) if j == 1], [i for i, j in enumerate(label) if j ==0]
    neg_index_choice = np.random.choice(neg_index, len(pos_index),replace=False)
    all_index = np.concatenate([pos_index, neg_index_choice], axis = 0)
    if len(all_index) > 0:
        label = label[all_index]
        symbol_final = symbol_all[all_index]
        print(len(label),len(symbol_final))
    else:
        symbol_final = symbol_all
        label = label
    return symbol_final,label


def cross_validation(X, y, n_trials=5, ker='rbf'):
    """Perform model selection via 5-fold cross validation"""

    # range of hyperparameters
    C_range = np.arange(1, 3, 1)
    if ker == 'rbf':
        gamma_range = 0.001*np.arange(56,66,2)
    elif ker == 'lin':
        gamma_range = [0]
    else:
        print ("### Wrong kernel.")


    # performance measures
    AUPR = []
    AUC = []
    F1 = []
    ACC = []
    RECALL = []
    PRECISION = []

    # shuffle and split training and test sets
    trials = ShuffleSplit(n_splits=n_trials, test_size=0.2, random_state=None)
    ss = trials.split(X)
    trial_splits = []
    for train_idx, test_idx in ss:
        trial_splits.append((train_idx, test_idx))

    it = 0
    for jj in range(0, n_trials):
        train_idx = trial_splits[jj][0]
        test_idx = trial_splits[jj][1]
        it += 1
        y_test  = y[test_idx]
        X_test = X[test_idx, :]
        y_train  = y[train_idx]
        X_train = X[train_idx, :]
        print ("### [Trial %d] Perfom cross validation...." % (it))
        print ("Train samples=%d; #Test samples=%d" % (y_train.shape[0], y_test.shape[0]))
        # setup for neasted cross-validation
        splits = ml_split(y_train)

        # parameter fitting
        C_opt = 1
        gamma_opt = 0.04
        max_aupr = 0
        for C in C_range:
            for gamma in gamma_range:
                cv_results = []
                for train, valid in splits:
                    clf = svm.SVC(kernel = ker,C = C, gamma = gamma, random_state=123, probability=True)
                    X_train_t =X_train[train]
                    X_train_v = X_train[valid]
                    y_train_t = y_train[train]
                    y_train_v = y_train[valid]
                    clf.fit(X_train_t, y_train_t)
                    y_score_valid = np.zeros(y_train_v.shape, dtype=float)
                    y_pred_valid =  np.zeros_like(y_train_v)
                    y_score_valid = clf.predict_proba(X_train_v)
                    y_pred_valid = clf.predict(X_train_v)
                    try:
                        perf_cv = evaluate_performance(y_train_v,
                                                        y_score_valid[:,1],
                                                        y_pred_valid)
                        cv_results.append(perf_cv['aupr'])
                    except:
                        continue
                cv_aupr = np.median(cv_results)
                print ("### gamma = %0.3f, AUPR = %0.3f" % (gamma, cv_aupr))
                if cv_aupr >= max_aupr:
                    C_opt = C
                    gamma_opt = gamma
                    max_aupr = cv_aupr
        print("### Optimal parameters: ")
        print("C_opt = %0.3f, gamma_opt = %0.3f" % (C_opt, gamma_opt))
        # print("### Train dataset: AUPR = %0.3f" % (max_aupr))
        print("### Using full training data...")
        clf = svm.SVC(kernel = ker,C = C_opt, gamma = gamma_opt, random_state=123, probability=True)
        clf.fit(X_train, y_train)

        # Compute performance on test set
        print("### Testing...")
        y_score_valid = np.zeros(y_test.shape, dtype=float)
        y_pred_valid =  np.zeros_like(y_test)
        y_score = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        y_pred = np.around(y_pred,0).astype(int)
        try:
            perf_trial = evaluate_performance(y_test, y_score[:,1], y_pred)
            AUPR.append(perf_trial['aupr'])
            F1.append(perf_trial['f1'])
            ACC.append(perf_trial['acc'])
            AUC.append(perf_trial['auc'])
            RECALL.append(perf_trial['recall'])
            PRECISION.append(perf_trial['precision'])
            print("### Test dataset: aupr= %0.3f, auc = %0.3f, acc = %0.3f" % (perf_trial['aupr'], perf_trial['auc'], perf_trial['acc']))
        except:
            continue

    perf = dict()
    perf['aupr'] = np.average(AUPR)
    perf['f1'] = np.average(F1)
    perf['acc'] = np.average(ACC)
    perf['auc'] = np.average(AUC)
    perf['recall'] = np.average(RECALL)
    perf['precision'] = np.average(PRECISION)


    return perf


if __name__ == '__main__':
    labels = ['diseaseID','aupr','auc','f1','recall','precision','acc','iter1','iter2',
            'dim1','dim2','upper','lower','lr1','lr2','batch_size','geneNum']

    DiseaseID = ['C0588008','C0011573','C0024517',
                'C0853193','C0085159','C0003431']#'C0005586',
    emb_vals = load_embedding(emb)
    dict_arr = []
    i = 0
    for id in DiseaseID:
        tmp = {}
        y_vals = readTargets(id)
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(emb_vals, y_vals)
        perf = cross_validation(X_resampled, y_resampled, n_trials=5)
        tmp['aupr'] = perf['aupr']
        tmp['auc'] = perf['auc']
        tmp['f1'] = perf['f1']
        tmp['recall'] = perf['recall']
        tmp['precision'] = perf['precision']
        tmp['acc'] = perf['acc']
        tmp['diseaseID'] = id
        tmp['iter1'] = '1700'
        tmp['iter2'] = '4000'
        tmp['dim1'] = '1600'
        tmp['dim2'] = '500'
        tmp['upper'] = '0.05'
        tmp['lower'] = '0.05'
        tmp['lr1'] = '1e-3'
        tmp['lr2'] = '1e-2'
        tmp['batch_size'] = '64' 
        tmp['geneNum'] = sum(y_vals)
        dict_arr.append(tmp)
        i += 1
        print(str(i/len(DiseaseID))+"\n")        
    try:
        with open(fileName, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=labels)
            writer.writeheader()
            for elem in dict_arr:
                writer.writerow(elem)
    except IOError:
        print("I/O error")
