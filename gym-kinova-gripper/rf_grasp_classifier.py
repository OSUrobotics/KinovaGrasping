"""
Trains a random forest grasp classifier via sklearn.
Exports the trained model as .pkl file.

Grasp classifier .pkl can be loaded with:
gc = pickle.load(open(filename, "rb"))
gc.predict(some_data)


Option to run exhaustive feature selection below.
"""
import pickle
from os import listdir
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve

def binary_class_distribution(y_arr):
    """
    Finds distribution of 0 and 1s
    
    Input:
        y_arr: list
    
    Returns: void
    """
    print("1: ", np.sum(y_arr), "\t", "{:.2%}".format(np.sum(y_arr)/len(y_arr)))
    print("0: ",len(y_arr)-np.sum(y_arr), "\t", "{:.2%}".format((len(y_arr)-np.sum(y_arr))/len(y_arr)))

def split_data(feat_in, target_in):
    """
    Splits features and targets into training and test set 80/20
    Input:
        feat_in: list 
        target_in: list
    Returns:
        X_train: list 
        y_train: list
        X_test: list
        y_test: list
    """
    X_train= []
    X_test= []
    y_train= []
    y_test= []
    for i in range(len(feat_in)):
        shuffle_split = StratifiedShuffleSplit(n_splits = 1, test_size=0.20, random_state=None)

        for train_idx, test_idx in shuffle_split.split(feat_in[i], target_in[i]):

            xt = np.array(feat_in[i])[train_idx]
            yt = np.array(target_in[i])[train_idx]
            xte = np.array(feat_in[i])[test_idx]
            yte = np.array(target_in[i])[test_idx]

            X_train.extend(xt)
            y_train.extend(yt)
        
            X_test.extend(xte)
            y_test.extend(yte)

    print("Training Distribution")
    binary_class_distribution(y_train)
    print("Testing Distribution")
    binary_class_distribution(y_test)
    return X_train, y_train, X_test, y_test

def train(X_train, y_train, export=False, export_filename=''):
    """
    Trains randomforest and optionally exports model

    Input:
        X_train: list
        y_train: list
    Returns:
        mod: sklearn object
    """
    #Shuffle training data
    X_train, y_train = shuffle(X_train, y_train)

    #Different of models to test
    mod = RandomForestClassifier(n_estimators=20)
    mod.fit(X_train, y_train)
    if export:
      if export_filename == '':
          pickle.dump(mod, open("gc_model.pkl", "wb"))
      else:
        pickle.dump(mod, open(export_filename+".pkl", "wb"))
        
    return mod

def run_model(X_train_in, y_train_in, X_test_in, y_test_in, feat_dict, export=False, export_filename=''):
    """
    Trains and tests model and optionally exports model
    
    Input:
        X_train_in: list
        y_train_in: list
        y_test_in:
        feat_dict: dict
        export: bool
        export_filename: string
    Returns:
        fpr: float
        tpr: float
        accuracy_score: float
    """
    X_train = []
    X_test = []

    #Get groups remaining
    #Testing
    for i in range(len(X_test_in)):
        tmp = []
        for key, val in feat_dict.items():
            if 'Rangefinder Prox' in key:
                for k in range(3):
                    tmp.append(X_test_in[i][val[k]])
            elif 'Rangefinder Dist' in key:
                for k in range(3):
                    tmp.append(X_test_in[i][val[k]])
            else:
                tmp.append(X_test_in[i][val])
        tmp = np.hstack(tmp)
        X_test.append(tmp)
    #Training
    for i in range(len(X_train_in)):
        tmp = []
        for key, val in feat_dict.items():
            if 'Rangefinder Prox' in key:
                for k in range(3):
                    tmp.append(X_train_in[i][val[k]])
            elif 'Rangefinder Dist' in key:
                for k in range(3):
                    tmp.append(X_train_in[i][val[k]])
            else:
                tmp.append(X_train_in[i][val])
        tmp = np.hstack(tmp)
        X_train.append(tmp)

    print(len(X_train_in[0]))
    print(len(X_train[0]))
    
    ### TRAIN MODEL ###
    model = train(X_train, y_train_in, export, export_filename)
    pred = model.predict(X_test)

    print('Features: ', *feat_dict.keys())
    print(classification_report(y_test_in, pred))

    ### MODEL ROC METRICS ### 
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test_in, y_prob)
    auc = roc_auc_score(y_test_in, y_prob)

    return fpr, tpr, auc, accuracy_score(y_test_in, pred)

def model_std(features, targets, params):
    """
    Trains and tests models 14 times and resplits the data each time
    to calculate standard deviation and mean

    Input:
        features: list
        targets: list
        params: dict
    Returns:
        auc_std: float
        acc_std: float
        auc_mean: float
        acc_mean: float
    """
    loop_fprs = []
    loop_tprs = []
    loop_aucs = []
    loop_accs = []
    for n in range(14): #Arbitrary number of re-splits
        #Approx 80 20 Split -> Run
        X_t_i, y_t_i, X_te_i, y_te_i = split_data(features, targets)
        loop_fpr, loop_tpr, loop_auc, loop_acc = run_model(X_t_i, y_t_i, X_te_i, y_te_i, params)

        #STORE FOR MEAN 
        loop_aucs.append(float(loop_auc))
        loop_accs.append(float(loop_acc))
    
    return np.std(loop_aucs, keepdims=False), np.std(loop_accs, keepdims=False), np.mean(loop_aucs, keepdims=False), np.mean(loop_accs, keepdims=False)

if __name__ == "__main__":
    '''
    (18,) Finger Pos
    (3,) Wrist Pos
    (3,) Obj Pos
    (9,) Joint States
    (3,) Obj Size
    (12,) Finger Object Distance
    (2,) X and Z angle
    (17,) Rangefinder data
    (3,) Gravity vector in local coordinates
    (3,) Object location based on rangefinder data
    (1,) Ratio of the area of the side of the shape to the open portion of the side of the hand
    (1,) Ratio of the area of the top of the shape to the open portion of the top of the hand
    '''

    metric_groups = {'Finger Pos Prox' : np.s_[0:9], 
                    'Finger Pos Dist' : np.s_[9:18],
                    'Wrist Pos': np.s_[18:21],
                    'Obj Pos': np.s_[21:24],
                    'Joint States XYZ': np.s_[24:27],
                    'Joint States Prox': np.s_[27:30],
                    'Joint States Dist': np.s_[30:33],
                    'Obj Size': np.s_[33:36],
                    'Finger Obj Dist Prox': np.s_[36:42],
                    'Finger Obj Dist Dist': np.s_[42:48],
                    'X Angle': np.s_[48], #Rm
                    'Z Angle': np.s_[49], #Rm 
                    'Rangefinder Palm': np.s_[50:55],  
                    'Rangefinder Prox': [np.s_[55:57], np.s_[59:61], np.s_[63:65]],
                    'Rangefinder Dist': [np.s_[57:59], np.s_[61:63], np.s_[65:67]], 
                    'Gravity Vector': np.s_[67:70],
                    'Obj Location': np.s_[70:73],
                    'Ratio Side': np.s_[73], #Rm
                    'Ratio Top': np.s_[74]} #Rm

    #################
    ### LOAD DATA ###
    PATH = 'YOUR DATA PATH HERE'
    all_files = listdir(PATH)
    files = []
    for f in all_files:
        if "pkl" in str(f):
            if "(1)" in str(f) or "Lemon" in str(f):
                print("NOT INCLUDED:", f)
            else:
                print(f)
                files.append(f)

    file_features = []
    file_targets = []
    for filename in files:
        myfile = open(PATH + filename, "rb")
        data = pickle.load(myfile)
        myfile.close()
        file_features.append(data["states"])
        file_targets.append(data["grasp_success"])
    print("Input Feature Files: {}, Outputs, {}".format(len(file_features), len(file_targets)))
    #################

    #########################################################
    ### EXAMPLE TRAINING & SAVING MODEL WITH ALL FEATURES ###
    X_train_in, y_train_in, X_test_in, y_test_in = split_data(file_features, file_targets)
    c_fpr, c_tpr, c_auc, c_acc = run_model(X_train_in, y_train_in, X_test_in, y_test_in, metric_groups, export=True, export_filename="all_feat_model")
    #Can also just use train() defined above
    #Open w/ gc = pickle.load(open(filename, "rb"))
    #gc.predict(data)
    print("Control AUC: {}".format(c_auc))
    print("Control Acc: {}".format(c_acc))
    ##########################################################



    ####################################
    ### EXHAUSTIVE FEATURE SELECTION ###
    run_feat_selection = False
    if run_feat_selection:
        remaining_groups = metric_groups.copy()

        result_table = pd.DataFrame(columns=['removed', 'fpr','tpr','auc', 'mean_auc', 'std_auc','acc', 'mean_acc', 'std_acc'])

        ### ALL PAMERTERS CONTROL GROUP ###
        c_fpr, c_tpr, c_auc, c_acc = run_model(X_train_in, y_train_in, X_test_in, y_test_in, remaining_groups)

        c_std_auc, c_std_acc, c_mean_auc, c_mean_acc= model_std(file_features, file_targets, remaining_groups)


        print("Control AUC: {} mean: {} std: {}".format(c_auc, c_mean_auc, c_std_auc))
        print("Control Acc: {} mean: {} std: {}".format(c_acc, c_mean_acc, c_std_acc))

        result_table = result_table.append({'removed': 0,
                                            'fpr':c_fpr,
                                            'tpr':c_tpr,
                                            'auc':c_auc,
                                            'mean_auc': c_mean_auc,
                                            'std_auc': c_std_auc,
                                            'acc':c_acc,
                                            'mean_acc': c_mean_acc,
                                            'std_acc': c_std_acc}, ignore_index=True)

        params_per_iter = []
        rm_per_iter = []

        best_auc = 0
        best_feats = []


        for i in range(len(remaining_groups)-1):
            removed_keys = []
                
            ### CREATE DICTIONARIES W/OUT 1 PARAM GROUP ###
            selected_groups = [remaining_groups.copy() for x in range(len(remaining_groups))]
            k = 0
            for key, val in remaining_groups.items():
                removed_keys.append(key)
                del selected_groups[k][key]
                k+=1

            ### GET ACCURACY OF EACH MODEL ###
            fprs = []
            tprs = []
            aucs = []
            accs = []
            for param_dict in selected_groups:
                fpr, tpr, auc, acc = run_model(X_train_in, y_train_in, X_test_in, y_test_in, param_dict)
                fprs.append(fpr)
                tprs.append(tpr)
                aucs.append(auc)
                accs.append(acc)

            ### FIND BEST PERFORMER ###
            max_idx = 0
            max = 0
            k = 0
            for score in aucs: 
                if max < score:
                    max = score
                    max_idx = k
                k+=1

            ### UPDATE REMAINING GROUPS TO BEST MODEL ###
            remaining_groups = selected_groups[max_idx]

            ### UPDATE MAX AUC ### 
            if aucs[max_idx] > best_auc:
                best_auc = aucs[max_idx]
                best_feats = [*remaining_groups.keys()]

            ### GET STD DEVIATION OF BEST MODEL ### 
            std_auc, std_acc, mean_auc, mean_acc = model_std(file_features, file_targets, selected_groups[max_idx])
            
            result_table = result_table.append({'removed': i+1,
                                                'tpr':tprs[max_idx],
                                                'fpr':fprs[max_idx],
                                                'auc':aucs[max_idx],
                                                'mean_auc':mean_auc,
                                                'std_auc':std_auc,
                                                'acc':accs[max_idx],
                                                'mean_acc':mean_acc,
                                                'std_acc':std_acc}, ignore_index=True)


            print("Removed: ", removed_keys[max_idx])
            rm_per_iter.append(removed_keys[max_idx])

            print("AUC: {} mean: {} std: {}".format(aucs[max_idx], mean_auc, std_auc))
            print("Acc: {} mean: {} std: {}".format(accs[max_idx], mean_acc, std_acc))

            print('Remaining: ', *remaining_groups.keys())
            params_per_iter.append([*remaining_groups.keys()])

            print("_" * 3)

        #SAVE TO CSV
        param_dt = pd.DataFrame(params_per_iter)
        rm_dt = pd.DataFrame(rm_per_iter)
        param_dt.to_csv("params_per_iter.csv")
        rm_dt.to_csv("rm_per_iter.csv")
        result_table.to_csv('backwards_results.csv')

