import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import csv
import random
import time


def single_col_timeseries(scol, split, timelag, maxlag):
    slen = int( len(scol) / split )
    res_col = pd.Series()
    for index in range(0, split):
        tmp = scol[int(index*slen+timelag):(slen+index*slen+timelag-maxlag)]
        res_col = res_col.append(tmp, ignore_index=True)
    # print(res_col)
    return res_col


def invert_expression_timeseries(exp_mat, split, maxlag, pan=0):
    df = pd.DataFrame()
    all_mean = np.mean(exp_mat.values)
    all_std = np.std(exp_mat.values)
    for index in range(0, len(exp_mat.columns)):
        sname=exp_mat.columns[index];
        #df[sname]=exp_mat[sname]
        for jindex in range(0+pan,maxlag+pan):
            #use random to change the sequence
            df[sname] = single_col_timeseries(exp_mat[sname],split,jindex,maxlag)
    # print(df)
    return df


def getlinks(target_name, name, importance_, inverse=False):
    feature_imp=pd.DataFrame(importance_, index=name, columns=['imp'])
    feature_large_set = {}
    for i in range(0, len(feature_imp.index)):
        tmp_name=feature_imp.index[i].split('_')
        if tmp_name[0] != target_name:
            if not inverse:
                if (tmp_name[0]+"\t"+target_name) not in feature_large_set:
                    tf_score = feature_imp.loc[feature_imp.index[i], 'imp']
                    feature_large_set[tmp_name[0] + "\t" + target_name] = tf_score
                else:
                    tf_score = feature_imp.loc[feature_imp.index[i], 'imp']
                    feature_large_set[tmp_name[0] + "\t" + target_name] = max(feature_large_set[tmp_name[0] + "\t" + target_name],tf_score)
            else:
                if (target_name + "\t" + tmp_name[0]) not in feature_large_set:
                    tf_score = feature_imp.loc[feature_imp.index[i], 'imp']
                    feature_large_set[target_name + "\t" + tmp_name[0]] = tf_score
                else:
                    tf_score = feature_imp.loc[feature_imp.index[i], 'imp']

                    feature_large_set[target_name + "\t" + tmp_name[0]] = max(
                        feature_large_set[target_name + "\t" + tmp_name[0]], tf_score)
    return feature_large_set


def compute_feature_importances(score_1, score_2, dicts_1, dicts_2):

    dict_all_1 = {}
    dict_all_2 = {}
    score_1 = 1-score_1 / sum(score_1)
    score_2 = 1-score_2 / sum(score_2)
    for i in range(len(score_1)):
        tmp_dict = dicts_1[i]
        for key in tmp_dict:
            tmp_dict[key] = tmp_dict[key]*score_1[i]
        dict_all_1.update(tmp_dict)

    for i in range(len(score_2)):
        tmp_dict = dicts_2[i]
        for key in tmp_dict:
            tmp_dict[key] = tmp_dict[key]*score_2[i]
        dict_all_2.update(tmp_dict)

    d1 = pd.DataFrame.from_dict(dict_all_1, orient='index')
    d1.columns = ["score_1"]
    d2 = pd.DataFrame.from_dict(dict_all_2, orient='index')
    d2.columns = ["score_2"]

    all_df = d1.join(d2)
    all_df['total'] = np.sqrt(all_df["score_1"] * all_df["score_2"])

    return all_df



if __name__ == '__main__':
    start_time = time.time()
    KN_TS_mainRun("insilico_size10_1_knockdowns.tsv", "insilico_size10_1_timeseries.tsv", 5, outputfile="KD_10_1.xls")
    end_time = time.time()
    print(end_time - start_time)
