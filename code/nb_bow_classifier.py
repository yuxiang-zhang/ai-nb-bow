import pandas as pd
import numpy as np

def predict(model: pd.DataFrame, class_log_prior: pd.Series, X):
    V = model.index
    
    # for each doc x: sum the log proba of its words
    score_single_doc = lambda word_list: (model.loc[V.intersection(word_list)]).sum()
    
    # broadcast-add prior to each sum of logs
    scores = class_log_prior + X.apply(score_single_doc)
    
    # find max score for each doc with the corresponding class, concat class and score to make a dataframe
    return pd.concat([scores.idxmax(axis=1), scores.max(axis=1)], axis=1).set_axis(['y_pred', 'score'], axis=1)

def trace_predict(model: pd.DataFrame, class_log_prior: pd.Series, X, y, path):
    # mapping boolean to str
    verdict_mapping = {True:'correct',False:'wrong'}
    
    # make dataframe with columns [y_pred, score, y_true, verdict]
    trace_df = predict(model, class_log_prior, X)
    trace_df['y_true'] = y
    trace_df['verdict'] = (trace_df.y_true == trace_df.y_pred).replace(verdict_mapping)
    
    np.savetxt(path, trace_df.reset_index(), delimiter='  ', fmt='%s')
    return trace_df