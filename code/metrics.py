import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def confusion_matrix(y_true: pd.Series, y_pred: pd.Series):
    positive_pred = y_pred == 'yes'
    positive_target = y_true == 'yes'
    
    N = len(y_true)
    np = len(y_true[positive_target])
    nn = N - np
    
    tp, fp = (y_true[positive_pred] == y_pred[positive_pred]).value_counts().reindex([True, False])
    fn = np - tp
    tn = nn - fp
    
    return pd.DataFrame([
        [tn,fp],
        [fn,tp]
    ], index=('no', 'yes'), columns=('no', 'yes'))

def accuracy_scores(tn, fp, fn, tp):
    N = sum((tn, fp, fn, tp))
    return pd.Series([
        (tp+tn)/N, (tp+tn)/N
    ], index=('yes', 'no'), dtype=float, name='accuracy')

def precision_scores(tn, fp, fn, tp):
    return pd.Series([
        tp/(tp+fp), tn/(fn+tn)
    ], index=('yes', 'no'), dtype=float, name='precision')

def recall_scores(tn, fp, fn, tp):
    return pd.Series([
        tp/(tp+fn), tn/(tn+fp)
    ], index=('yes', 'no'), dtype=float, name='recall')

def f1_scores(tn, fp, fn, tp):
    prec = precision_scores(tn, fp, fn, tp)
    reca = recall_scores(tn, fp, fn, tp)
    return (2 * prec * reca / (prec + reca)).rename('f1')

def eval_model(y_true: pd.Series, y_pred: pd.Series, path):
    # outcomes = (tn, fp, fn, tp)
    outcomes = confusion_matrix(y_true, y_pred).to_numpy().ravel()
    
    # combine scores in a dataframe
    df = pd.concat([get_scores(*outcomes) for get_scores in (accuracy_scores, precision_scores, recall_scores, f1_scores)], axis=1)
    
    # format output scores
    lines = ['{:.4f}  {:.4f}\n'.format(*df[col].tolist()) for col in df.columns]
    
    # format accuracy
    lines[0] = lines[0].rsplit(' ', 1)[-1]
    
    # write scores to file
    with open(path, 'w') as f:
        f.writelines(lines)

    return df

def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series):
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')