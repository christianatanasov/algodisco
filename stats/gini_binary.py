import numpy as np
import pandas as pd

def gini(df : pd.DataFrame, x_col, y_col):
    """
    Gini impurity calculator;
    First step is to define all possible values (x) which could be decision making.
    s_l and s_r are the two new states for <x< (called left and right here)
    w_l and w_r are the weights or the sizes of the new states
    t_l and t_r are the number of identifiers of type 1 in each of the states
    f_l and f_r are the number of identifiers of type 0 in each of the states
    Gini impurity is calculated for both states - gi_l and gi_r
    The result for the given value of decision is the weighted average of gi_l,gi_r with w_l,w_r
    The minimum impurity (cost) is chosen minimising disorder, and a decision making value is returned.
    """
    x = sorted(df[x_col].rolling(2).mean().dropna().unique())
    y_len = len(df[y_col])
    impurity_vals = pd.DataFrame(columns=['DECISION_VALUE', 'IMPURITY'])
    for val in x:
        try:
            s_l, s_r = [x for _, x in df.groupby(df[x_col] < val)]
            w_l, w_r = [x.shape[0] for x in [s_l, s_r]]
            t_l, t_r = [x[y_col].sum() for x in [s_l, s_r]]
            f_l, f_r = w_l - t_l, w_r - t_r
            gi_l = 1 - np.power(t_l / w_l, 2) - np.power(f_l / w_l, 2)
            gi_r = 1 - np.power(t_r / w_r, 2) - np.power(f_r / w_r, 2)
            gi = (w_l / y_len) * gi_l + (w_r / y_len) * gi_r
            impurity_vals = impurity_vals.append(pd.DataFrame({'DECISION_VALUE' : [val], 'IMPURITY' : [gi]}))
        except:
            pass #Manages machine precision error when a branch is not splittable
    gini_impurity = impurity_vals.loc[impurity_vals['IMPURITY']==impurity_vals['IMPURITY'].min()].groupby('IMPURITY', as_index=False).mean()
    gini_impurity['FEATURE'] = x_col
    return gini_impurity