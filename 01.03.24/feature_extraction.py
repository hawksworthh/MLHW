import numpy as np
import pandas as pd
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_regression, SelectFromModel
from sklearn.feature_selection import f_regression
import shap
from sklearn.inspection import permutation_importance

class feature:

    def __init__(self, data, target, numercial = 0, categorial = 0, 
                 correlation_threshold = 0.5, f_p_threshold = 0.05, shapley_threshold = 0.05, 
                 alpha_error_first_type = False):
        #self.numerical = numercial
        #self.categorial = categorial
        self.target = target
        self.threshold_1 = correlation_threshold
        self.threshold_2 = f_p_threshold
        self.threshold_3 = shapley_threshold
        self.alpha_error_indicator = alpha_error_first_type
        self.__df_cols = data.columns
        self.cols = data.columns.to_numpy()

    def mutual_info_selector(self, x, y, feature_number, _mode = 'k_best'):
        selector = GenericUnivariateSelect(score_func = mutual_info_regression,
                                    mode = _mode)
        selector.fit(x, y)
        return pd.DataFrame(data = {'mutual_score': selector.scores_},
                            index = self.cols[self.cols != self.target])

    def f_test_selector(self, x, y, feature_number, _mode = 'k_best'):
        selector = GenericUnivariateSelect(score_func = f_regression,
                                    mode = _mode,
                                    param = feature_number)
        selector.fit(x, y)
        return pd.DataFrame(data = {'f_p_values': selector.pvalues_},
                            index = self.cols[self.cols != self.target])
    
    def shapley_selection(self, x_train_, y_train_, x_test_, y_test_, model, feature_number):
        model = model.fit(x_train_, y_train_)
        explainer = shap.Explainer(model, x_test_)
        shap_values = explainer(x_test_)
        
        df_loc = pd.DataFrame({'col_name': x_train_.columns,
                               'shapley_mean' : np.mean(abs(shap_values.values), axis = 0)})
        return df_loc.set_index('col_name')

    def permutation_check(self,  x_train_, y_train_, x_test_, y_test_, model, n_rep = 30, rand_st = 0):
        model_loc = model.fit(x_train_, y_train_)

        r = permutation_importance(model, x_test_, y_test_,
                           n_repeats = n_rep,
                           random_state = rand_st)
        df1 = pd.DataFrame({'col_name' : [], 'imp_mean' : [], 'imp_std' : []})

        for i in r.importances_mean.argsort()[::-1]:
            #if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            dict1 = {'col_name' :[self.__df_cols[i]],'imp_mean' : [r.importances_mean[i]],'imp_std' : [r.importances_std[i]]}
            df1 = pd.concat([df1, pd.DataFrame(dict1)], ignore_index = True)
        return df1.set_index(df1.col_name).drop(['col_name'], axis = 1)
    
    @staticmethod
    def indocator_simple(x, y):
        if abs(x) >= y:
            return 1
        else: return 0
    
    @staticmethod
    def overall_selector(data, perm_score_threshold, f_test_threshold, mututal_info_threshold, shapley_value_threshold):
        loc_df = data.copy()
        col_names = ['imp_mean', 'f_p_values', 'mutual_score', 'shapley_mean']
        for col, ind in zip(col_names, [perm_score_threshold, f_test_threshold, mututal_info_threshold, shapley_value_threshold]):
            loc_df[f'{col}_thr'] = loc_df[col].map(lambda x: feature.indocator_simple(x, ind) )
        return loc_df
    