import pandas as pd
import numpy as np
import warnings
from sklearn import linear_model, svm, tree, ensemble, neighbors, gaussian_process, kernel_ridge, neural_network, dummy, \
    naive_bayes
from xgboost import XGBRegressor, XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier, \
    RUSBoostClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore')


class RegressionModels:
    def __init__(self, x_train, x_test, y_train, y_test, decode=True):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.decode_input = decode

        if self.y_train.ndim > 1:
            self.multioutput = True
        else:
            self.multioutput = False

        self.not_trained_ = []
        self.not_evaluated_ = []
        self.evaluation_errors_ = []

        self.top_10_socres = None
        self.all_scores = None

    def train_models(self):
        models = {
            'LinearRegression': linear_model.LinearRegression(),
            'Ridge': linear_model.Ridge(random_state=42),
            'RidgeCV': linear_model.RidgeCV(),
            'Lasso': linear_model.Lasso(random_state=42),
            'ElasticNet': linear_model.ElasticNet(random_state=42),
            'ElasticNetCV': linear_model.ElasticNetCV(random_state=42),
            'BayesianRidge': linear_model.BayesianRidge(),
            'ARDRegression': linear_model.ARDRegression(),
            'SGDRegressor': linear_model.SGDRegressor(random_state=42),
            'PassiveAggressiveRegressor': linear_model.PassiveAggressiveRegressor(),
            'RANSACRegressor': linear_model.RANSACRegressor(random_state=42),
            'TheilSenRegressor': linear_model.TheilSenRegressor(),
            'HuberRegressor': linear_model.HuberRegressor(),
            'PoissonRegressor': linear_model.PoissonRegressor(),
            'TweedieRegressor': linear_model.TweedieRegressor(),
            'GammaRegressor': linear_model.GammaRegressor(),
            'OrthogonalMatchingPursuit': linear_model.OrthogonalMatchingPursuit(),
            'OrthogonalMatchingPursuitCV': linear_model.OrthogonalMatchingPursuitCV(),
            'LassoLarsIC': linear_model.LassoLarsIC(),
            'LassoCV': linear_model.LassoCV(),
            'Lars': linear_model.Lars(),
            'LarsCV': linear_model.LarsCV(),
            'LassoLars': linear_model.LassoLars(),
            'LassoLarsCV': linear_model.LassoLarsCV(),
            'KNeighborsRegressor': neighbors.KNeighborsRegressor(),
            'GaussianProcessRegressor': gaussian_process.GaussianProcessRegressor(random_state=42),
            'KernelRidge': kernel_ridge.KernelRidge(),
            'LinearSVR': svm.LinearSVR(),
            'NuSVR': svm.NuSVR(),
            'SVR': svm.SVR(),
            'DecisionTreeRegressor': tree.DecisionTreeRegressor(random_state=42),
            'RandomForestRegressor': ensemble.RandomForestRegressor(random_state=42),
            'GradientBoostingRegressor': ensemble.GradientBoostingRegressor(random_state=42),
            'XGBRegressor': XGBRegressor(random_state=42),
            'AdaBoostRegressor': ensemble.AdaBoostRegressor(random_state=42),
            'BaggingRegressor': ensemble.BaggingRegressor(random_state=42),
            'ExtraTreesRegressor': ensemble.ExtraTreesRegressor(random_state=42),
            'HistGradientBoostingRegressor': ensemble.HistGradientBoostingRegressor(random_state=42),
            'MLPRegressor': neural_network.MLPRegressor(),
            'DummyRegressor': dummy.DummyRegressor(),

        }

        trained_models = {}

        # if y_train is not a 1-d, then use MultiOutputRegressor

        for model_name, model in tqdm(models.items(), desc="Training models"):
            try:
                model.fit(self.x_train, self.y_train)
                trained_models[model_name] = model
            except ValueError:
                self.not_trained_.append(model_name)

        return trained_models

    def decode_targets(self, y):
        y = y.copy()
        y = np.exp(y) - 1
        return y

    def evaluate_models(self, trained_models):
        model_scores = {}
        for model_name, model in tqdm(
                trained_models.items(), desc="Evaluating models"):
            try:
                y_pred_insample = model.predict(self.x_train)
                y_pred_outsample = model.predict(self.x_test)
                if self.decode_input:
                    y_pred_outsample = self.decode_targets(y_pred_outsample)

                insample_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_insample))
                insample_mae = mean_absolute_error(self.y_train, y_pred_insample)
                insample_r2 = r2_score(self.y_train, y_pred_insample)
                insample_ev_score = explained_variance_score(
                    self.y_train,y_pred_insample)

                outsample_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_outsample))
                outsample_mae = mean_absolute_error(self.y_test, y_pred_outsample)
                outsample_r2 = r2_score(self.y_test, y_pred_outsample)
                outsample_ev_score = explained_variance_score(
                    self.y_test, y_pred_outsample)

                model_scores[model_name] = [
                    insample_rmse,
                    insample_mae,
                    insample_r2,
                    insample_ev_score,
                    outsample_rmse,
                    outsample_mae,
                    outsample_r2,
                    outsample_ev_score]
            except ValueError as e:
                self.not_evaluated_.append(model_name)
                self.evaluation_errors_.append(str(e))

        return model_scores

    def format_scores(self, model_scores):
        model_scores = pd.DataFrame(
            model_scores,
            index=[
                'Insample RMSE',
                'Insample MAE',
                'Insample R2',
                'Insample EV Score',
                'Outsample RMSE',
                'Outsample MAE',
                'Outsample R2',
                'Outsample EV Score']).T
        model_scores = model_scores.sort_values(by='Outsample RMSE')

        #round everything to 3 decimal points
        model_scores = model_scores.round(3)
        return model_scores

    def run_evaluation(self):
        trained_models = self.train_models()
        model_scores = self.evaluate_models(trained_models)
        model_scores = self.format_scores(model_scores)

        if len(self.not_evaluated_) > 0:
            self.not_evaluated_ = 'The models ' + \
                                  ', '.join(list(set(self.not_evaluated_))) + ' could not be evaluated.'
        else:
            self.not_evaluated_ = 'All models were evaluated successfully.'

        if len(self.not_trained_) > 0:
            self.not_trained_ = 'The models ' + \
                                ', '.join(list(set(self.not_trained_))) + ' could not be trained.'
        else:
            self.not_trained_ = 'All models were trained successfully.'

        self.top_10_socres = model_scores.head(10)
        self.all_scores = model_scores

        return 'Training and Evaluation completed.'


class ClassificationModels:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.not_evaluated_ = []
        self.not_trained_ = []
        self.n_unique_targets = len(np.unique(self.y_test))

    def train_models(self):
        models = {
            'Logistic Regression': linear_model.LogisticRegression(),
            'Ridge Classifier': linear_model.RidgeClassifier(),
            'Ridge Classifier CV': linear_model.RidgeClassifierCV(),
            'SGD Classifier': linear_model.SGDClassifier(),
            'Perceptron': linear_model.Perceptron(),
            'Passive Aggressive Classifier': linear_model.PassiveAggressiveClassifier(),
            'Bernoulli Naive Bayes': naive_bayes.BernoulliNB(),
            'Gaussian Naive Bayes': naive_bayes.GaussianNB(),
            'Multinomial Naive Bayes': naive_bayes.MultinomialNB(),
            'AdaBoost': ensemble.AdaBoostClassifier(),
            'Bagging': ensemble.BaggingClassifier(),
            'Extra Trees': ensemble.ExtraTreesClassifier(),
            'Gradient Boosting': ensemble.GradientBoostingClassifier(),
            'Random Forest': ensemble.RandomForestClassifier(),
            'XGBoost': XGBClassifier(),
            'Support Vector Machine': svm.SVC(),
            'Linear Support Vector Machine': svm.LinearSVC(),
            'Nu-Support Vector Machine': svm.NuSVC(),
            'Decision Tree': tree.DecisionTreeClassifier(),
            'K-Nearest Neighbors': neighbors.KNeighborsClassifier(),
            'Gaussian Process': gaussian_process.GaussianProcessClassifier(),
            'Kernel Ridge': kernel_ridge.KernelRidge(),
            'Neural Network': neural_network.MLPClassifier(),
            'Dummy Classifier': dummy.DummyClassifier(),
            'Balanced Random Forest Classifier': BalancedRandomForestClassifier(random_state=42),
            'Balanced Bagging Classifier': BalancedBaggingClassifier(random_state=42),
            'Easy Ensemble Classifier': EasyEnsembleClassifier(random_state=42),
            'RUSBoost Classifier': RUSBoostClassifier(random_state=42),
        }

        trained_models = {}
        for model_name, model in tqdm(models.items(), desc="Training models"):
            try:
                model.fit(self.x_train, self.y_train)
                trained_models[model_name] = model
            except ValueError:
                self.not_trained_.append(model_name)

        return trained_models

    def evaluate_models(self, trained_models):
        model_scores = {}
        for model_name, model in tqdm(
                trained_models.items(), desc="Evaluating models"):
            try:
                y_pred = model.predict(self.x_test)

                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_pred)

                confusion_matrix = metrics.confusion_matrix(
                    self.y_test, y_pred)

                # class_wise acc
                class_wise_acc = {}
                for i in range(self.n_unique_targets):
                    class_wise_acc[i] = confusion_matrix[i][i] / sum(confusion_matrix[i])

                class_wise_acc_list = [class_wise_acc[i] for i in range(self.n_unique_targets)]

                model_scores[model_name] = [
                                               accuracy, precision, recall, f1, roc_auc,
                                           ] + class_wise_acc_list

            except ValueError:
                self.not_evaluated_.append(model_name)

        return model_scores

    def format_scores(self, model_scores):
        class_wise_acc_labels = ['Class-{} Acc'.format(i) for i in range(self.n_unique_targets)]
        model_scores = pd.DataFrame(
            model_scores,
            index=[
                      'Accuracy',
                      'Precision',
                      'Recall',
                      'F1 Score',
                      'ROC AUC',
                  ] + class_wise_acc_labels).T
        model_scores = model_scores.sort_values(by='Class-0 Acc', ascending=False)

        return model_scores

    def run_evaluation(self):
        trained_models = self.train_models()
        model_scores = self.evaluate_models(trained_models)
        model_scores = self.format_scores(model_scores)

        self.not_evaluated_ = 'The models' + \
                              ', '.join(list(set(self.not_evaluated_))) + ' could not be evaluated.'
        self.not_trained_ = 'The models' + \
                            ', '.join(list(set(self.not_trained_))) + ' could not be trained.'

        self.top_10_socres = model_scores.head(10)
        self.all_scores = model_scores

        return 'Training and Evaluation completed.'
