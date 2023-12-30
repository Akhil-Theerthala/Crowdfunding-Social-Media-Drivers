import pandas as pd
import numpy as np
import warnings
from sklearn import linear_model, svm, tree, ensemble, neighbors, gaussian_process, kernel_ridge, neural_network, dummy
from xgboost import XGBRegressor, XGBClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error, explained_variance_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm


warnings.filterwarnings('ignore')


class RegressionModels:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        if self.y_train.ndim > 1:
            self.multioutput = True
        else:
            self.multioutput = False
            
        self.not_trained_ = []
        self.not_evaluated_ = []

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
            'RANSACRegressor': linear_model.RANSACRegressor(),
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
            'BayesianRidge': linear_model.BayesianRidge(),
            'ARDRegression': linear_model.ARDRegression(),
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
                y_pred = model.predict(self.x_test)
                y_pred = self.decode_targets(y_pred)

                overall_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                overall_mae = mean_absolute_error(self.y_test, y_pred)
                overall_r2 = r2_score(self.y_test, y_pred)
                overall_msle = mean_squared_log_error(self.y_test, y_pred)
                overall_ev_score = explained_variance_score(
                    self.y_test, y_pred)

                model_scores[model_name] = [
                    overall_rmse,
                    overall_mae,
                    overall_r2,
                    overall_msle,
                    overall_ev_score]
            except ValueError:
                self.not_evaluated_.append(model_name)

        return model_scores

    def format_scores(self, model_scores):
        model_scores = pd.DataFrame(
            model_scores,
            index=[
                'RMSE',
                'MAE',
                'R2',
                'MSLE',
                'EV Score']).T
        model_scores = model_scores.sort_values(by='RMSE')

        # round to 3 decimal places
        model_scores.R2 = model_scores.R2.round(3)
        model_scores['EV Score'] = model_scores['EV Score'].round(3)
        model_scores.MSLE = model_scores.MSLE.round(3)
        model_scores.RMSE = model_scores.RMSE.round(3)
        model_scores.MAE = model_scores.MAE.round(3)

        return model_scores

    def run_evaluation(self):
        trained_models = self.train_models()
        model_scores = self.evaluate_models(trained_models)
        model_scores = self.format_scores(model_scores)

        self.not_evaluated_ = 'The models ' + \
            ', '.join(list(set(self.not_evaluated_))) + ' could not be evaluated.'
        self.not_trained_ = 'The models ' + \
            ', '.join(list(set(self.not_trained_))) + ' could not be trained.'

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

    def train_models(self):
        models = {
            'Logistic Regression': linear_model.LogisticRegression(),
            'Support Vector Machine': svm.SVC(),
            'Decision Tree': tree.DecisionTreeClassifier(),
            'Random Forest': ensemble.RandomForestClassifier(),
            'K-Nearest Neighbors': neighbors.KNeighborsClassifier(),
            'Gaussian Process': gaussian_process.GaussianProcessClassifier(),
            'Kernel Ridge': kernel_ridge.KernelRidge(),
            'Neural Network': neural_network.MLPClassifier(),
            'Dummy Classifier': dummy.DummyClassifier()
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

                model_scores[model_name] = [
                    accuracy, precision, recall, f1, roc_auc]
            except ValueError:
                self.not_evaluated_.append(model_name)

        return model_scores

    def format_scores(self, model_scores):
        model_scores = pd.DataFrame(
            model_scores,
            index=[
                'Accuracy',
                'Precision',
                'Recall',
                'F1 Score',
                'ROC AUC']).T
        model_scores = model_scores.sort_values(by='Accuracy', ascending=False)

        # Round to 3 decimal places
        model_scores.Accuracy = model_scores.Accuracy.round(3)
        model_scores.Precision = model_scores.Precision.round(3)
        model_scores.Recall = model_scores.Recall.round(3)
        model_scores['F1 Score'] = model_scores['F1 Score'].round(3)
        model_scores['ROC AUC'] = model_scores['ROC AUC'].round(3)

        return model_scores

    def run_evaluation(self):
        trained_models = self.train_models()
        model_scores = self.evaluate_models(trained_models)
        model_scores = self.format_scores(model_scores)

        self.not_evaluated_ = 'The models' + \
            ', '.join(list(set(self.not_evaluated_))) + ' could not be evaluated.'
        self.not_trained_ = 'The models' + \
            ', '.join(list(set(self.not_trained_))) + ' could not be trained.'

        return model_scores
