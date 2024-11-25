import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn import metrics

target_cols = []

def evaluate_model_performance(y_test, pred, target_cols=None , mode = 'regression'):
    engagement_scores = []
    try:
        n_outputs = y_test.shape[1]
    except IndexError:
        n_outputs = 1
        
    for i in range(n_outputs):
        mse = metrics.mean_squared_error(
            y_true=y_test.values[:,i], 
            y_pred=pred[:,i]
            )
        rmse = np.sqrt(mse)

        if mode == 'regression':
            score_cols = ['RMSE', 'R2', 'MSLE', 'EV Score']
            r2 = metrics.r2_score(
                y_pred=pred[:,i],
                y_true=y_test.values[:,i]
                )
                
            msle = metrics.mean_squared_log_error(
                y_pred=pred[:,i],
                y_true=y_test.values[:,i]
                )

            ev_score = metrics.explained_variance_score(
                y_pred=pred[:,i],
                y_true=y_test.values[:,i]
            )

            engagement_scores.append([rmse, r2, msle, ev_score])

        elif mode == 'classification':
            score_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            accuracy = metrics.accuracy_score(
                y_pred=pred[:,i],
                y_true=y_test.values[:,i]
                )
            
            precision = metrics.precision_score(
                y_pred=pred[:,i],
                y_true=y_test.values[:,i]
                )
            
            recall = metrics.recall_score(
                y_pred=pred[:,i],
                y_true=y_test.values[:,i]
                )
            
            f1_score = metrics.f1_score(
                y_pred=pred[:,i],
                y_true=y_test.values[:,i]
                )
            
            engagement_scores.append([accuracy, precision, recall, f1_score])


    engagement_scores = pd.DataFrame(engagement_scores, 
                                     columns=score_cols, 
                                     index=target_cols)
    
    return engagement_scores

def save_model(model, name):
    save_path = '/workspaces/Crowdfunding-Social-Media-Drivers/Modelling/02_success_engagement/'
    joblib.dump(model, save_path+f'/{name}.pkl')

    return None

def load_model(name):
    save_path = '/workspaces/Crowdfunding-Social-Media-Drivers/Modelling/02_success_engagement/'
    return joblib.load(save_path+f'/{name}.pkl')
    

def encode_targets(y):
    y = y.copy()
    y = np.log(y+1)
    return y

def decode_targets(y):
    y = y.copy()
    y = np.exp(y)-1
    return y

def plot_permutation_importance(model, test_inputs, test_outputs, imp_columns):

    importance = permutation_importance(
        model,
        test_inputs,
        test_outputs.values,
        n_repeats=25,
        random_state=42,
        n_jobs=-1
    )

    importance_df= pd.DataFrame(
        data=importance.importances_mean,
        index=imp_columns,
        columns=['Importance']
    ).sort_values(by='Importance', ascending=False)

    importance_df.plot(kind='bar', figsize=(20,10))
    plt.title('Permutation Importance')
    plt.ylabel('Importance')
    
    plt.savefig('/workspaces/Crowdfunding-Social-Media-Drivers/Results/02_success_engagement_results/feature_importance.png')

    return importance_df



def get_partial_dependence_plot(model, features, x, y, feature_name, categirical_features=None):
    n_features = len(features)
    n_cols = n_features//3
    n_rows = np.ceil(n_features / n_cols)

    fig, axs = plt.subplots(int(n_rows), n_cols, figsize=(15, 20))
    
    axs = axs.ravel()  # Flatten the array of axes

    PartialDependenceDisplay.from_estimator(
    model,
    x,
    features=features,
    categorical_features=categirical_features,
    target=y,
    n_jobs=-1,
    ax=axs[:n_features]  # Use only the needed number of axes
    )

    fig.suptitle(f'Partial dependence plots for {target_cols[y]}')
    save_dir = f'/workspaces/Crowdfunding-Social-Media-Drivers/Results/Results/02_domain-post_results/{feature_name}/'
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    

    plt.savefig(save_dir+ f'{feature_name}.png')
    return None


def save_partial_dependence_plots(model, features, x, feature_name, categirical_features=None):
    for i in range(len(target_cols)):
        get_partial_dependence_plot(model, features, x, i, feature_name, categirical_features=categirical_features)
    
    return None


def save_all_plots(model, x, y):
    imp_df = plot_permutation_importance(model, x, y)

    print('Importance Calculation Done.')

    top_15_imp = imp_df.head(15).index.to_list()
    topic_cols = ['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5',
                  'topic_6', 'topic_7', 'topic_8', 'topic_9', 'topic_10']

    categorical_features = ['post_sponsored', 'type_photo', 'type_video',
        'page_name_GoFundMe', 'page_name_Indiegogo', 'page_name_Kickstarter',
        'entity_ORG', 'entity_PERSON', 'entity_DATE', 'entity_CARDINAL',
        'entity_GPE', 'entity_WORK_OF_ART', 'entity_ORDINAL',
        'entity_MONEY', 'entity_TIME']

    categorical_top_important = [x for x in top_15_imp if x in categorical_features]

    print('Starting top 15 features partial dependence plots')
    save_partial_dependence_plots(model, 
                                top_15_imp,
                                x, 
                                feature_name = 'top_15_features', 
                                categirical_features=categorical_top_important)
    
    print('Starting topic partial dependence plots')
    save_partial_dependence_plots(model, 
                                topic_cols,
                                x, 
                                feature_name = 'topics', 
                                categirical_features=None)
    
    return None


def decode_amount(amount):
    '''
    Since the projects have atleast collected atleast 1 USD, there is no need to add 1 to the amount for log transformation.
    '''
    amount = np.exp(amount)
    return amount