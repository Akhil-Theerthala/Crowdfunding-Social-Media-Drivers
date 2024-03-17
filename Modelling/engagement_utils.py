import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, explained_variance_score


target_cols = ['likes', 'shares', 'comments', 'positive_reactions', 'negative_reactions']
scalable_features = ['likes_at_posting', 'followers_at_posting']
noticable_entities = ['ORG', 'PERSON', 'DATE',
                        'CARDINAL', 'GPE', 'PRODUCT', 
                        'WORK_OF_ART','ORDINAL', 'MONEY',
                        'TIME', 'NORP']

def evaluate_model_performance(y_test, pred):
    target_cols = ['likes', 'shares', 'comments', 'positive_reactions', 'negative_reactions']
    engagement_scores = []
    for i in range(5):
        
        mse = mean_squared_error(
            y_true=y_test.values[:,i], 
            y_pred=pred[:,i]
            )
        rmse = np.sqrt(mse)


        r2 = r2_score(
            y_pred=pred[:,i],
            y_true=y_test.values[:,i]
            )
            
        msle = mean_squared_log_error(
            y_pred=pred[:,i],
            y_true=y_test.values[:,i]
            )

        ev_score = explained_variance_score(
            y_pred=pred[:,i],
            y_true=y_test.values[:,i]
        )

        engagement_scores.append([rmse, r2, msle, ev_score])

    engagement_scores = pd.DataFrame(engagement_scores, 
                                     columns=['RMSE', 'R2', 'MSLE', 'EV Score'], 
                                     index=target_cols)
    
    return engagement_scores

def save_model(model, name):
    save_path = '/home/theerthala/Documents/repos/Crowdfunding-Social-Media-Drivers/Modelling/final_models/01_Enagement_prediction/'
    joblib.dump(model, save_path+f'/{name}.pkl')

    return None

def load_model(name):
    save_path = '/home/theerthala/Documents/repos/Crowdfunding-Social-Media-Drivers/Modelling/final_models/01_Enagement_prediction/'
    return joblib.load(save_path+f'/{name}.pkl')
    


def pre_process(x):
    scaler = RobustScaler()
    x.loc[:,scalable_features] = scaler.fit_transform(x.loc[:,scalable_features])

    x = pd.concat([x, pd.get_dummies(x.type, prefix='type', drop_first=True).astype(int)], axis=1).drop('type', axis=1)
    x = pd.concat([x, pd.get_dummies(x.page_name, prefix='page_name').astype(int)], axis=1).drop('page_name', axis=1)
    for entity in noticable_entities:
        x[f'entity_{entity}'] = x.entities_identified.fillna('None').str.split(',').apply(lambda entity_list: entity in entity_list).astype(int)
    
    x = x.drop('entities_identified', axis=1)
    return x


def process_targets(y):
    y = y.copy()
    y = np.log(y+1)
    return y

def decode_targets(y):
    y = y.copy()
    y = np.exp(y)-1
    return y


def plot_permutation_importance(model, test_inputs, test_outputs):

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
        index=test_inputs.columns,
        columns=['Importance']
    ).sort_values(by='Importance', ascending=False)

    importance_df.plot(kind='bar', figsize=(20,10))
    plt.title('Permutation Importance')
    plt.ylabel('Importance')
    
    plt.savefig('/home/theerthala/Documents/repos/Crowdfunding-Social-Media-Drivers/Results/01 - FB Engagement/feature_importance.png')
    importance_df.to_csv('/home/theerthala/Documents/repos/Crowdfunding-Social-Media-Drivers/Results/01 - FB Engagement/feature_importance.csv')
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

    save_dir = f'/home/theerthala/Documents/repos/Crowdfunding-Social-Media-Drivers/Results/01 - FB Engagement/{feature_name}/'
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    

    plt.savefig(save_dir+ f'partial_dependence_{target_cols[y]}')
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
                  'topic_6', 'topic_7', 'topic_8', 'topic_9', 'topic_10', 'topic_11']

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
