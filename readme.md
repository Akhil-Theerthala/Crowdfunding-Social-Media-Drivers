# Project Description

This project analyzes the impact of social media features on crowdfunding project success. It evaluates the influence of likes, shares, and other features, as well as the impact of social media post characteristics on project engagement.

## File Structure

The project includes the following files:

* **Data**: Code files for data processing and figures related to exploratory data analysis.
* **Modelling**: Code files for testing and fine-tuning different models.
* **Results**: Results of the final fine-tuned models.

## Modelling

The project uses Latent Dirchlet Allocation (LDA) to extract topics from social media post text, which are then used as features in the model. The number of topics is determined using the,

* **Coherence score**, which is a measure of how well the topics are separated from each other;
* **Topic Diveristy**, which is a measure of how well the topics are distributed across the corpus; and
* **Perplexity score**, which is a measure of how well the model predicts the corpus. (In the project we used a log_transformed perplexity score, which explains the negative values in the table below. The closer the perplexity is to 1, the better the model.)

The scores for different number of topics are presented in the following table:

| Number of Topics | Coherence Score | Topic Diversity | Perplexity Score |
| :---: | :---: | :---: | :---: |
| 13 | 0.419  | 0.959 | -8.907 |
| 12 | 0.405 | 0.9733 | -8.58 |
| 11 | 0.393 | 0.9709 | -8.49 |
| 14 | 0.391| 0.960 |-9.195 |
| 9 | 0.391 | 0.964 |-8.487 |
| 8 | 0.389 | 0.985 |-8.468 |
| 10 | 0.387 | 0.976 |-8.482 |
| 5 | 0.372 | 0.984 |-8.371 |
| 4 | 0.340 | 0.990 |-8.291 |
| 7 | 0.334 | 0.994 |-8.452 |
| 6 | 0.322 | 0.980 |-8.417 |  

The final choice was the model with 12 topics, because of similar coherence, with higher topic diversity. The perplexity score was also comparitively better than the 13-topic model.

The project follows a two-step modelling approach. The first step involves generating baseline metrics using a custom class, where different models are tested on the same dataset.

The final fine-tuned models produced better results, which are presented in the following tables:

* Social Media Engagement Prediction:

| Model | RMSE | R2 | MSLE | EV Score |
| :---: | :---: | :---: | :---: | :---: |
| likes |34.753422 | 0.654390 |0.166498 |0.666743 |
| shares | 16.302093 | 0.456943 | 0.515671 | 0.483227 |
| comments | 24.027714 | 0.590026 | 0.507668 | 0.603402 |
| positive_reactions | 18.340882 | 0.535504 | 0.457282 | 0.554334 |
| negative_reactions | 43.634842 | 0.188065 | 0.894120 | 0.211936 |

* Success Prediction:

| Model | F1 | ROC_AUC | Class_0 Acc | Class_1 Acc | Overall Acc|
| :---: | :---: | :---: | :---: | :---: | :---: |
| Domain and Post Features | 0.79 | 0.74 | 0.82 | 0.67 | 0.68 |
| Domain Features and post engagement | 0.82 | 0.65 | 0.55 | 0.74 | 0.72 |

* Number of Backers Prediction:

| Model | R2 | MSLE | RMSE |
| :---: | :---: | :---: | :---: |
| Domain and Post Features | 0.29 | 1.15 | 3708.6 |
| Domain Features and post engagement | 0.33 | 1.05 | 3587.19 |

* Collection Ratio Prediction:

| Model | R2 | RMSE | MSLE |
| :---: | :---: | :---: | :---: |
| Domain and Post Features | 0.22 | 1.7 | 0.2 |
| Domain Features and post engagement | 0.1 | 1.88 | 0.22 |

The second step involves using the baseline metrics to select the best performing models and further tuning them using GridSearchCV.

---

## Results

The results of the experiments can be divided into two parts:

1. The Permutation Importance Plots
2. The partial dependence plots

### Permutation Importance Plots

The permutation importance plots show the importance of each feature in the model. This is done by shuffling the values of each feature and measuring the impact on the model's performance. The overall permutation importance is then normalized to get an overall measure of importance.

These importance figures are present in the respective folder under the name `feature_importnace.png` in the results folder.

### Partial Dependence Plots

The partial dependence plots show the impact of each feature on the model's prediction. This is done by varying the values of each feature and measuring the impact on the model's prediction. On a high level, this is similar to the permutation importance plots, but instead of measuring the impact on the model's performance, it measures the impact on the model's prediction.

The partial dependence is evaluated on 3 parts:

1. The top-15 important features as identified by the permutation importance plots.
2. The topic scores of the document as estimated by the LDA model.
3. The text-complexity of the social media post, which includes the Lix score, the Automatic Readability score, and the entropy score.

These are saved in the corresponding folders can be accessed in the `Results` Folder.
