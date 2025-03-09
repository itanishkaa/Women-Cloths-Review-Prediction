# Women Clothing Reviews Prediction

This repository contains a machine learning model for predicting customer reviews of women's clothing using Multi Nominal Naive Bayes. The dataset is sourced from an e-commerce platform and analyzed using Python in a Jupyter Notebook.

## Features

- **Data Preprocessing**: Handling missing values, tokenization, and text vectorization.
- **Model Training**: Using Multinomial Naive Bayes for sentiment classification.
- **Evaluation Metrics**: Confusion matrix and classification report.
- **Re-categorization**: Binary classification of ratings into "Good" and "Poor".
  
## Dataset
The dataset used is Women Clothing E-Commerce Review.csv and can be accessed here.

### Dataset Columns
1. Clothing ID
2. Age
3. Title
4. Review
5. Rating
6. Recommended
7. Positive Feedback
8. Division
9. Department
10. Category

## Installation and Usage
### Dependencies
Ensure you have the following Python libraries installed

```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
```
### Running the Notebook
- Clone the repository
```bash
https://github.com/itanishkaa/Women-Cloths-Review-Prediction.git

cd womens-cloths-review-predictions
```
- Open the Jupyter Notebook
```bash
jupyter notebook Women_Cloth_Reviews_Prediction.ipynb
```
- Run the cells sequentially.

## Model Workflow
1. **Data Import & Preprocessing**
- Load dataset using pandas
- Handle missing values by replacing empty reviews with "No Review"
- Convert text reviews into tokens using CountVectorizer

2. **Train-Test Split**
- Split data into 70% training and 30% testing using train_test_split

3. **Feature Extraction**
- Convert text data into numerical vectors using n-grams (bigrams and trigrams)

4. **Model Training**
- Train a Multinomial Naive Bayes classifier

5. **Model Evaluation**
- Compute accuracy, precision, recall, and F1-score
- Display confusion matrix

6. **Re-Categorization**
- Convert ratings into binary labels (0 = Poor, 1 = Good)
- Retrain and evaluate the model with new categories

## Conclusion
- The initial multi-class classification resulted in an overall low precision for lower ratings.
- After re-categorizing ratings into a binary classification (Poor vs. Good), the model achieved an accuracy of 70%.
- The Naive Bayes model performs well for predicting positive reviews but struggles with negative ones due to class imbalance.

## Future Enhancements
- Use TF-IDF for better feature extraction.
- Experiment with other classifiers like Logistic Regression, Random Forest, or Deep Learning.
- Address class imbalance using oversampling or weighted loss functions.
