import pickle
import dill

from datetime import datetime

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def main():

    # Загрузим обработанный датафрэйм
    with open('data/union_dataset.pkl', 'rb') as file:
        df = pickle.load(file)

    X = df.drop(['target_action', 'hit_month', 'hit_day', 'hit_time', 'hit_number', 'visit_number', 'visit_hour', 'visit_min',
                 'hit_referer', 'event_category', 'car_brand', 'car_model'], axis=1)
    y = df['target_action']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(max_categories=10, handle_unknown='ignore', sparse_output=False))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int32', 'float32'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include='category'))
    ])

    preprocessor = Pipeline(steps=[
        ('column_transformer', column_transformer)
    ])

    models = (
        LogisticRegression(penalty='l2', solver='sag', max_iter=1000),
        RandomForestClassifier(n_estimators=200, max_depth=15),
        MLPClassifier(random_state=5, max_iter=300),
        AdaBoostClassifier(n_estimators=500, random_state=20)
    )

    best_ROC_AUC = .0
    best_pipe = None

    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Проверяем вероятности только для положительного исхода
        # Рассчитываем ROC_AUC
        ROC_AUC = roc_auc_score(y, pipe.fit(X, y).predict_proba(X)[:, 1])
        print(f'model: {type(model).__name__}, ROC_AUC: {ROC_AUC:.4f}')

        if ROC_AUC > best_ROC_AUC:
            best_ROC_AUC = ROC_AUC
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, ROC_AUC: {best_ROC_AUC:.4f}')

    best_pipe.fit(X, y)
    with open('target_action_prediction_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Target action prediction model',
                'author': 'Viktoriia Ilina',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'ROC_AUC': best_ROC_AUC
            }
        }, file)

if __name__ == '__main__':
    main()
