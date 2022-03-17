import os
import csv
import pickle
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def find_delimiter(csv_path: str) -> str:
    with open(csv_path, 'r') as f:
        sniffer = csv.Sniffer()
        return str(sniffer.sniff(f.readline()).delimiter)


def get_facial_emotions_estimator(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path, delimiter=find_delimiter(csv_path))
    X = df.drop('Emotion', axis=1)
    y = df['Emotion']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    pre_processing = [
        StandardScaler(),
        SimpleImputer(strategy='mean')
    ]
    pipelines = {
        'LogisticRegression': make_pipeline(*pre_processing, LogisticRegression(max_iter=1000, random_state=17)),
        'RandomForestClassifier': make_pipeline(*pre_processing, RandomForestClassifier(random_state=17)),
        'KNeighborsClassifier': make_pipeline(*pre_processing, KNeighborsClassifier(n_neighbors=7))
    }

    evaluated = pd.DataFrame(columns=['model', 'acc_score', 'eval_time'], dtype='object')
    for name, pipeline in pipelines.items():
        print(f'\t - Evaluating {name}...', end=' ')
        start = time.time()
        # Train
        model = pipeline.fit(X_train, y_train)
        # Predict
        y_pred = model.predict(X_test)
        end = time.time()
        # Evaluate
        acc_score = accuracy_score(y_test, y_pred)
        eval_time = end - start
        # Update
        evaluated.loc[name] = [model, acc_score, eval_time]
        print('Accuracy: {:.2f}%  Time: {:.2f}s'.format(acc_score * 100, eval_time))

    best_estimator = evaluated.sort_values(['acc_score', 'eval_time'], ascending=[False, True]).index[0]

    print(f'\n\tMost efficient estimator is {best_estimator}')

    with open(output_path, 'wb') as f:
        pickle.dump(evaluated.loc[best_estimator, 'model'], f)
        print(f'\t{output_path} successfully dumped!\n')


if __name__ == '__main__':
    for i, prefix in enumerate(['emotion', 'gesture'], start=1):
        print(f' [{i}] {prefix.title()}')
        path_to_csv, path_to_model = [
            os.path.join(folder, f'{prefix}_{suffix}')
            for folder, suffix in [('data', 'data.csv'), ('models', 'classifier.pkl')]
        ]
        get_facial_emotions_estimator(path_to_csv, path_to_model)
