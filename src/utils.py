import os
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from tabulate import tabulate  # we'll use this to print tables nicely; install via pip if needed

from sklearn.model_selection import GridSearchCV
from tabulate import tabulate

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    report = []  # list of tuples for tabulate
    best_score = float('-inf')
    best_model_name = None
    best_model = None

    for model_name, model in models.items():
        param = params.get(model_name, {})
        gs = GridSearchCV(model, param, cv=3, n_jobs=-1, scoring='r2')
        gs.fit(X_train, y_train)

        r2_train = gs.score(X_train, y_train)
        r2_test = gs.score(X_test, y_test)

        report.append((model_name, round(r2_train, 4), round(r2_test, 4)))

        if r2_test > best_score:
            best_score = r2_test
            best_model_name = model_name
            best_model = gs.best_estimator_

    # Print model performance table
    print("\nModel Performance:\n")
    print(tabulate(report, headers=["Model", "R2 Train", "R2 Test"], tablefmt="fancy_grid"))

    # Print best model highlighted in green (if your terminal supports ANSI colors)
    print(f"\n\033[92m✅ Best Model: {best_model_name} with R2 Test Score: {best_score:.4f}\033[0m\n")

    return best_model_name, best_score, best_model



def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
        
def load_object(file_path):
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
    return obj
