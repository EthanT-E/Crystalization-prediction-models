import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
import optuna as op
import matplotlib.pyplot as plt
import sys


def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0),
        'tree_method': trial.suggest_categorical('tree_method', ['exact', 'hist', 'approx'])
    }
    target_df = pd.read_csv('data/Dataset_dHm.csv')
    desc_df = pd.read_csv('data/tsne_dhm_desc.csv')
    pca = PCA(n_components=43)
    pca_desc = pca.fit_transform(desc_df)

    tnse = TSNE(perplexity=9)
    X_tsne = tnse.fit_transform(pca_desc)
    x_train, x_test, y_train, y_test = train_test_split(
        desc_df, target_df["dHm"], test_size=0.3, train_size=0.7)

    model = xgb.XGBRegressor(**param)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
#     return root_mean_squared_error(y_test, pred)
    return root_mean_squared_error(y_test, pred), mean_absolute_error(y_test, pred)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        n_trials = 100
    else:
        n_trials = int(sys.argv[1])
    study = op.create_study(directions=["minimize", "minimize"])
    # study = op.create_study()
    op.logging.set_verbosity(op.logging.WARNING)
    study.optimize(objective, n_trials=n_trials,
                   timeout=6000, show_progress_bar=True)
    trials = study.best_trials
    best_trial_index = 0
    min_rmse = 10000
    print("Best trials:")
    for trial in trials:
        print(f"  Value: {trial.values}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    if len(sys.argv) == 2:
        fig = op.visualization.matplotlib.plot_optimization_history(
            study, target=lambda t: t.values[0], target_name="SMRE")
        fig = op.visualization.matplotlib.plot_optimization_history(
            study, target=lambda t: t.values[1], target_name="MAE")
        fig = op.visualization.matplotlib.plot_param_importances(
            study, target=lambda t: t.values[0], target_name="SMRE")
        fig = op.visualization.matplotlib.plot_param_importances(
            study, target=lambda t: t.values[1], target_name="MAE")
        plt.show()
