import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import xgboost as xgb
import optuna as op
import matplotlib.pyplot as plt


def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }
    target_df = pd.read_csv('data/Dataset_dHm.csv')
    desc_df = pd.read_csv('data/filt_desc.csv')
    x_train, x_test, y_train, y_test = train_test_split(
        desc_df, target_df["dHm"], test_size=0.3, train_size=0.7)

    model = xgb.XGBRegressor(**param)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return root_mean_squared_error(y_test, pred), mean_absolute_error(y_test, pred)


if __name__ == '__main__':
    study = op.create_study(directions=["minimize", "minimize"])
    op.logging.set_verbosity(op.logging.WARNING)
    study.optimize(objective, n_trials=500,
                   timeout=600, show_progress_bar=True)
    trials = study.best_trials
    best_trial_index = 0
    min_rmse = 10000
    for index in range(0, len(trials)):
        rmse = trials[index].values[0]
        if min_rmse > rmse:
            best_trial_index = index

    best_trial = trials[best_trial_index]
    print("Best trial:")
    print(f"  Value: {best_trial.values}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    fig = op.visualization.matplotlib.plot_optimization_history(
        study, target=lambda t: t.values[0], target_name="SMRE")
    fig = op.visualization.matplotlib.plot_optimization_history(
        study, target=lambda t: t.values[1], target_name="MAE")
    plt.show()
