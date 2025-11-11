import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import xgboost as xgb
from pickle import dump


def main():
    target_df = pd.read_csv('data/Dataset_dHm.csv')
    desc_df = pd.read_csv('data/filt_desc.csv')
    x_train, x_test, y_train, y_test = train_test_split(
        desc_df, target_df["dHm"], test_size=0.3, train_size=0.7)

    model = xgb.XGBRegressor(max_depth=8,
                             learning_rate=0.02,
                             n_estimators=136,
                             subsample=0.77,
                             colsample_bytree=0.84,
                             min_child_weight=4,
                             gamma=4.409,
                             alpha=0.54
                             )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print(f"SMRE: {root_mean_squared_error(y_test, pred)}, MAE {
          mean_absolute_error(y_test, pred)}")
    download = input("Download Model? [yes/no/exit]")
    if download == "yes":
        model.save_model('models/dHm_model.json')
        return False
    elif download == 'exit':
        return False
    else:
        return True


if __name__ == '__main__':
    repeat = True
    while (repeat == True):
        repeat = main()
