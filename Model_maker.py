import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import xgboost as xgb

target_df = pd.read_csv('data/Dataset_dHm.csv')
desc_df = pd.read_csv('data/filt_desc.csv')
x_train, x_test, y_train, y_test = train_test_split(
    desc_df, target_df["dHm"], test_size=0.3, train_size=0.7)

model = xgb.XGBRegressor(max_depth=4,
                         learning_rate=0.077938038687677507,
                         n_estimators=185,
                         subsample=0.886066446719408,
                         colsample_bytree=0.07793808687677507,
                         min_child_weight=2,
                         gamma=4.403073292629419)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"SRE: {root_mean_squared_error(y_test, pred)}, MAE {
      mean_absolute_error(y_test, pred)}")
