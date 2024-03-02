import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv(
    'saved_models/QQQ/w_1_8_custom_bin_price_QQQ_1hour_2022_1/intraday_OOS_summary.csv')
print(df)
# df["Train_Loss"] = df["Train_Loss"] / 10000

# plt.plot(df.index, df["Train_Loss"], label='Train_Loss')
plt.plot(df.index, df["Total_Accuracy"], label='OOS_TotAcc')
plt.plot(df.index, df["Long_Accuracy"], label='OOS_LongAcc')
plt.plot(df.index, df["Threshold_Accuracy"], label='OOS_ThreshAcc')
plt.plot(df.index, df["Threshold"], label='Thresh')

plt.title('Time (OHE)')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("In Sample Loss and OOS Accuracy")
plt.show()
