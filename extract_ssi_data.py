import pandas as pd
import matplotlib.pyplot as plt

weekly_test_regions_file = "data/Data-Epidemiologiske-Rapport-04012021-ja21/Test_regioner.csv"
test_pos_over_time_file = "data/Data-Epidemiologiske-Rapport-04012021-ja21/Test_pos_over_time.csv"

weekly_test_regions = pd.read_csv(weekly_test_regions_file, sep=";")
test_pos_over_time = pd.read_csv(test_pos_over_time_file, sep=";")

df = test_pos_over_time[["Date", "PosPct"]].iloc[:-2]
print(df)

pd.set_option("display.max_columns", 400)
# print("weekly test regions")
# print(weekly_test_regions)

df.plot(x ="Date", y = "PosPct", kind = "bar")
plt.show()



