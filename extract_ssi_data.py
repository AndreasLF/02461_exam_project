import pandas as pd

weekly_test_regions_file = "data/Data-Epidemiologiske-Rapport-04012021-ja21/Test_regioner.csv"
test_pos_over_time_file = "data/Data-Epidemiologiske-Rapport-04012021-ja21/Test_pos_over_time.csv"

weekly_test_regions = pd.read_csv(weekly_test_regions_file, sep=";")
test_pos_over_time = pd.read_csv(test_pos_over_time_file, sep=";")

print(test_pos_over_time[["Date","PosPct"]])
