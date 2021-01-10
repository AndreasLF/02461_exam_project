import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import numpy as np


def extract_ssi_data():
    """ Extract new daily positive cases
    
    Returns:
        (pandas.DataFrame) daily positive covid cases from SSI
     """
    test_pos_over_time_file = "data/Data-Epidemiologiske-Rapport-04012021-ja21/Test_pos_over_time.csv"

    # Read data
    test_pos_over_time = pd.read_csv(test_pos_over_time_file, sep=";", decimal=",")

    # Remove last two rows
    df = test_pos_over_time["NewPositive"].iloc[:-2]

    # Set index to datetime
    df.index = pd.to_datetime(test_pos_over_time["Date"].iloc[:-2])

    # Remove . from data
    df = df.str.replace(".","").astype(int)

    return df

def data_exploration(df):
    """ Plot data 

    Args: 
        df (pandas.DataFrame): Daily positive covid cases 
    """
    df.plot()
    plt.title("Daily covid cases");
    plt.show()

def extract_dmi_temp_data():
    """ Extract data from dmi temperature csv-files

    Returns:
        df (pd.DataFrame) containing lowest, highest and middle temperature each day in 2020
    """
    months = ["januar", "februar", "marts", "april", "maj", "juni", "juli", "august", "september", "oktober", "november", "december"]

    first_month_file = f"data/DMI-TEMP/hele-landet-{months[0]}-2020.csv"
    df = pd.read_csv(first_month_file, sep=";")
    for month in months[1:]:
        dmi_file = f"data/DMI-TEMP/hele-landet-{month}-2020.csv"
        dmi_data = pd.read_csv(dmi_file, sep=";")
        df = df.append(dmi_data)

    df.index = pd.to_datetime(df["DateTime"])
    df = df.drop(["DateTime"],axis=1)
    return df
# data_exploration(extract_ssi_data())


# df = extract_dmi_temp_data()
# df.insert(0, "NewPositive",extract_ssi_data())
# print(df)
# print(extract_ssi_data())