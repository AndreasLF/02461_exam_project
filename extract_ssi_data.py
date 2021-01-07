import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters


def extract_ssi_data():
    """ Extract new daily positive cases
    
    Return:
        panda dataframe 
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
    params: 
        df (panda dataframe)
    """
    df.plot()
    plt.title("Daily covid cases");
    plt.show()


data_exploration(extract_ssi_data())
    