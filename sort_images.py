import pandas as pd
from datetime import datetime
import math

def group_images(pics_dict, seconds_between_groups = 10):
    '''
    :param pics_dict: dictionary with full file paths to pictures and matching timestamps
    :param seconds_between_groups: seconds between photos allowed before switching groups
    :return: pd DataFrame with columns file_names, time_datetime and group
    '''
    #Transform dictionary to pd DataFrame
    data_items = pics_dict.items()
    data_list = list(data_items)
    df = pd.DataFrame(data_list)

    # Set names
    df.columns = ["file_names","time_str"]

    #Convert time to datetime object and add column
    time_str = []
    for index, time in enumerate(df["time_str"]):
        tm = datetime.strptime(time,'%Y:%m:%d %H:%M:%S')
        time_str.append(tm)
    df["time_datetime"] = time_str

    #Sort dataframe based on time
    df_sorted = df.sort_values(by = "time_datetime")

    #Calculate timedifference in minutes
    df_sorted["time_datetime_lag"] = df_sorted["time_datetime"].shift(1)
    df_sorted["time_sec"] = (df_sorted["time_datetime"] - df_sorted["time_datetime_lag"]).dt.total_seconds()

    #Assign groups to pictures, increment if there is more than 10 seconds between pictures
    assignment = [0] #First picture is group 0
    number = 0
    for index, value in enumerate(df_sorted["time_sec"]):
        if not math.isnan(value):
            if value > seconds_between_groups: #longer than 10 seconds between pictures?
                number += 1
            assignment.append(number)
    df_sorted["group"] = assignment

    #Select relevant columns
    return df_sorted[df_sorted.columns[[0,2,5]].tolist()]