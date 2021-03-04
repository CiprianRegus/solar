import pandas as pd
import date_utils
from importlib import reload

reload(date_utils)
"""
    This function returns the  
    For extracting input and target tensors (https://www.kaggle.com/anikannal/solar-power-generation-data)
    offset is used for specifying the lookback interval for energy quantity
"""
def columns_of_interrest(original_production_data, original_weather_data):
    j = 0
    weather_date_times = {} 
    ret = {"DATE_TIME":[], "SOURCE_KEY":[], "DC_POWER":[], "AC_POWER":[], "DAILY_YIELD":[]}
    rows_for_deletion = [] # List of rows that will be deleted because the 2 datasets are incomplete 
    for i in range(len(original_production_data)):
        # Production data is added
        for col in ret.keys():
            ret[col].append(original_production_data.loc[i, col])
    ret["AMBIENT_TEMPERATURE"] = []
    ret["IRRADIATION"] = []
    # Hashmap of type (date_time, index)
    for i in range(len(original_weather_data)):
        weather_date_times[original_weather_data["DATE_TIME"][i]] = i
    # Weather data is added
    for i in range(len(original_production_data)):
        current_date_time = original_production_data["DATE_TIME"][i]
        try:
            j = weather_date_times[current_date_time]
        except:
            """
                There is no data recorded at this date_time
                Invalid data is added so the dataframe can be created from the dictionary
                The entire row will be deleted later
            """
            rows_for_deletion.append(current_date_time)
            ret["AMBIENT_TEMPERATURE"].append("NaN")
            ret["IRRADIATION"].append("NaN")
            continue
        ret["AMBIENT_TEMPERATURE"].append(original_weather_data['AMBIENT_TEMPERATURE'][j])
        ret["IRRADIATION"].append(original_weather_data['IRRADIATION'][j])
    df = pd.DataFrame(data=ret)
    # Remove the invalid rows
    for row in rows_for_deletion:
        df[df.DATE_TIME != row]
    # Split DATE_TIME in DATE and TIME columns
    for i in range(len(df['DATE_TIME'])):
        date, time = df['DATE_TIME'][i].split(" ")
        df.loc[i, 'DATE'] = date
        df.loc[i, 'TIME'] = time
    del df['DATE_TIME']
    return df

def add_offset_columns(data, offset):
    for e in range(offset, len(data[offset:-offset])):
        data.loc[e, 'PREVIOUS_INTERVAL_DC'] = data.loc[e - offset, 'DC_POWER']
    return data[offset: -offset]
    
"""
    The 2 datasets (production and weather have different formats for column DATE_TIME)
    This fuction converts the column to "yyyy-mm-dd HH:MM:SS" format
"""
def convert_date_format(date_time):
    for i in range(len(date_time)): 
        e = date_time['DATE_TIME'][i]
        date, time = e.split(" ")
        split_date = date.split("-")
        split_date[0], split_date[2] = split_date[2], split_date[0]
        time = time[0:5] # Eliminam secundele
        date_time.loc[i, "DATE_TIME"] = "-".join(split_date) + " " + time   
