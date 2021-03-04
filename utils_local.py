
def get_hour(date_time):
    split_time = date_time.split(" ")
    if len(split_time) != 2:
        raise Exception("Invalid format for DATE_TIME")

    time = split_time[1]
    return time.split(":")[0]

