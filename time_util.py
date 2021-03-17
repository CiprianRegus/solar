# date_time format must be "date HH:..."
def date_hour_equals(date_time1, date_time2):
    date1, time1 = date_time1.split(" ")
    date2, time2 = date_time2.split(" ")
    if date1 != date2:
        return False
    hours1 = time1.split(":")[0]
    hours2 = time2.split(":")[0]
    if hours1 != hours2:
        return False
    return True


def get_hour(date_time):
    split_time = date_time.split(" ")
    if len(split_time) != 2:
        raise Exception("Invalid format for DATE_TIME")

    time = split_time[1]
    return time.split(":")[0]

def get_hour(date_time):
    split_time = date_time.split(" ")
    if len(split_time) != 2:
        raise Exception("Invalid format for DATE_TIME")

    time = split_time[1]
    return time.split(":")[0]

