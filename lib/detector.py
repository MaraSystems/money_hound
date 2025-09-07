from lib import engineer

def unsual_amount(df):
    # detect unsual amounts
    columns = [
        'large_amount', 'large_amount_drain', 'large_amount_pump', 
        'holder_amount_bound_frequency', 
        'holder_large_amount_drain_True_occurance', 'holder_large_amount_pump_True_occurance', 
        'holder_amount_avg_30D'
    ]
    return engineer.anomalize(df, 'unsual_amount', columns)


def unsual_balance(df):
    # detect unsual balances
    columns = [
        'balance_jump_rate', 'balance_jump_rate_absolute', 'drained_balance', 'pumped_balance', 
        'holder_balance_jump_bound_frequency', 'holder_balance_jump_rate_bound_frequency', 
        'holder_drained_balance_True_occurance', 'holder_pumped_balance_True_occurance', 
        'holder_balance_avg_30D'
    ]
    return engineer.anomalize(df, 'unsual_balance', columns)


def unsual_location(df):
    # detect unsual locations
    columns = [
        'distance_from_home (km)', 
        'far_distance',
        'holder_far_distance_True_occurance',
        'holder_distance_from_home (km)_avg_30D'
    ]
    return engineer.anomalize(df, 'unsual_location', columns)


def unsual_device(df):
    # detect unsual devices
    columns = ['holder_device_count_frequency', 'holder_device_has_history', 'is_opening_device', 'holder_holder_device_count_frequency_avg_30D']
    return engineer.anomalize(df, 'unsual_device', columns)


def unsual_time(df):
    # detect unsual time
    columns = [
        'holder_hour_bound_frequency', 'holder_holder_hour_bound_frequency_avg_30D'
    ]
    return engineer.anomalize(df, 'unsual_time', columns)
