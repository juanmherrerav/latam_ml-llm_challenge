from datetime import datetime
import pandas as pd

def get_period_day(date):
    """_summary_

    Args:
        date (_type_): _description_

    Returns:
        _type_: _description_
    """
    date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
    morning_min = datetime.strptime("05:00", "%H:%M").time()
    morning_max = datetime.strptime("11:59", "%H:%M").time()
    afternoon_min = datetime.strptime("12:00", "%H:%M").time()
    afternoon_max = datetime.strptime("18:59", "%H:%M").time()
    evening_min = datetime.strptime("19:00", "%H:%M").time()
    evening_max = datetime.strptime("23:59", "%H:%M").time()
    night_min = datetime.strptime("00:00", "%H:%M").time()
    night_max = datetime.strptime("4:59", "%H:%M").time()

    if date_time > morning_min and date_time < morning_max:
        return "mañana"
    elif date_time > afternoon_min and date_time < afternoon_max:
        return "tarde"
    elif (date_time > evening_min and date_time < evening_max) or (
        date_time > night_min and date_time < night_max
    ):
        return "noche"


def is_high_season(fecha):
    fecha_año = int(fecha.split("-")[0])
    fecha = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
    range1_min = datetime.strptime("15-Dec", "%d-%b").replace(year=fecha_año)
    range1_max = datetime.strptime("31-Dec", "%d-%b").replace(year=fecha_año)
    range2_min = datetime.strptime("1-Jan", "%d-%b").replace(year=fecha_año)
    range2_max = datetime.strptime("3-Mar", "%d-%b").replace(year=fecha_año)
    range3_min = datetime.strptime("15-Jul", "%d-%b").replace(year=fecha_año)
    range3_max = datetime.strptime("31-Jul", "%d-%b").replace(year=fecha_año)
    range4_min = datetime.strptime("11-Sep", "%d-%b").replace(year=fecha_año)
    range4_max = datetime.strptime("30-Sep", "%d-%b").replace(year=fecha_año)

    if (
        (fecha >= range1_min and fecha <= range1_max)
        or (fecha >= range2_min and fecha <= range2_max)
        or (fecha >= range3_min and fecha <= range3_max)
        or (fecha >= range4_min and fecha <= range4_max)
    ):
        return 1
    else:
        return 0


def get_min_diff(data: pd.DataFrame):
    """Get the minimum difference between fecha_o original flight date and fecha_i actual flight date

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
    fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
    min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
    return min_diff


# def get_top_features(model: xgb.Booster, feature_names: list, top_n: int = 10) -> list:
#     """
#     Get the top n feature names based on feature importance from an XGBoost model.

#     Args:
#         model (xgb.Booster): Trained XGBoost Booster model.
#         feature_names (list): List of feature names corresponding to the model.
#         top_n (int): Number of top features to retrieve.

#     Returns:
#         list: List of top feature names ordered by importance.
#     """
#     # Retrieve the feature importance dictionary
#     importance_dict = model.get_score(importance_type="weight")

#     # Map feature indices to names and importance scores
#     importance_list = [
#         (feature_names[int(k[1:])], v) for k, v in importance_dict.items()
#     ]

#     # Sort features by importance and extract the top n names
#     top_features = [
#         name
#         for name, _ in sorted(importance_list, key=lambda x: x[1], reverse=True)[:top_n]
#     ]
#     return top_features
