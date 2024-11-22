from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from challenge.model_utils import get_min_diff


class DelayModel:
    """A class method that implements the DelayModel ."""

    def __init__(self):
        # self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
        self._model = None  # Model should be saved in this attribute.
        self.threshold_in_minutes = 15
        self.top_features_n = 10
        self.top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]
        self.target = ["delay"]

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        data["min_diff"] = data.apply(get_min_diff, axis=1)
        data["delay"] = np.where(data["min_diff"] > self.threshold_in_minutes, 1, 0)

        # Feature encoding
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )
        filtered_features_with_importance_df = features[self.top_10_features]

        target_df = (
            pd.DataFrame(data[target_column]) if target_column is not None else pd.DataFrame(data["delay"])
        )

        # x_train, _, y_train, _ = train_test_split(
        #    features, target_df, test_size=0.33, random_state=42
        # )

        # Train the initial XGBoost model
        # xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        # xgb_model.fit(x_train, y_train)

        # Select the top 10 features
        # xgb_fea_imp = (
        ##    pd.DataFrame(
        #        list(xgb_model.get_booster().get_fscore().items()),
        ##        columns=["feature", "importance"],
        #    )
        #    .sort_values("importance", ascending=False)
        #    .head(self.top_features_n)
        # )
        # print("", xgb_fea_imp["feature"].to_list())

        # Filter the original dataset for these features + target column
        # filtered_features_with_importance_df = data[xgb_fea_imp["feature"].to_list()]

        if target_column is None:
            return filtered_features_with_importance_df
        else:
            return filtered_features_with_importance_df, target_df

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        n_y0 = target["delay"].value_counts()[0]
        # n_y0 = len(target[y_train == 0])
        n_y1 = target["delay"].value_counts()[1]
        # n_y1 = len(target[y_train == 1])
        scale = n_y0 / n_y1
        print("Scale =" + str(scale))

        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale
        )
        self._model.fit(features, target)
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.#

        Args:
            features (pd.DataFrame): preprocessed data.
        Returns:
            (List[int]): predicted targets.
        """
        response = self._model.predict(features)
        print("Response_type"+type(response))
        return response
