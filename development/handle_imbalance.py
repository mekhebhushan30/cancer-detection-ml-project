from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def handle_imbalance(
    X_train,
    y_train,
    method
):

    if method == "smote":

        sampler = SMOTE(
            random_state=42
        )

    elif method == "random_over":

        sampler = RandomOverSampler(
            random_state=42
        )

    elif method == "random_under":

        sampler = RandomUnderSampler(
            random_state=42
        )

    else:

        raise ValueError(
            "Invalid imbalance handling method. "
            "Use: smote, random_over, random_under"
        )

    X_train_resampled, y_train_resampled = sampler.fit_resample(
        X_train,
        y_train
    )

    return X_train_resampled, y_train_resampled