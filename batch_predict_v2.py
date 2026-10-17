import sys
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================================
# PRECISION-GOV BATCH PREDICTION ENGINE
# VERSION 2
# ============================================================================
#
# Purpose:
# Optimized batch prediction engine for:
#
# BMIS -> PHP -> PRECISION-GOV -> Random Forest
#
# FINAL MODEL:
#     Random Forest
#
# FINAL PREDICTORS:
#     age
#     water_access_encoded
#     electricity_access_encoded
#
# IMPORTANT PERFORMANCE OPTIMIZATION:
#
# SHAP explanations are calculated only for UNIQUE feature combinations.
#
# Because the final predictor set contains:
#
#     age                  = 6 to 24
#     water_access_encoded = 0, 1, 2
#     electricity_encoded  = 0, 1, 2
#
# there can be a maximum of:
#
#     19 x 3 x 3 = 171 unique combinations.
#
# SHAP values are deterministic for the same input combination, so the
# calculated explanation can safely be reused for residents with the same
# three predictor values.
#
# This dramatically reduces SHAP computation for large batches.
#
# The model is loaded only.
# The model is NOT retrained or modified.
#
# ============================================================================


# ============================================================================
# START TIMER
# ============================================================================

TOTAL_START_TIME = time.perf_counter()


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "precision_gov_random_forest_final.pkl"
)


# ============================================================================
# FINAL MODEL FEATURES
# ============================================================================

FEATURES = [
    "age",
    "water_access_encoded",
    "electricity_access_encoded"
]


EXPECTED_MODEL_FEATURES = [
    "age",
    "water_access_encoded",
    "electricity_access_encoded"
]


# ============================================================================
# MODEL VERSION
# ============================================================================

MODEL_VERSION = "v2"


# ============================================================================
# MODEL CLASS ENCODING
# ============================================================================

TARGET_LABELS = {
    0: "High Educational Risk",
    1: "Low Educational Risk",
    2: "Moderate Educational Risk"
}


EXPECTED_CLASSES = [
    0,
    1,
    2
]


# ============================================================================
# DECISION-SUPPORT PRIORITY
# ============================================================================

PRIORITY_MAPPING = {
    "High Educational Risk": 1,
    "Moderate Educational Risk": 2,
    "Low Educational Risk": 3
}


# ============================================================================
# CONFIDENCE CATEGORY
# ============================================================================

def get_confidence_category(confidence):

    confidence = float(
        confidence
    )

    if confidence >= 0.80:

        return "High Confidence"

    if confidence >= 0.60:

        return "Moderate Confidence"

    return "Low Confidence"


# ============================================================================
# ERROR HANDLER
# ============================================================================

def output_error(message):

    response = {
        "success": False,
        "error": str(message)
    }

    print(
        json.dumps(
            response,
            ensure_ascii=False
        )
    )

    sys.exit(1)


# ============================================================================
# VALIDATE MODEL FILE
# ============================================================================

if not os.path.exists(
    MODEL_PATH
):

    output_error(
        "Random Forest model not found: "
        + MODEL_PATH
    )


# ============================================================================
# LOAD MODEL
# ============================================================================

MODEL_LOAD_START_TIME = time.perf_counter()


try:

    model = joblib.load(
        MODEL_PATH
    )

except Exception as e:

    output_error(
        "Unable to load Random Forest model: "
        + str(e)
    )


MODEL_LOAD_TIME = (
    time.perf_counter()
    - MODEL_LOAD_START_TIME
)


# ============================================================================
# VALIDATE MODEL TYPE
# ============================================================================

model_class_name = type(
    model
).__name__


if model_class_name != "RandomForestClassifier":

    output_error(
        "Invalid model type. Expected "
        "RandomForestClassifier but found: "
        + model_class_name
    )


# ============================================================================
# VALIDATE MODEL FEATURES
# ============================================================================

if not hasattr(
    model,
    "feature_names_in_"
):

    output_error(
        "Loaded Random Forest model does not contain "
        "feature_names_in_."
    )


model_features = list(
    model.feature_names_in_
)


if model_features != EXPECTED_MODEL_FEATURES:

    output_error(
        "Model feature order mismatch. Expected: "
        + str(EXPECTED_MODEL_FEATURES)
        + " | Found: "
        + str(model_features)
    )


# ============================================================================
# VALIDATE MODEL CLASSES
# ============================================================================

if not hasattr(
    model,
    "classes_"
):

    output_error(
        "Loaded Random Forest model does not contain classes_."
    )


model_classes = [
    int(value)
    for value in model.classes_
]


if model_classes != EXPECTED_CLASSES:

    output_error(
        "Model class encoding mismatch. Expected: "
        + str(EXPECTED_CLASSES)
        + " | Found: "
        + str(model_classes)
    )


# ============================================================================
# VALIDATE MODEL INTERFACE
# ============================================================================

if not hasattr(
    model,
    "predict"
):

    output_error(
        "Loaded model does not provide predict()."
    )


if not hasattr(
    model,
    "predict_proba"
):

    output_error(
        "Loaded model does not provide predict_proba()."
    )


# ============================================================================
# READ JSON INPUT FROM PHP
# ============================================================================

try:

    input_text = sys.stdin.read()


    if not input_text.strip():

        output_error(
            "No prediction input was received."
        )


    input_data = json.loads(
        input_text
    )


except Exception as e:

    output_error(
        "Invalid JSON input: "
        + str(e)
    )


# ============================================================================
# VALIDATE INPUT STRUCTURE
# ============================================================================

if not isinstance(
    input_data,
    list
):

    output_error(
        "Prediction input must be a JSON array."
    )


if len(
    input_data
) == 0:

    output_error(
        "No records were supplied for prediction."
    )


# ============================================================================
# PREPARE FEATURE DATA
# ============================================================================

FEATURE_PREPARATION_START_TIME = time.perf_counter()


feature_rows = []

record_ids = []

record_references = []


try:

    for record_index, record in enumerate(
        input_data,
        start=1
    ):

        if not isinstance(
            record,
            dict
        ):

            raise ValueError(
                "Prediction record "
                + str(record_index)
                + " must be an object."
            )


        # ------------------------------------------------------------
        # REQUIRED FEATURES
        # ------------------------------------------------------------

        for feature in FEATURES:

            if feature not in record:

                raise ValueError(
                    "Prediction record "
                    + str(record_index)
                    + " is missing feature: "
                    + feature
                )


        # ------------------------------------------------------------
        # CONVERT FEATURES
        # ------------------------------------------------------------

        try:

            age = int(
                record["age"]
            )

            water = int(
                record["water_access_encoded"]
            )

            electricity = int(
                record["electricity_access_encoded"]
            )

        except Exception:

            raise ValueError(
                "Invalid numeric feature value in "
                "prediction record "
                + str(record_index)
                + "."
            )


        # ------------------------------------------------------------
        # VALIDATE AGE
        # ------------------------------------------------------------

        if age < 6 or age > 24:

            raise ValueError(
                "Age must be between 6 and 24. "
                "Invalid age in prediction record "
                + str(record_index)
                + ": "
                + str(age)
            )


        # ------------------------------------------------------------
        # VALIDATE WATER
        # ------------------------------------------------------------

        if water not in [0, 1, 2]:

            raise ValueError(
                "Invalid water_access_encoded value "
                "in prediction record "
                + str(record_index)
                + ": "
                + str(water)
            )


        # ------------------------------------------------------------
        # VALIDATE ELECTRICITY
        # ------------------------------------------------------------

        if electricity not in [0, 1, 2]:

            raise ValueError(
                "Invalid electricity_access_encoded value "
                "in prediction record "
                + str(record_index)
                + ": "
                + str(electricity)
            )


        # ------------------------------------------------------------
        # RESIDENT ID
        # ------------------------------------------------------------

        if record.get(
            "resident_id"
        ) is None:

            raise ValueError(
                "Missing resident_id in prediction record "
                + str(record_index)
                + "."
            )


        try:

            resident_id = int(
                record.get(
                    "resident_id"
                )
            )

        except Exception:

            raise ValueError(
                "Invalid resident_id in prediction record "
                + str(record_index)
                + "."
            )


        # ------------------------------------------------------------
        # RESIDENT REFERENCE
        # ------------------------------------------------------------

        resident_reference = record.get(
            "resident_reference"
        )


        if resident_reference is None:

            resident_reference = ""


        # ------------------------------------------------------------
        # STORE
        # ------------------------------------------------------------

        feature_rows.append(
            [
                age,
                water,
                electricity
            ]
        )


        record_ids.append(
            resident_id
        )


        record_references.append(
            str(
                resident_reference
            )
        )


except Exception as e:

    output_error(
        "Input validation failed: "
        + str(e)
    )


FEATURE_PREPARATION_TIME = (
    time.perf_counter()
    - FEATURE_PREPARATION_START_TIME
)


# ============================================================================
# CREATE DATAFRAME
# ============================================================================

try:

    feature_df = pd.DataFrame(
        feature_rows,
        columns=FEATURES
    )

except Exception as e:

    output_error(
        "Unable to create feature dataframe: "
        + str(e)
    )


# ============================================================================
# FINAL FEATURE ORDER CHECK
# ============================================================================

if list(
    feature_df.columns
) != EXPECTED_MODEL_FEATURES:

    output_error(
        "Prepared feature dataframe does not match "
        "the final predictor order."
    )


# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

PREDICTION_START_TIME = time.perf_counter()


try:

    predictions = model.predict(
        feature_df
    )


    probabilities = model.predict_proba(
        feature_df
    )


except Exception as e:

    output_error(
        "Prediction failed: "
        + str(e)
    )


PREDICTION_TIME = (
    time.perf_counter()
    - PREDICTION_START_TIME
)


# ============================================================================
# CREATE UNIQUE FEATURE COMBINATIONS
# ============================================================================
#
# This is the main V2 performance optimization.
#
# SHAP does not need to be calculated separately for residents having the
# exact same predictor values.
#
# Example:
#
# Resident A:
#     age = 15
#     water = 2
#     electricity = 2
#
# Resident B:
#     age = 15
#     water = 2
#     electricity = 2
#
# Their SHAP values are identical because their model input is identical.
#
# ============================================================================

UNIQUE_FEATURE_START_TIME = time.perf_counter()


unique_feature_keys = []

unique_feature_rows = []

unique_feature_lookup = {}


for row in feature_rows:

    feature_key = (
        int(row[0]),
        int(row[1]),
        int(row[2])
    )


    if feature_key not in unique_feature_lookup:

        unique_feature_lookup[
            feature_key
        ] = len(
            unique_feature_rows
        )


        unique_feature_keys.append(
            feature_key
        )


        unique_feature_rows.append(
            [
                feature_key[0],
                feature_key[1],
                feature_key[2]
            ]
        )


unique_feature_df = pd.DataFrame(
    unique_feature_rows,
    columns=FEATURES
)


UNIQUE_FEATURE_TIME = (
    time.perf_counter()
    - UNIQUE_FEATURE_START_TIME
)


# ============================================================================
# GENERATE SHAP EXPLANATIONS
# ============================================================================
#
# SHAP is now calculated ONLY for unique feature combinations.
#
# Maximum possible rows:
#
#     19 x 3 x 3 = 171
#
# rather than the full resident population.
#
# ============================================================================

SHAP_START_TIME = time.perf_counter()


try:

    explainer = shap.TreeExplainer(
        model
    )


    try:

        shap_output = explainer.shap_values(
            unique_feature_df,
            check_additivity=False
        )

    except TypeError:

        shap_output = explainer.shap_values(
            unique_feature_df
        )


except Exception as e:

    output_error(
        "SHAP explanation generation failed: "
        + str(e)
    )


SHAP_TIME = (
    time.perf_counter()
    - SHAP_START_TIME
)


# ============================================================================
# SHAP OUTPUT NORMALIZATION
# ============================================================================

def get_shap_values_for_prediction(
    shap_values,
    row_index,
    predicted_class,
    classes
):

    predicted_class = int(
        predicted_class
    )


    if predicted_class not in classes:

        raise ValueError(
            "Predicted class "
            + str(predicted_class)
            + " is not present in model.classes_."
        )


    class_position = classes.index(
        predicted_class
    )


    # ----------------------------------------------------------------
    # LIST OF ARRAYS
    # ----------------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        if class_position >= len(
            shap_values
        ):

            raise ValueError(
                "SHAP class position is outside "
                "the returned SHAP value range."
            )


        class_values = np.asarray(
            shap_values[
                class_position
            ]
        )


        if class_values.ndim != 2:

            raise ValueError(
                "Unexpected SHAP class array dimensions."
            )


        return class_values[
            row_index
        ]


    # ----------------------------------------------------------------
    # NUMPY ARRAY
    # ----------------------------------------------------------------

    shap_array = np.asarray(
        shap_values
    )


    # ----------------------------------------------------------------
    # samples x features x classes
    # ----------------------------------------------------------------

    if shap_array.ndim == 3:

        return shap_array[
            row_index,
            :,
            class_position
        ]


    # ----------------------------------------------------------------
    # samples x features
    # ----------------------------------------------------------------

    if shap_array.ndim == 2:

        return shap_array[
            row_index
        ]


    raise ValueError(
        "Unsupported SHAP output format. "
        "Dimensions: "
        + str(
            shap_array.shape
        )
    )


# ============================================================================
# PREPARE UNIQUE SHAP CACHE
# ============================================================================
#
# The cache stores SHAP explanations by:
#
#     (age, water, electricity, predicted_class)
#
# This guarantees that each resident receives the correct explanation.
#
# ============================================================================

SHAP_CACHE_START_TIME = time.perf_counter()


classes = [
    int(value)
    for value in model.classes_
]


shap_cache = {}


for unique_index, feature_key in enumerate(
    unique_feature_keys
):

    # ------------------------------------------------------------
    # Find prediction for this unique combination
    # ------------------------------------------------------------

    unique_prediction = model.predict(
        unique_feature_df.iloc[
            [unique_index]
        ]
    )[0]


    predicted_class = int(
        unique_prediction
    )


    # ------------------------------------------------------------
    # Extract SHAP values
    # ------------------------------------------------------------

    shap_values_for_prediction = (
        get_shap_values_for_prediction(
            shap_output,
            unique_index,
            predicted_class,
            classes
        )
    )


    shap_values_for_prediction = np.asarray(
        shap_values_for_prediction
    ).reshape(
        -1
    )


    if len(
        shap_values_for_prediction
    ) != len(FEATURES):

        raise ValueError(
            "SHAP feature count mismatch for "
            + str(feature_key)
            + ". Expected "
            + str(len(FEATURES))
            + " features but received "
            + str(len(shap_values_for_prediction))
            + "."
        )


    shap_cache[
        (
            feature_key[0],
            feature_key[1],
            feature_key[2],
            predicted_class
        )
    ] = shap_values_for_prediction


SHAP_CACHE_TIME = (
    time.perf_counter()
    - SHAP_CACHE_START_TIME
)


# ============================================================================
# PREPARE RESULTS
# ============================================================================

RESULT_PREPARATION_START_TIME = time.perf_counter()


results = []


try:

    for index in range(
        len(predictions)
    ):

        # ------------------------------------------------------------
        # PREDICTED CLASS
        # ------------------------------------------------------------

        predicted_encoded = int(
            predictions[index]
        )


        # ------------------------------------------------------------
        # PREDICTED RISK
        # ------------------------------------------------------------

        predicted_risk = TARGET_LABELS.get(
            predicted_encoded
        )


        if predicted_risk is None:

            raise ValueError(
                "Unknown predicted risk encoding: "
                + str(
                    predicted_encoded
                )
            )


        # ------------------------------------------------------------
        # PROBABILITIES
        # ------------------------------------------------------------

        probability_values = probabilities[
            index
        ]


        probability_map = {}


        for class_index, class_value in enumerate(
            classes
        ):

            probability_map[
                int(class_value)
            ] = float(
                probability_values[
                    class_index
                ]
            )


        high_probability = probability_map.get(
            0,
            0.0
        )


        low_probability = probability_map.get(
            1,
            0.0
        )


        moderate_probability = probability_map.get(
            2,
            0.0
        )


        # ------------------------------------------------------------
        # CONFIDENCE
        # ------------------------------------------------------------

        confidence = max(
            high_probability,
            low_probability,
            moderate_probability
        )


        confidence_category = get_confidence_category(
            confidence
        )


        # ------------------------------------------------------------
        # PRIORITY
        # ------------------------------------------------------------

        decision_support_priority = PRIORITY_MAPPING.get(
            predicted_risk,
            3
        )


        # ------------------------------------------------------------
        # HIGH-RISK FLAG
        # ------------------------------------------------------------

        model_predicted_high_risk = (
            predicted_encoded == 0
        )


        # ------------------------------------------------------------
        # FEATURE KEY
        # ------------------------------------------------------------

        current_feature_key = (
            int(feature_rows[index][0]),
            int(feature_rows[index][1]),
            int(feature_rows[index][2]),
            predicted_encoded
        )


        # ------------------------------------------------------------
        # GET CACHED SHAP VALUES
        # ------------------------------------------------------------

        if current_feature_key not in shap_cache:

            raise ValueError(
                "SHAP cache entry not found for resident "
                + str(
                    record_ids[index]
                )
            )


        shap_values_for_prediction = shap_cache[
            current_feature_key
        ]


        # ------------------------------------------------------------
        # BUILD SHAP EXPLANATIONS
        # ------------------------------------------------------------

        shap_explanations = []


        for feature_index, feature_name in enumerate(
            FEATURES
        ):

            shap_value = float(
                shap_values_for_prediction[
                    feature_index
                ]
            )


            feature_value = feature_df.iloc[
                index
            ][
                feature_name
            ]


            shap_explanations.append(
                {
                    "feature_name":
                        feature_name,

                    "feature_value":
                        str(
                            feature_value
                        ),

                    "shap_value":
                        round(
                            shap_value,
                            6
                        ),

                    "abs_shap_value":
                        round(
                            abs(
                                shap_value
                            ),
                            6
                        )
                }
            )


        # ------------------------------------------------------------
        # RANK SHAP FEATURES
        # ------------------------------------------------------------

        shap_explanations.sort(
            key=lambda item:
                item[
                    "abs_shap_value"
                ],
            reverse=True
        )


        for rank, explanation in enumerate(
            shap_explanations,
            start=1
        ):

            explanation[
                "feature_rank"
            ] = rank


        # ------------------------------------------------------------
        # FINAL RESULT
        # ------------------------------------------------------------

        results.append(
            {
                "resident_id":
                    record_ids[index],

                "resident_reference":
                    record_references[index],

                "predicted_risk":
                    predicted_risk,

                "predicted_risk_encoded":
                    predicted_encoded,

                "probability_high_risk":
                    round(
                        high_probability,
                        6
                    ),

                "probability_low_risk":
                    round(
                        low_probability,
                        6
                    ),

                "probability_moderate_risk":
                    round(
                        moderate_probability,
                        6
                    ),

                "prediction_confidence":
                    round(
                        confidence,
                        6
                    ),

                "prediction_confidence_category":
                    confidence_category,

                "decision_support_priority":
                    decision_support_priority,

                "model_predicted_high_risk":
                    model_predicted_high_risk,

                "shap_explanations":
                    shap_explanations
            }
        )


except Exception as e:

    output_error(
        "Unable to prepare prediction and SHAP results: "
        + str(e)
    )


RESULT_PREPARATION_TIME = (
    time.perf_counter()
    - RESULT_PREPARATION_START_TIME
)


# ============================================================================
# TOTAL PROCESSING TIME
# ============================================================================

TOTAL_PROCESSING_TIME = (
    time.perf_counter()
    - TOTAL_START_TIME
)


# ============================================================================
# MODEL METADATA
# ============================================================================

model_metadata = {

    "name":
        "Random Forest",

    "type":
        type(model).__name__,

    "model_file":
        "precision_gov_random_forest_final.pkl",

    "predictors":
        FEATURES,

    "classes":
        {
            "0":
                "High Educational Risk",

            "1":
                "Low Educational Risk",

            "2":
                "Moderate Educational Risk"
        },

    "n_estimators":
        int(
            getattr(
                model,
                "n_estimators",
                0
            )
        ),

    "random_state":
        getattr(
            model,
            "random_state",
            None
        ),

    "class_weight":
        getattr(
            model,
            "class_weight",
            None
        ),

    "version":
        MODEL_VERSION,

    "retrained":
        False
}


# ============================================================================
# PROCESSING METADATA
# ============================================================================

processing_metadata = {

    "records_processed":
        len(results),

    "unique_feature_combinations":
        len(unique_feature_rows),

    "maximum_possible_feature_combinations":
        171,

    "model_load_seconds":
        round(
            MODEL_LOAD_TIME,
            6
        ),

    "feature_preparation_seconds":
        round(
            FEATURE_PREPARATION_TIME,
            6
        ),

    "prediction_seconds":
        round(
            PREDICTION_TIME,
            6
        ),

    "unique_feature_processing_seconds":
        round(
            UNIQUE_FEATURE_TIME,
            6
        ),

    "shap_seconds":
        round(
            SHAP_TIME,
            6
        ),

    "shap_cache_seconds":
        round(
            SHAP_CACHE_TIME,
            6
        ),

    "result_preparation_seconds":
        round(
            RESULT_PREPARATION_TIME,
            6
        ),

    "total_seconds":
        round(
            TOTAL_PROCESSING_TIME,
            6
        )
}


# ============================================================================
# FINAL RESPONSE
# ============================================================================

response = {

    "success":
        True,

    "model":
        model_metadata,

    "processing":
        processing_metadata,

    "records_processed":
        len(results),

    "predictions":
        results
}


print(
    json.dumps(
        response,
        ensure_ascii=False
    )
)