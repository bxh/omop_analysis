import pandas as pd
import os
import glob
import argparse
from tabulate import tabulate


def load_omop_data(data_path):
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    omop_tables = {}

    for file in csv_files:
        table_name = os.path.splitext(os.path.basename(file))[0]
        omop_tables[table_name] = pd.read_csv(file)
        print(
            f"Loaded {table_name} table with {omop_tables[table_name].shape[0]} rows and {omop_tables[table_name].shape[1]} columns."
        )

    return omop_tables


def map_concept_ids(df, column, concept_df):
    mapping = concept_df.set_index("concept_id")["concept_name"].to_dict()
    return df[column].map(mapping)


def unique_counts(df, table_name):
    print(f"\nUnique value counts for categorical columns in {table_name}:")
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    count_data = [(col, df[col].nunique()) for col in categorical_cols]
    print(tabulate(count_data, headers=["Column", "Unique Values"], tablefmt="grid"))


def run_analysis(data_path):
    omop_data = load_omop_data(data_path)

    if "CONCEPT" in omop_data:
        concept_df = omop_data["CONCEPT"]
    else:
        raise ValueError(
            "Concept table is required for mapping concept IDs to descriptions."
        )

    key_tables = [
        "PERSON",
        "VISIT_OCCURRENCE",
        "CONDITION_OCCURRENCE",
        "DRUG_EXPOSURE",
        "MEASURE",
        "OBSERVATION",
    ]

    for table_name in key_tables:
        if table_name in omop_data:
            df = omop_data[table_name]

            unique_counts(df, table_name)

            if table_name == "CONDITION_OCCURRENCE":
                df["condition_name"] = map_concept_ids(
                    df, "condition_concept_id", concept_df
                )
                print("\nCondition Type Distribution (with names):")
                print(
                    tabulate(
                        df["condition_name"].value_counts().reset_index(),
                        headers=["Condition Name", "Count"],
                        tablefmt="grid",
                    )
                )

            elif table_name == "DRUG_EXPOSURE":
                df["drug_name"] = map_concept_ids(df, "drug_concept_id", concept_df)
                print("\nDrug Type Distribution (with names):")
                print(
                    tabulate(
                        df["drug_name"].value_counts().reset_index(),
                        headers=["Drug Name", "Count"],
                        tablefmt="grid",
                    )
                )

            elif table_name == "MEASUREMENT":
                df["measurement_name"] = map_concept_ids(
                    df, "measurement_concept_id", concept_df
                )
                print("\nMeasurement Type Distribution (Lab Tests, with names):")
                print(
                    tabulate(
                        df["measurement_name"].value_counts().reset_index(),
                        headers=["Measurement Name", "Count"],
                        tablefmt="grid",
                    )
                )

                print("\nUnits of Measurement:")
                df["unit_name"] = map_concept_ids(df, "unit_concept_id", concept_df)
                print(
                    tabulate(
                        df["unit_name"].value_counts().reset_index(),
                        headers=["Unit Name", "Count"],
                        tablefmt="grid",
                    )
                )

                print(df["unit_name"].value_counts())
                print("\nValue Range for Measurements:")
                print(
                    tabulate(
                        df["value_as_number"].describe().reset_index(),
                        headers=["Statistic", "Value"],
                        tablefmt="grid",
                    )
                )

            elif table_name == "OBSERVATION":
                df["observation_name"] = map_concept_ids(
                    df, "observation_concept_id", concept_df
                )
                print("\nObservation Type Distribution (with names):")
                print(
                    tabulate(
                        df["observation_name"].value_counts().reset_index(),
                        headers=["Observation Name", "Count"],
                        tablefmt="grid",
                    )
                )

                print("\nObservation Value Distribution:")
                df["value_as_concept_name"] = map_concept_ids(
                    df, "value_as_concept_id", concept_df
                )
                print(
                    tabulate(
                        df["value_as_concept_name"].value_counts().reset_index(),
                        headers=["Observation Value", "Count"],
                        tablefmt="grid",
                    )
                )
    if "MEASUREMENT" in omop_data:
        measurement = omop_data["MEASUREMENT"]
        measurement["measurement_name"] = map_concept_ids(
            measurement, "measurement_concept_id", concept_df
        )

        print("\nTop 10 Measurement Types (Lab Tests, with names):")
        print(
            tabulate(
                measurement["measurement_name"].value_counts().head(10).reset_index(),
                headers=["Measurement Name", "Count"],
                tablefmt="grid",
            )
        )

        common_measurements = measurement["measurement_name"].value_counts().head(10).index
        avg_values = (
            measurement[measurement["measurement_name"].isin(common_measurements)]
            .groupby("measurement_name")["value_as_number"]
            .mean()
            .reset_index()  # Convert Series to DataFrame
        )
        print("\nAverage values for common measurements (with names):")
        print(
            tabulate(
                avg_values,
                headers=["Measurement Name", "Average Value"],
                tablefmt="grid",
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run analysis on OMOP data in specified directory."
    )
    parser.add_argument(
        "data_path", type=str, help="Path to the directory containing OMOP CSV files"
    )

    args = parser.parse_args()

    run_analysis(args.data_path)
