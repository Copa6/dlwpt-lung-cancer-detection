import os
import pandas as pd
import numpy as np

annotations_file = "../data/annotations.csv"
candidates_file = "../data/candidates.csv"
candidates_with_diameters_file = "../data/candidates_with_dia.csv"

def add_diameter_to_candidates(override=False):
    if not(os.path.exists(candidates_with_diameters_file)) or override:
        annotations_df = pd.read_csv(annotations_file)
        candidates_df = pd.read_csv(candidates_file)

        diameters = [0] * len(candidates_df)
        for idx, row in candidates_df.iterrows():
            s_uid, x, y, z, label = row
            diameter_table = annotations_df.loc[annotations_df["seriesuid"] == s_uid]
            for _, a_row in diameter_table.iterrows():
                _a, a_x, a_y, a_z, dia = a_row
                if all([abs(i - j) < j / 4 for i, j in zip([a_x, a_y, a_z], [x, y, z])]):
                    diameters[idx] = dia
                    break
            candidates_df["diameters"] = diameters
            candidates_df.sort_values("diameters", inplace=True)
            candidates_df.to_csv(candidates_with_diameters_file, index=False)

        else:
            print("Found existing file and no override flag.")






