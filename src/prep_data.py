import os
import glob
import pandas as pd
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from util.util import XyzTuple, xyz2irc, enumerateWithEstimate

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(base_dir, "data")
annotations_file = os.path.join(data_dir, "annotations.csv")
candidates_file = os.path.join(data_dir, "candidates.csv")
candidates_with_diameters_file = os.path.join(data_dir, "candidates_with_dia.csv")


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
                if all([abs(i - j) <= (dia/4) for i, j in zip([a_x, a_y, a_z], [x, y, z])]):
                    diameters[idx] = dia
                    break
            candidates_df["diameters"] = diameters
            candidates_df.sort_values("diameters", inplace=True)
            candidates_df.to_csv(candidates_with_diameters_file, index=False)

        else:
            print("Found existing file and no override flag.")


class Ct:
    def __init__(self, series_uid):
        self.series_uid = series_uid
        mhd_path = glob.glob(f"{data_dir}/subset*/{series_uid}.mhd")[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_array = np.asarray(sitk.GetArrayFromImage(ct_mhd), np.float32)

        self.ct = np.clip(ct_array, -1000, 1000)

        self.origin = XyzTuple(*ct_mhd.GetOrigin())
        self.voxel_size = XyzTuple(*ct_mhd.GetSpacing())
        self.direction = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_ct_slice(self, slice_dimensions, center_xyz):
        center_irc = xyz2irc(
            center_xyz,
            self.origin,
            self.voxel_size,
            self.direction
        )

        ct_shape = self.ct.shape
        slice_list = []
        for i, dim in enumerate(slice_dimensions):
            start_idx = center_irc[i] + dim//2
            end_idx = start_idx + dim

            if start_idx < 0:
                start_idx = 0
                end_idx = dim

            if end_idx > ct_shape[i]:
                start_idx = ct_shape[i] - dim
                end_idx = ct_shape[i]

            slice_list.append(slice(start_idx, end_idx))

        ct_slice = self.ct[tuple(slice_list)]
        return ct_slice, center_irc


class LunaDataset(Dataset):
    def __init__(self, is_val_set=False, val_stride=10):
        df = pd.read_csv(candidates_with_diameters_file)
        self.ct_slice_size = [32, 48, 48]
        if is_val_set:
            self.data = df.iloc[::val_stride, :].reset_index(drop=True)
        else:
            self.data = df.drop(df.index[::val_stride]).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seriesuid, coord_x, coord_y, coord_z, nodule_class, _ = self.data.iloc[idx, :]
        ct_slice, center_irc = Ct(seriesuid).get_ct_slice(
            self.ct_slice_size,
            [coord_x, coord_y, coord_z])

        ct_tensor = torch.tensor(ct_slice, dtype=torch.float32)
        ct_tensor = ct_tensor.unsqueeze(0)

        label_tensor = torch.tensor(
            [not nodule_class, nodule_class],
            dtype=torch.long)

        return (
            ct_tensor,
            label_tensor,
            seriesuid,
            center_irc
        )


if __name__ == '__main__':
    train_dl = DataLoader(LunaDataset(), batch_size=16, num_workers=8, pin_memory=True)
    for idx, batch in enumerateWithEstimate(train_dl, "Training DL"):
        if idx > 2:
            break
        ct_t, label_t, series_uid, center_irc = batch
        print(f"{len(series_uid)} elements in batch")
        print(ct_t[0].shape)
        print(label_t.shape)

