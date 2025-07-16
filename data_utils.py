# 코드 상태 초기화로 인해 다시 필요한 import 및 정의 실행

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from PIL import Image
import os
import torch.optim as optim


def firemask_name_process(name):
    
    return name
    
    if(name == 'firemask_output\white.png'):
        return name
    
    parts = name.split('_')
    
    
    date = parts[2]
    date = date[:4] + '-' + date[4:6] + '-' + date[6:8]
    
    return f"{parts[0]}_{parts[1]}_{date}_{parts[3]}_{parts[4]}"


class WildfireRowDatasetV2(Dataset):
    def __init__(self, csv_path, image_base_path, img_size=128, log_path="data/missing_images.log"):
        self.image_base_path = image_base_path
        self.img_size = img_size
        self.feature_cols = [
            'prev_fire_point', 'today_point_count',
            'prev_frp_mean', 'today_frp_mean',
            'NDVI', 'EVI',
            'wind_dir', 'wind_speed',
            'stn_id', 'TA_AVG', 'TA_MAX', 'TA_MIN',
            'HM_AVG', 'RN_DAY', 'altitude'
        ]

        raw_data = pd.read_csv(csv_path)

        with open(log_path, "w") as log_file:
            valid_rows = []
            for idx, row in raw_data.iterrows():
                prev_path = os.path.join(self.image_base_path, firemask_name_process(row['prev_firemask']))
                label_path = os.path.join(self.image_base_path, firemask_name_process(row['today_firemask']))

                # NaN 포함 여부 확인
                has_nan = row[self.feature_cols].isna().any()

                # 이미지 존재 여부 확인
                prev_exists = os.path.exists(prev_path)
                label_exists = os.path.exists(label_path)

                if has_nan:
                    log_file.write(
                        f"[NaN Found] index={idx} | columns with NaN: {row[self.feature_cols].isna()[row[self.feature_cols].isna()].index.tolist()}\n"
                    )
                elif not (prev_exists and label_exists):
                    log_file.write(
                        f"[Missing Image] index={idx} | prev={row['prev_firemask']} | today={row['today_firemask']}\n"
                    )
                else:
                    valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 입력 이미지 로딩
        prev_mask_path = os.path.join(self.image_base_path, firemask_name_process(row['prev_firemask']))
        prev_img = Image.open(prev_mask_path).convert("L").resize((self.img_size, self.img_size))
        prev_array = 255.0 - np.array(prev_img).astype(np.float32)
        prev_img_tensor = torch.from_numpy(prev_array).unsqueeze(0) / 255.0

        # 수치 feature broadcasting
        features = torch.tensor([row[col] for col in self.feature_cols], dtype=torch.float32)
        feature_tensor = features.view(-1, 1, 1).expand(-1, self.img_size, self.img_size)

        input_tensor = torch.cat([prev_img_tensor, feature_tensor], dim=0)  # (19, H, W)

        # 정답 마스크
        label_path = os.path.join(self.image_base_path, firemask_name_process(row['today_firemask']))
        label_img = Image.open(label_path).convert("L").resize((self.img_size, self.img_size))
        label_tensor = torch.from_numpy((np.array(label_img) < 127).astype(np.float32)).unsqueeze(0)

        return input_tensor, label_tensor
