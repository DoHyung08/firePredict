import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def visualize_model_output(dataset, model, device='cpu', img_size=128):
    model.eval()
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    inputs, labels = next(iter(dataloader))  # 배치 하나 가져오기

    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
    
    
    # info_names = ['date','lon_min','lon_max','lat_min','lat_max']
    # info_values = inputs[0, 1:1+len(info_names), 0, 0].cpu().numpy()  # (broadcast된 채널에서 하나의 좌표값만 대표로 추출)
    feature_names = dataset.feature_cols
    feature_values = inputs[0, 1:1+len(feature_names), 0, 0].cpu().numpy()  # (broadcast된 채널에서 하나의 좌표값만 대표로 추출)
    
    # print("\n📊 Input Infos:")
    # for name, value in zip(info_names, info_values):
    #     print(f"  {name:25s}: {value:.7f}")
    
    print("\n📊 Input Features:")
    for name, value in zip(feature_names, feature_values):
        print(f"  {name:25s}: {value:.7f}")
        
    
    # (1, C, H, W) -> squeeze
    input_img = inputs[0, 0].cpu().numpy()  # 입력 이미지 (그레이스케일)
    pred_mask = outputs[0, 0].cpu().numpy()  # 예측 마스크 (0~1)
    true_mask = labels[0, 0].cpu().numpy()  # 실제 마스크 (0 or 1)

    # 안전하게 변환
    pred_mask = np.nan_to_num(pred_mask, nan=0.0, posinf=1.0, neginf=0.0)
    true_mask = np.nan_to_num(true_mask, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 시각화
    plt.figure(figsize=(12, 4))
    
    assert not np.isnan(pred_mask).any(), "pred_mask contains NaN"
    assert not np.isinf(pred_mask).any(), "pred_mask contains Inf"
    assert not np.isnan(true_mask).any(), "true_mask contains NaN"
    assert not np.isinf(true_mask).any(), "true_mask contains Inf"
    assert not np.isnan(input_img).any(), "input_img contains NaN"
    assert not np.isinf(input_img).any(), "input_img contains Inf"
        
    print("Input min/max:", np.min(input_img), np.max(input_img))
    print("Pred min/max:", np.min(pred_mask), np.max(pred_mask))
    print("True min/max:", np.min(true_mask), np.max(true_mask))
    
    print("input_img shape:", input_img.shape)
    print("pred_mask shape:", pred_mask.shape)
    print("true_mask shape:", true_mask.shape)

    
    plt.subplot(1, 3, 1)
    plt.title("Input (Prev Firemask)")
    plt.imshow(input_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Model Output (Pred)")
    plt.imshow(pred_mask, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth (Label)")
    plt.imshow(true_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    

from model import UNet
from data_utils import WildfireRowDatasetV2
import numpy as np

# encoder_channels_list = [[32, 64], [32, 64, 128],[32, 64, 128], [32, 64, 128, 256]]
# decoder_channels_list = [[64, 32], [128, 64, 32],[128, 64, 32], [256, 128, 64, 32]]
# kernel_sizes = [[3,3,3,3,3],[3,3,3,3,3,3,3],[7,5,3,3,3,5,7],[3,3,3,3,3,3,3,3,3]]
# img_size = [128,128,256,128]
# save_paths = ['0715_1','0715_2','0715_3','0715_4']

encoder_channels_list = [[32, 64, 128],[64, 128,256], [32, 64, 128, 256], [32, 64, 128, 256]]
decoder_channels_list = [[128, 64, 32],[256,128, 64], [256, 128, 64, 32], [256, 128, 64, 32]]
kernel_sizes = [[3,3,3,5,3,3,3],[3,3,3,7,3,3,3],[3,3,3,5,7,5,3,3,3],[7,3,3,5,7,5,3,3,7]]
img_size = [128,256,256,256]
save_paths = ['0715_5','0715_6','0715_7','0715_8']

hyperparams = [[encoder_channels_list[i], decoder_channels_list[i], kernel_sizes[i], img_size[i]] for i in range(0,4)]

i = 0

# 모델과 데이터셋 생성
model = UNet(
    input_channels=16,
    encoder_channels=hyperparams[i][0],
    decoder_channels=hyperparams[i][1],
    kernel_sizes=hyperparams[i][2],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device:{device}')
model.load_state_dict(torch.load(save_paths[i]+"/model_best.pth", map_location=device))
model.to(device)


dataset = WildfireRowDatasetV2(csv_path="data/firemask_modis_wind_weather_end_2day.csv", 
                               image_base_path="data", 
                               img_size=hyperparams[i][3],)

# 시각화 실행
visualize_model_output(dataset, model, device=device, img_size=hyperparams[i][3])
