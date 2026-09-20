# TopAneu 2026 Task2 r5 submission package

此包使用 CURRENT_BEST_DELIVERY 的 MRofficial7_S32corrected_CTE16_r5。原 r5 封存包不修改。

## 結構與模型
- `models/config.json`：固定推論設定總表（完整參數由 configs、模型 plans 及原始程式實作）。
- `models/detector_CT`, `models/detector_MR`：nnDetection 權重、plan.pkl、config.yaml。
- `models/segmentation_CT.pt`, `models/segmentation_MR.pt`：CT E16 / 修正版 MR S32 自訂分割。
- `models/location_*.joblib`, `models/filter_*`：固定位置分類及 CT 過濾模型。
- `models/ta36/*/{plans.json,dataset.json,fold_4/checkpoint_final.pth}`：三模型共享血管解剖 ensemble。
- `configs/location_mapping.json`：標籤0–52。
- `inference.py`：GC入口；`run_inference.py`：既有單影像流程。
- `postprocessing.py`：輸出幾何及標籤檢查。原演算法後處理保留於 `source/scripts/delivery/refine_dense_fusion.py`、`source/scripts/astra6_e32/common.py`、`source/scripts/astra6_e13/infer.py`、`source/scripts/delivery/refine.py`。

這不是 CT/MR 各一套 nnU-Net；不偽造 DatasetXXX 目錄。nnDetection 的 `model_last.ckpt` 是 r5 實際部署選定權重，不是額外附帶的中途訓練快照。MR 閾值、CT cascade、connected-component/fusion/既有 CT empty-mask fallback 均保留；未改演算法。部分 train.py 定義部署網路，必須保留程式碼，但無訓練資料、optimizer、scheduler、AMP訓練狀態、RNG快照、預測或 .git。MR 非預設的 legacy/fusion 支援權重保留以維持既有入口完整性。

## 官方 I/O
核對於2026-09-20：
https://github.com/Bangulli/TopAneu-26/blob/main/templates/task2/main.py
https://github.com/Bangulli/TopAneu-26/blob/main/README.md

讀取 `/input/inputs.json`；CT socket `head-ct-angiography` → `/input/images/head-ct-angio`；MR socket `head-mr-angiography` → `/input/images/head-mr-angio`。每次單一 scalar 3D volume。
輸出 `/output/images/aneurysm-segmentation/output.mha`，uint8，0背景、1–52動脈瘤位置類別。不是獨立的全血管分割提交；TA36只在內部使用。
入口先把影像寫成 NIfTI 供原流程使用，再驗證 shape、spacing、origin、direction、實體角點與標籤，確認只有NIfTI浮點header舍入誤差後寫成精確沿用原輸入geometry的MHA；不為掩蓋錯位而重採樣。暫存只在 /tmp，結束後清除。

## 建置與本地使用
特殊需求為兩個隔離環境：detector Python3.9/Torch1.11 CUDA11.3，refinement Python3.11/Torch2.5.1 CUDA12.4。requirements 分列於 environment；完整舊環境快照亦附上。TA36使用包內vendor fork；不要以其他nnUNet實作替換。nnDetection CUDA NMS需編譯，Dockerfile包含來源建置並指定T4 sm75，未附帶舊機器編譯.so。

```bash
docker build -t topaneu-task2:r5 .
docker run --rm --network none --gpus 'device=0' --memory=31g --shm-size=1g --mount type=bind,src="$(pwd)/test/input",dst=/input,readonly --mount type=bind,src="$(pwd)/test/output",dst=/output topaneu-task2:r5
```
本地 test/output 必須可讓 uid1000 寫入。正式720秒/31GB主RAM限制須在T4完成全流程測試；以上命令不自動施加720秒截止，也不構成通過保證。CUDA12.4 runtime需要相容宿主機driver。

使用既有環境時：
```bash
TASK2_DETECTOR_PYTHON=/path/to/nndet/bin/python TASK2_REFINEMENT_PYTHON=/path/to/nnunet/bin/python TASK2_GPU=0 /path/to/nnunet/bin/python inference.py --input-dir /path/input --output-dir /path/output
```
既有 nnDetection 環境必須已有相容 `nndet._C`；Docker建置則會在包內編譯。原始NIfTI CLI見 `python run_inference.py --help`。

## 驗證與限制
權重已CPU載入，去除訓練狀態並逐tensor bitwise核對。`environment/WEIGHT_STRIP_VERIFICATION.json`與`PACKAGE_MANIFEST.json`提供來源/新檔hash。幾何及socket adapter有CPU合成案例測試，不代表實際GPU模型推論通過。
**本次未建置Docker、未執行模型推論、未驗證T4 16GB/720秒/31GB RAM、未提交GC。** Dockerfile是待建置配方，依賴安裝及CUDA extension仍須在目標建置機验证。原r5/r4驗證差異見environment/ORIGINAL_R5_README.md；本地反覆開發成績不是hidden test成績。

## GitHub
已附 .gitignore 與 .gitattributes。先安裝/啟用 Git LFS，再加入模型；不要把zip直接提交普通Git。此包不含repository remote或credentials，未push。訓練/研究與monitor維持停止。
