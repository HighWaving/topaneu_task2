from pathlib import Path
import json,shutil
P=Path(__file__).resolve().parents[2];R=P/'delivery_20260909';O=P/'artifacts/current_official_20260909';E=P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z'
read=lambda p:json.loads(p.read_text());comp=read(O/'paired_comparison.json');diag=read(E/'evaluation/extended_diagnostics.json');metrics=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
text='''# TopAneu Task2 — 2026-09-09 研發與驗收報告

已完成 E04 候選式3D分割器的預定訓練、MR40整體推論、最新版官方六指標、配對比較與病灶診斷。採用 E04 作目前已驗證的工作區最佳版本，保留 E01/E02/E03 全部歷史產物。交付資源與容器驗證另列，不以研究評估通過等同競賽容器完成。

## 官方 evaluator 版本修正

本地歷史 evaluate.py SHA256為5e24667a…，與現行官方不同。現已封存並直接使用官方 commit `60765a5159a2cd517038a60ab26b5e091a204f4e`，其 TopBrain dependency commit為 `03ba48e47790609317916bacfea8904b596cd888`。固定numpy1.25.2與SimpleITK2.2.1，13項官方評分函式測試通過。4項CLI測試因本環境被偵測為Docker而尋找不存在的/input，不能算通過；本次直接呼叫官方函式的 execute_in_docker=False 入口，未修改評分實作。

新版依case/class是否存在計算TP/FP/FN/TN，各指標忽略NaN平均；有效類別數隨指標不同。HD95單位為mm，非舊版對角線正規化值。六項原始數值不能平均為官方總分，未推估未知隊伍排名。E01只重算保存遮罩，未重做模型推論。

來源：[官方Task2 evaluator](https://github.com/Bangulli/TopAneu-26/tree/60765a5159a2cd517038a60ab26b5e091a204f4e/eval/task2)、[參賽規格](https://topaneu-26.grand-challenge.org/participation/)。

## 固定 MR40 比較

| 版本 | Precision↑ | Recall↑ | MCC↑ | Dice↑ | VS↑ | HD95 mm↓ |
|---|---:|---:|---:|---:|---:|---:|
'''
for v in ['E01','E02','E04']:
 a=read(O/v/'official.json')['overall'];text+='| '+v+' | '+' | '.join(f'{a[k]:.6f}' for k in metrics)+' |\n'
a=read(O/'E04/official.json')['overall'];text+='\nE04各指標有效類別數（同上順序）：'+', '.join(str(a['count_valid_'+k]) for k in metrics)+'。MR40實際GT位置類別20，case/class presence57，26連通病灶58；不可混用這些分母。\n'
text+='\nE04對E02的配對case bootstrap（2000次、seed20260909、95%百分位區間）：\n\n| 指標 | 差值 | 95% CI |\n|---|---:|---:|\n'
b=comp['E04_vs_E02']
for k in metrics:
 ci=b['paired_case_bootstrap'][k];text+=f"| {k} | {b['delta'][k]:.6f} | [{ci['low']:.6f}, {ci['high']:.6f}] |\n"
text+='''
## 改善來源與殘餘錯誤

使用與官方統計獨立的病灶診斷：每個位置label的26連通元件，以binary Dice Hungarian一對一配對，只接受非零交集；不將位置判對當成配對必要條件。

| 指標 | E02 | E04 |
|---|---:|---:|
'''
for label,k in [('成功配對病灶','matched'),('FP/case','FP_per_case'),('matched binary Dice','matched_binary_Dice_mean'),('matched HD95 mm','matched_HD95_mm_mean'),('配對位置正確','location_correct')]:
 text+=f"| {label} | {diag['versions']['E02'][k]:.6f} | {diag['versions']['E04'][k]:.6f} |\n"
text+='\n候選覆蓋53/58、位置正確36/53；E04並未改善候選漏失、17個FP或17個配對位置錯誤。72候選中0個空mask fallback，因此這次改善確實來自網路分割。尺寸使用體積等效球徑，非最大Feret徑：\n\n| 等效直徑 mm | 配對/GT | E02 matched Dice | E04 matched Dice |\n|---|---:|---:|---:|\n'
for k in ['<=3','(3,5]','(5,7]','>7']:
 a=diag['versions']['E02']['size'][k];b=diag['versions']['E04']['size'][k];text+=f"| {k} | {b['matched']}/{b['total']} | {a['mean_matched_dice']:.6f} | {b['mean_matched_dice']:.6f} |\n"
text+='''
完整52類confusion及病灶逐例資料位於 `artifacts/astra6_e04_crop_segmentation_20260909T031146Z/evaluation/extended_diagnostics.json`。尺寸較大兩組僅10與5顆，解讀有限。

## CT比較與切分限制

資料實際為416例：307MR、109CT。E02與E04來源為156MR+62CT，皆無center2；並非MR-only訓練。CT detector fold2驗證10例中，center4_ct_051的後綴組跨train/val，且全部center4已進入下游fit。因此只保留5例center2 CT作所有模組無fit重疊的探索比較，不能把其他受污染病例算泛化證據。

| 版本 | Precision↑ | Recall↑ | MCC↑ | Dice↑ | VS↑ | HD95 mm↓ |
|---|---:|---:|---:|---:|---:|---:|
'''
for v in ['CT_E02','CT_E04']:
 a=read(O/v/'official.json')['overall'];text+='| '+v+' | '+' | '.join(f'{a[k]:.6f}' for k in metrics)+' |\n'
text+='''
5例極少，CI見current_official_20260909/paired_comparison.json；未用其錯例回填訓練。CT原始GT與舊輸出存儲dtype不一致；為滿足官方SimpleITK型別契約，保存uint8 GT副本及預測，像素label值不變，原檔保存，未修改原資料集。

## 訓練、架構與重現

既有E04已完成218例來源裁切快取（2493crops／4986鏡射rows），直接重用並驗證hash。接手時第14epoch中断，只存第13epoch最佳權重，缺optimizer。原檔封存於recovery_20260909；從13epoch權重續接，明確記錄optimizer重置，非精確resume。修正後每epoch原子保存model、optimizer、CPU/GPU RNG與dataloader RNG。來源174/44 case split，原定max80、min20、patience12；開發於25epoch停止，選13epoch；全218例正式fit完整13epoch、2028optimizer updates。這不是只跑短測試後停止。

模型為既有32³ box-normalized小型3D U-Net，輸入原影像與候選橢球prior，二元BCE+Dice；E02 ExtraTrees使用預測血管943維解剖特徵作52類分類。S使用共同CT/MR來源，按每例百分位正規化。此版仍限制candidate box範圍、每candidate保留最大連通域、沿用paint order，尚非文件提議的較大Residual U-Net與顯式血管channel。先完成已有實驗的理由是來源快取可用、修改可歸因，且現在證據支持形狀改善。GT aneurysm只作source target；推論沒有GT vessel輸入或fallback。

後續優先瓶頸為位置17/53錯誤和17FP，而非直接更換已改善的分割器。既有E03深度位置模型失敗，未因架構偏好替換有效ExtraTrees。F尚缺來源側獨立候選校準，現有source proposals為detector in-sample；本次沒有假稱已訓練/驗證F，也未任意選holdout閾值。這些屬未完成的後續研發範圍。

MR center2反覆用於歷史研發、全為陽性，屬固定開發比較集，不是未接觸blind test。欠缺跨scan patient linkage，case/center隔離不能證明所有病人完全獨立；bootstrap單位只能稱case。保留detector舊308MR lineage（現307，含後來移除病例的歷史訓練）及TA36外部預訓練來源，不宣稱純TopAneu-only模型。

## 交付入口與範圍

`delivery_20260909/run_inference.sh --modality MR|CT --image INPUT --work NEW_DIRECTORY --output OUTPUT.mha` 以原始影像執行detector、TA36、E02位置與E04形狀。可用--gpu指定Task2已分配GPU，--runtime-config覆寫依賴與模型路徑。GT路徑不是推論參數。預設工作區環境，跨機/Docker可攜性仍需驗證。

`MODEL_MANIFEST.json`保存權重與SHA256；configs保存訓練、切分、類別、feature schema、依賴版本；PREDICTION_MANIFEST.json保存MR40+CT5原生MHA的逐檔hash與geometry/type檢查。權重原件、正式checkpoint與歷史模型仍在artifacts。

資源測試結果將更新於RESOURCE_VALIDATION_20260909.json。官方條件為12min/case、T4 16GB、RAM32GB含1GB保留；本機V100結果不能代替T4確認。目前沒有docker/podman executable或daemon socket，亦無T4，因此Docker映像和GC環境驗證未完成，不能將此研究交付宣稱為最終可提交容器。
'''
(R/'REPORT_20260909.md').write_text(text);shutil.copy2(R/'REPORT_20260909.md',P/'reports/EXECUTION_REPORT_20260909.md')
for v in ['E01','E02','E04','CT_E02','CT_E04']:
 shutil.copy2(O/v/'official.json',R/f'reports/{v}_current_official.json')
shutil.copy2(O/'paired_comparison.json',R/'reports/paired_comparison.json');shutil.copy2(E/'evaluation/extended_diagnostics.json',R/'reports/extended_diagnostics.json')
