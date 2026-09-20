# TopAneu Task2 官方七項交付：修正版S32／固定D＋C02／CT E16

預設MR為固定D/S32/C02；來源D閾值.955078125，不使用F14、S17fallback或dense分支。可選 --mr-policy dense_fusion 重現修正版S32的完整融合對照。CT保留E16。輸入僅原始影像、模態及包內權重；原生shape/affine、uint8標籤0–52；不讀GT或依GT尺寸分流。

使用：python run_inference.py --image /path/image.nii.gz --modality MR --output /path/prediction.nii.gz --work /path/new_work_dir --gpu 0 --detector-python /path/nndet/python --refinement-python /path/nnunet/python

S32完成新版資料修正微調：development30×150＋final30×150，共9000成功更新，完整optimizer/scheduler/scaler/all RNG保存並CPU載入驗證。模型架構、推論輸入與來源閾值未改。相對r4，新版相同MR40＋CT5官方Dice、HD95、VolSim改善，其餘四項相同；MR40單獨VolSim有微小下降，原值與代價見reports。不能把資料修正控制稱為新架構收益。相對舊r3仍是六項更好、Recall較低。原r3與r4保留。

Dataset switch_inventory_c432a44c65ada602，evaluator660da7ab699446b58748dec4bf6fef6e5efd34ab，metrics03ba48e47790609317916bacfea8904b596cd888。F1由各class計數macro，不反推、不平均七項原始值成總分。本地候選rank不是隱藏測試名次，MR40/CT5反覆開發且上游血統有限制。validation_predictions含完整45病例mask；reports含Task1位置清單與四項。

r4封裝原始MR016與CT190已通過逐體素/hash/affine一致性，V100耗時453.211與432.586秒，峰值PSS約11.92與12.30GiB。這些是r4完整原始計時。r5僅S32權重改變，已在保留的原始D/TA36實際產物上重新執行新精修，5.0秒且native mask/hash bitexact；沒有重算不受影響階段，也沒有把分段執行聲稱為重新實測r5全程計時。詳見BUNDLE_STATUS.json。尚未測T4或使用者容器；31GB是主RAM，非顯存。

9/19中午UTC交付模型與方法以保留容器驗證時間；截止9/21 21:59UTC。720秒、31GB主RAM；sanity每隊每天2次、final每task1次。容器、Contact Form、聯絡身分及正式提交由使用者處理，未自行提交。前五隊排除主辦方可邀論文，需方法、容器、可驗證身分，預計9/27前通知。
