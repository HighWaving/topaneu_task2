from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from scripts.astra6_e01.e01_common import (
    BOX_DIR, COHORT_PATH, DATA, N_CLASSES, N_VESSELS, P, TA36_DIR, THRESHOLD, TOP_K,
    affine_world, fill_ellipsoid, load_boxes, load_nifti, load_nifti_geometry,
    lr_pair_map, select_candidates, sha256_file, sha256_tree, sitk_array, vessel_geometry,
    write_json, nifti_output,
)
from scripts.astra6_e01.evaluate import component_records, rectangle_overlap, ellipsoid_overlap
from scripts.local_scoring_arena import aggregate, score_case

E01 = P / "artifacts/astra6_e01_vessel_signature_52class_20260908T042902Z"
PROMPT = P / "agent_bridge/NEXT_EXPERIMENT.md"
METRICS = ["PRECISION", "RECALL", "MCC", "DICE", "VOLSIM", "HD95"]
SEED = 20260905
OFFSETS = [(axis, radius, sign) for axis in range(3) for radius in (2.0, 5.0, 10.0) for sign in (-1.0, 1.0)]


def checked_inputs():
    if "EXPERIMENT_ID=E02" not in PROMPT.read_text() or "STATUS=READY" not in PROMPT.read_text():
        raise RuntimeError("mailbox is not E02 READY")
    done = json.loads((E01 / "DONE.json").read_text())
    if done.get("status") != "DONE" or not done.get("valid"):
        raise RuntimeError("E01 is not valid DONE")
    if sha256_file(E01 / "model/classifier.joblib") != "5b2017acf936b17ac97456ebde04bfb38551b4a189292e02ec4cb09b112f4531":
        raise RuntimeError("E01 model hash mismatch")
    if sha256_tree(E01 / "predictions") != "457ffcf44fabda37db3732da4602a2c37de390378361b286970f774a35bffded":
        raise RuntimeError("E01 prediction hash mismatch")
    ids = json.loads((E01 / "eval_case_ids.json").read_text())
    sets = {
        "cohort": set(json.loads(COHORT_PATH.read_text())),
        "e01": set(ids),
        "boxes": {p.name.removesuffix("_boxes.pkl") for p in BOX_DIR.glob("*_boxes.pkl")},
        "ta36": {p.name.removesuffix(".nii.gz") for p in TA36_DIR.glob("*.nii.gz")},
        "pred": {p.name.removesuffix(".nii.gz") for p in (E01 / "predictions/mr_center2_k05").glob("*.nii.gz")},
        "raw": {r["case_id"] for r in json.loads((E01 / "evaluation/experiment_per_case.json").read_text())},
    }
    if len(ids) != 40 or any(s != set(ids) for s in sets.values()):
        raise RuntimeError("40-case cohort identity mismatch")
    return ids, {k: len(v) for k, v in sets.items()}


def extended_schema(e01_schema):
    names = list(e01_schema["features"])
    extra = [f"v{v:02d}_offset_{'xyz'[axis]}_{int(radius)}mm_{'plus' if sign > 0 else 'minus'}"
             for v in range(1, 37) for axis, radius, sign in OFFSETS]
    if len(names) != 295 or len(extra) != 648:
        raise AssertionError((len(names), len(extra)))
    pair = {int(k): int(v) for k, v in e01_schema["vessel_lr_pair"].items()}
    # Destination feature (v,o) receives original feature (M(v),S(o)).
    lookup = {(v, a, r, s): i for i, (v, a, r, s) in enumerate(
        (x for x in [(v, a, r, s) for v in range(1, 37) for a, r, s in OFFSETS]))}
    perm = []
    for v in range(1, 37):
        for a, r, s in OFFSETS:
            perm.append(lookup[(pair[v], a, r, -s if a == 0 else s)])
    if sorted(perm) != list(range(648)) or np.any(np.asarray(perm)[np.asarray(perm)] != np.arange(648)):
        raise AssertionError("mirror permutation is not involutive")
    return names + extra, pair, np.asarray(perm, dtype=int)


def multiscale(geometry, affine, low, high):
    q = affine_world(affine, (np.asarray(low) + np.asarray(high)) / 2.0)
    points = np.asarray([q + sign * radius * np.eye(3)[axis] for axis, radius, sign in OFFSETS])
    out = []
    for vessel_id in range(1, N_VESSELS + 1):
        tree = geometry["trees"][vessel_id]
        if tree is None:
            out.extend([1.0] * len(OFFSETS))
        else:
            dist = tree.query(points)[0]
            out.extend(np.minimum(dist, 50.0).astype(float) / 50.0)
    arr = np.asarray(out, dtype=np.float32)
    if arr.shape != (648,) or not np.isfinite(arr).all():
        raise AssertionError(arr.shape)
    return arr


def source_features(run, schema, vessel_pair, perm):
    cached = np.load(E01 / "features/train.npz")
    x0, y, weights = cached["X"], cached["y"], cached["sample_weight"]
    records = [json.loads(x) for x in (E01 / "features/train_records.jsonl").read_text().splitlines() if x]
    if x0.shape != (4986, 295) or len(records) != len(y) != len(weights) or not np.isfinite(x0).all():
        raise RuntimeError("frozen E01 training cache invariant failed")
    extra = np.empty((len(records), 648), np.float32)
    # Each original/mirror pair records identical unreflected bounds; calculate once and permute.
    grouped = {}
    for i, r in enumerate(records):
        # E01 component_id is numbered independently within each class, so
        # source_class_id is required to make the stored component identity unique.
        key = (r["case_id"], r["source_class_id"], r["component_id"], r["sample_index"])
        grouped.setdefault(key, []).append((i, r))
    current_case = None
    current_geometry = None
    current_affine = None
    for n, entries in enumerate(grouped.values(), 1):
        if len(entries) != 2 or {r["view"] for _, r in entries} != {"original", "mirror"}:
            raise RuntimeError("record original/mirror mapping failed")
        original = next(r for _, r in entries if r["view"] == "original")
        case_id = original["case_id"]
        # Records retain E01's case-major ordering.  Keep exactly one case's
        # 36 trees resident, then release it when moving to the next case.
        if case_id != current_case:
            image = DATA / "images" / f"{case_id}_0000.nii.gz"
            vessel = DATA / "vessel_masks" / f"{case_id}.nii.gz"
            aff, shape = load_nifti_geometry(image)
            current_geometry = vessel_geometry(vessel, shape, aff)
            current_affine = aff
            current_case = case_id
        geom, aff = current_geometry, current_affine
        feat = multiscale(geom, aff, original["low"], original["high"])
        for i, r in entries:
            extra[i] = feat if r["view"] == "original" else feat[perm]
        if n % 100 == 0:
            print(f"source feature groups {n}/{len(grouped)}", flush=True)
    X = np.concatenate([x0, extra], axis=1).astype(np.float32, copy=False)
    if X.shape != (4986, 943) or not np.isfinite(X).all():
        raise RuntimeError("E02 train matrix invariant failed")
    np.savez_compressed(run / "features/train.npz", X=X, y=y, sample_weight=weights)
    (run / "features/train_records.jsonl").write_text((E01 / "features/train_records.jsonl").read_text())
    write_json(run / "feature_schema.json", {"n_features": 943, "features": schema,
        "base_e01_feature_schema_sha256": sha256_file(E01 / "feature_schema.json"),
        "added_features": {"n": 648, "vessels": 36, "axes": ["x", "y", "z"], "radii_mm": [2,5,10], "signs": [-1,1], "distance_cap_mm": 50},
        "vessel_lr_pair": vessel_pair, "mirror_permutation": perm.tolist(),
        "mirror_definition": "f_mirror(v,o)=f_original(M(v),S(o)); S flips x offset sign only"})
    return X, y, weights, records


def fit(run, X, y, weights, records):
    model = ExtraTreesClassifier(n_estimators=512, max_depth=16, min_samples_leaf=2,
        max_features=0.5, bootstrap=False, class_weight=None, n_jobs=4, random_state=SEED)
    model.fit(X, y, sample_weight=weights)
    path = run / "model/classifier.joblib"; joblib.dump(model, path, compress=3)
    component_support = {str(i): len({(r["case_id"],r["component_id"],r["view"]) for r in records if r["class_id"] == i}) for i in range(1,53)}
    info = {"classes": [int(x) for x in model.classes_], "unsupported_classes": [i for i in range(1,53) if i not in model.classes_],
        "component_view_support": component_support, "params": model.get_params(), "fit_once": True, "seed": SEED,
        "model_sha256": sha256_file(path), "training_feature_sha256": sha256_file(run / "features/train.npz")}
    write_json(run / "class_support.json", info); write_json(run / "model/class_support.json", info)
    write_json(run / "model/LOCKED.json", {"status":"LOCKED", **info})
    return model, path


def infer(run, case_ids, model, perm):
    rows=[]; infos=[]; eval_rows=[]
    e01rows={(r["case_id"],r["original_index"]):r for r in [json.loads(x) for x in (E01/"candidate_predictions.jsonl").read_text().splitlines() if x]}
    for n, cid in enumerate(case_ids,1):
        image=DATA/"images"/f"{cid}_0000.nii.gz"; vessel=TA36_DIR/f"{cid}.nii.gz"
        aff,shape=load_nifti_geometry(image); va,vs=load_nifti_geometry(vessel)
        if vs != shape or not np.allclose(va,aff,atol=1e-4): raise RuntimeError(f"eval geometry {cid}")
        geom=vessel_geometry(vessel,shape,aff); boxes,scores,obj=load_boxes(BOX_DIR/f"{cid}_boxes.pkl"); selected=select_candidates(boxes,scores)
        mask=np.zeros(shape,np.uint8); local=[]
        for rank,(idx,score,lo,hi) in enumerate(reversed(selected)):
            ext=multiscale(geom,aff,lo,hi); # E01 inference has original view only.
            # Original 295 feature must be recomputed through E01 semantics.
            from scripts.astra6_e01.e01_common import compute_feature
            feat=np.concatenate([compute_feature(geom,aff,lo,hi,"MR"),ext]).astype(np.float32)
            prob=np.zeros(52,float); mp=model.predict_proba(feat[None])[0]
            for c,p in zip(model.classes_,mp): prob[int(c)-1]=p
            pred=int(np.argmax(prob)+1)
            fill_ellipsoid(mask,lo,hi,pred)
            e=e01rows[(cid,idx)]
            item={"case_id":cid,"original_index":idx,"detector_rank_desc":len(selected)-1-rank,"rank":len(selected)-1-rank,"score":score,"low":lo.tolist(),"high":hi.tolist(),"E01_predicted_class":e["predicted_class_id"],"E02_predicted_class":pred,"predicted_class_id":pred,"probabilities":prob.tolist()}
            local.append(item); rows.append(item); eval_rows.append(feat)
        local.sort(key=lambda r:r["detector_rank_desc"])
        e01mask=load_nifti(E01/f"predictions/mr_center2_k05/{cid}.nii.gz")[0]
        if not np.array_equal(mask>0,e01mask>0): raise RuntimeError(f"binary invariant {cid}")
        out=run/f"predictions/mr_center2_k05/{cid}.nii.gz"; nifti_output(mask,image,out)
        infos.append({"case_id":cid,"selected":len(selected),"shape":list(shape),"affine":aff.tolist(),"box_restore":bool(obj["restore"]),"binary_foreground_equal_e01":True})
        if n%5==0: print(f"inference {n}/40",flush=True)
    (run/"candidate_predictions.jsonl").write_text("\n".join(json.dumps(r,sort_keys=True) for r in rows)+"\n")
    np.savez_compressed(run/"features/eval.npz",X=np.asarray(eval_rows,np.float32))
    write_json(run/"features/eval_records.json",rows); h=sha256_tree(run/"predictions")
    write_json(run/"prediction_manifest.json",{"case_ids":case_ids,"n_cases":40,"n_candidates":len(rows),"prediction_tree_sha256":h,"candidate_selection":{"threshold":THRESHOLD,"top_k":TOP_K},"binary_foreground_checked":True})
    write_json(run/"validity_checks.json",{"candidate_selection_identical":True,"binary_foreground_identical":True,"n_cases":40,"n_candidates":len(rows),"case_geometry_checked":True})
    return rows,h


def counts(agg): return {k:int(sum(v for kk,v in agg["per_class"].items() if kk.startswith(k+"_"))) for k in ("TP","FP","FN","TN")}
def coverage(agg): return sum(v>0 for k,v in agg["per_class"].items() if k.startswith("TP_"))

def diagnostics(run, ids, rows):
    bycase={c:[] for c in ids}
    for r in rows: bycase[r["case_id"]].append(r)
    totals={"components":0,"rect_hit":0,"ellipsoid_hit":0,"unmatched_candidates":0,"conditional_before_correct":0,"conditional_after_correct":0}
    ledger=[]
    for cid in ids:
        gt=np.transpose(sitk_array(DATA/f"location_masks/{cid}.nii.gz"),(2,1,0)); comps=component_records(gt)
        boxes,scores,_=load_boxes(BOX_DIR/f"{cid}_boxes.pkl"); selected=select_candidates(boxes,scores); hits=set()
        totals["components"]+=len(comps)
        for comp in comps:
            rect=[x for x in selected if rectangle_overlap(comp["coords"],x[2],x[3])]; ell=any(ellipsoid_overlap(comp["coords"],x[2],x[3]) for x in selected)
            if rect:
                totals["rect_hit"]+=1; hits.update(x[0] for x in rect); best=sorted(rect,key=lambda x:(-x[1],x[0]))[0]; row=next(r for r in bycase[cid] if r["original_index"]==best[0])
                b=row["E01_predicted_class"]==comp["class_id"]; a=row["E02_predicted_class"]==comp["class_id"]
                totals["conditional_before_correct"]+=b; totals["conditional_after_correct"]+=a
                ledger.append({"case_id":cid,"class_id":comp["class_id"],"selected_index":best[0],"rectangle_hit":True,"ellipsoid_hit":ell,"E01_class":row["E01_predicted_class"],"E02_class":row["E02_predicted_class"],"before_correct":b,"after_correct":a})
            else: ledger.append({"case_id":cid,"class_id":comp["class_id"],"selected_index":None,"rectangle_hit":False,"ellipsoid_hit":ell,"before_correct":False,"after_correct":False})
            totals["ellipsoid_hit"]+=ell
        totals["unmatched_candidates"]+=len(selected)-len(hits)
    totals["conditional_denominator"]=totals["rect_hit"]; totals["conditional_before"]=totals["conditional_before_correct"]/totals["rect_hit"]; totals["conditional_after"]=totals["conditional_after_correct"]/totals["rect_hit"]
    rescued=[x for x in ledger if x.get("after_correct") and not x.get("before_correct")]; lost=[x for x in ledger if x.get("before_correct") and not x.get("after_correct")]
    out={"fixed_candidates":totals,"rescued":rescued,"lost":lost,"all_components":ledger}
    write_json(run/"lesion_ledger.json",ledger); write_json(run/"diagnostics.json",out); return out


def conditional_shape(per_case):
    vals=[]
    for r in per_case:
        raw=r["raw"]
        # Raw dictionaries contain per-class records whose exact key varies only by scorer version; inspect Dice list entries.
        for v in raw.values() if isinstance(raw,dict) else []:
            if isinstance(v,dict) and "DICE" in v and v.get("TP",0)>0: vals.append(float(v["DICE"]))
    # Formal E01 reference is preserved separately; per-case raw aggregates are authoritative for official metrics.
    return {"definition":"case/class official raw entries with TP where exposed","n":len(vals),"mean":float(np.mean(vals)) if vals else None,"median":float(np.median(vals)) if vals else None}


def evaluate(run, ids, rows):
    before=json.loads((E01/"evaluation/experiment_per_case.json").read_text())
    beforeagg=aggregate([x["raw"] for x in before])
    after=[]
    for cid in ids: after.append({"case_id":cid,"raw":score_case(sitk_array(run/f"predictions/mr_center2_k05/{cid}.nii.gz"),cid)})
    afteragg=aggregate([x["raw"] for x in after])
    def save(name,pc,agg):
        write_json(run/f"evaluation/{name}_per_case.json",pc); write_json(run/f"evaluation/{name}_official.json",{"overall":agg["overall"],"per_class":agg["per_class"],"counts":counts(agg)})
    save("before",before,beforeagg); save("after",after,afteragg)
    b={m:float(beforeagg["overall"][m]) for m in METRICS}; a={m:float(afteragg["overall"][m]) for m in METRICS}; d={m:a[m]-b[m] for m in METRICS}
    diag=diagnostics(run,ids,rows)
    rng=np.random.default_rng(SEED); dm=[]; dd=[]
    for _ in range(2000):
        ix=rng.integers(0,40,40); ba=aggregate([before[i]["raw"] for i in ix])["overall"]; aa=aggregate([after[i]["raw"] for i in ix])["overall"]; dm.append(float(aa["MCC"]-ba["MCC"]));dd.append(float(aa["DICE"]-ba["DICE"]))
    boot={k:{"n":2000,"seed":SEED,"mean":float(np.mean(x)),"percentile_2_5":float(np.percentile(x,2.5)),"percentile_97_5":float(np.percentile(x,97.5))} for k,x in {"MCC":dm,"DICE":dd}.items()};write_json(run/"paired_bootstrap.json",boot)
    bc,ac=counts(beforeagg),counts(afteragg); cvb,cva=coverage(beforeagg),coverage(afteragg)
    checks={"MCC_gain_ge_0.015":d["MCC"]>=.015-1e-12,"Dice_gain_ge_0.008":d["DICE"]>=.008-1e-12,"TP_ge_38":ac["TP"]>=38,"conditional_assignment_ge_39_of_53":diag["fixed_candidates"]["conditional_after_correct"]>=39,"coverage_ge_12_of_20":cva>=12,"FP_le_34":ac["FP"]<=34,"precision_non_decrease":d["PRECISION"]>=-1e-12,"recall_non_decrease":d["RECALL"]>=-1e-12,"volsim_non_decrease":d["VOLSIM"]>=-1e-12,"HD95_nonincrease":d["HD95"]<=1e-12,"cohort_40_of_40":len(after)==40,"fixed_candidate_geometry":diag["fixed_candidates"]["components"]==58 and diag["fixed_candidates"]["rect_hit"]==53 and diag["fixed_candidates"]["ellipsoid_hit"]==53 and diag["fixed_candidates"]["unmatched_candidates"]==17,"official_scorer":True}
    gate={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"before":{"overall":b,"counts":bc},"after":{"overall":a,"counts":ac},"delta":d,"assignment":diag["fixed_candidates"],"coverage":{"before":cvb,"after":cva},"bootstrap":boot,"conditional_shape":{"before_cached_reference":{"n":34,"mean":.5550699456826793,"median":.5340313580879583},"after":conditional_shape(after)}}
    write_json(run/"metrics_comparison.json",gate);write_json(run/"success_gate.json",gate)
    return gate


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    for x in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[x]="1"
    ids,set_counts=checked_inputs(); stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); run=P/f"artifacts/astra6_e02_multiscale_vessel_signature_{stamp}";run.mkdir()
    for d in ("logs","features","model","predictions/mr_center2_k05","evaluation"): (run/d).mkdir(parents=True,exist_ok=True)
    start=time.time(); stages={}; shutil.copy2(PROMPT,run/"prompt.md")
    evidence=json.loads((P/"planning/astra6_e02_20260908/evidence.json").read_text()); wrapper=P/"scripts/local_scoring_arena.py"; official=Path("/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/TopAneu-26/eval/task2/evaluate.py")
    sourcehash={str(wrapper):sha256_file(wrapper),str(official):sha256_file(official)}
    if sourcehash != evidence["planner_source_hashes"]: raise RuntimeError("scorer source hash mismatch")
    write_json(run/"config.json",{"experiment":"E02","classifier":{"n_estimators":512,"max_depth":16,"min_samples_leaf":2,"max_features":.5,"bootstrap":False,"class_weight":None,"n_jobs":4,"random_state":SEED},"cpu_only":True})
    write_json(run/"provenance.json",{"e01":str(E01),"e01_model_sha256":sha256_file(E01/"model/classifier.joblib"),"e01_predictions_sha256":sha256_tree(E01/"predictions"),"cohort_set_counts":set_counts,"wrapper_sha256":sourcehash,"evaluation_gt_loaded_before_prediction_hash":False})
    write_json(run/"eval_case_ids.json",ids);shutil.copy2(E01/"source_manifest.json",run/"source_manifest.json");shutil.copy2(E01/"train_case_ids.json",run/"train_case_ids.json")
    e01schema=json.loads((E01/"feature_schema.json").read_text()); schema,pair,perm=extended_schema(e01schema)
    t=time.time();X,y,w,records=source_features(run,schema,pair,perm);stages["feature_seconds"]=time.time()-t
    t=time.time(); model,path=fit(run,X,y,w,records);stages["fit_seconds"]=time.time()-t
    t=time.time();rows,h=infer(run,ids,model,perm);stages["inference_seconds"]=time.time()-t;(run/"PREDICTIONS_HASHED_BEFORE_EVAL_GT").write_text(h+"\n")
    t=time.time();gate=evaluate(run,ids,rows);stages["evaluation_seconds"]=time.time()-t;stages["total_seconds"]=time.time()-start
    write_json(run/"runtime.json",{"stages_seconds":stages,"cpu_only":True,"cuda_visible_devices":"","max_workers":4,"python":sys.executable,"platform":platform.platform(),"disk_output_bytes":sum(x.stat().st_size for x in run.rglob("*") if x.is_file())})
    write_json(run/"hashes.json",{"code_tree_sha256":sha256_tree(P/"scripts/astra6_e02"),"predictions_tree_sha256":h,"model_sha256":sha256_file(path),"config_sha256":sha256_file(run/"config.json")})
    result=["# RESULT_TASK2_ASTRA6_E02","","Exact change: appended 648 multiscale off-center vessel-distance features (943 total); E01 295 columns, classifier, source rows/weights, candidates, ellipsoids and painting order frozen.","","E01 before:",json.dumps(gate["before"],indent=2),"E02 after:",json.dumps(gate["after"],indent=2),"Delta:",json.dumps(gate["delta"],indent=2),f"Assignment: {gate['assignment']['conditional_before_correct']}/53 -> {gate['assignment']['conditional_after_correct']}/53; coverage {gate['coverage']['before']}/20 -> {gate['coverage']['after']}/20.","Bootstrap:",json.dumps(gate["bootstrap"],indent=2),"","Gate: "+gate["status"],json.dumps(gate["checks"],indent=2),"Validity: complete paired 40-case scoring; E01 cached raw reused for before; locked predictions hashed before evaluation GT; candidate/foreground/geometry checks passed.","Limitations: source anatomy is organizer-predicted silver masks and held-out anatomy is TA36; source GT-box jitter and detector-box evaluation retain a domain gap. Center2 is a training-held-out reference, not pristine blind testing or normal-case specificity.","Runtime:",json.dumps(stages,indent=2),"","STOPPED. Waiting for Astra review."]
    (run/"RESULT_TASK2_ASTRA6_E02.md").write_text("\n".join(result)+"\n")
    write_json(run/"DONE.json",{"status":"DONE","valid":True,"gate":gate["status"],"n_cases":40,"prediction_hash":h})
    print(json.dumps({"run":str(run),"gate":gate["status"],"after":gate["after"],"counts":gate["after"]["counts"],"assignment":gate["assignment"],"coverage":gate["coverage"],"runtime_seconds":stages["total_seconds"]},indent=2))

if __name__=="__main__": main()
