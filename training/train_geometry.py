"""Fit an auditable 1-10 geometric regression from the original MEBeauty 68-point table.

No face identity, health, or other sensitive-attribute classifier is trained.
The deployed MediaPipe topology uses corresponding semantic points; this is
a cross-detector approximation, not a claimed end-to-end accuracy result.
"""
from pathlib import Path
import csv, json, hashlib, math, collections, argparse
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=PROJECT/'training/data')
parser.add_argument('--out',type=Path,default=PROJECT/'training/results')
parser.add_argument('--assets',type=Path,default=PROJECT/'app/src/main/assets')
args=parser.parse_args()
DATA=args.data
OUT=args.out
OUT.mkdir(parents=True,exist_ok=True)
ASSETS=args.assets
ASSETS.mkdir(parents=True,exist_ok=True)
FEATURES=[
    ('face_aspect','面部高宽比'),('inner_eye_ratio','眼间距比例'),
    ('eye_width_ratio','眼宽比例'),('eye_open_ratio','眼睛开合比例'),
    ('brow_eye_ratio','眉眼间距比例'),('nose_width_ratio','鼻宽比例'),
    ('nose_length_ratio','鼻部投影长度比例'),('mouth_width_ratio','嘴宽比例'),
    ('lip_fullness_ratio','外唇高度比例'),('mouth_open_ratio','嘴部开合比例'),
    ('lower_face_ratio','下半面部比例'),('symmetry_error','左右差异比例'),
    ('jaw_width_ratio','下颌宽度比例'),('jaw_angle','下颌夹角比例'),
    ('nose_mouth_ratio','鼻宽与嘴宽比')]

def features68(points):
    p=np.asarray(points,dtype=float).reshape(68,2)
    le=p[36:42].mean(axis=0); re=p[42:48].mean(axis=0)
    axis=re-le; length=np.linalg.norm(axis)
    if length<=1e-7: raise ValueError('eye centers coincide')
    u=axis/length; v=np.array([-u[1],u[0]])
    if np.dot(p[8]-(le+re)/2,v)<0: v=-v
    q=np.stack(((p-(le+re)/2)@u,(p-(le+re)/2)@v),axis=1)
    width=abs(q[16,0]-q[0,0])
    brow=q[17:27].mean(axis=0)
    height=q[8,1]-brow[1]
    if width<=1 or height<=1: raise ValueError('invalid face extent')
    q/=width; height/=width
    d=lambda a,b:float(np.linalg.norm(q[a]-q[b]))
    ew=(d(36,39)+d(42,45))/2
    mw=d(48,54)
    if ew<=1e-7 or mw<=1e-7: raise ValueError('invalid component width')
    ear=((d(37,41)+d(38,40))/(2*d(36,39))+(d(43,47)+d(44,46))/(2*d(42,45)))/2
    eye_centers=(q[36:42].mean(axis=0),q[42:48].mean(axis=0))
    browdist=(np.linalg.norm(q[17:22].mean(axis=0)-eye_centers[0])+np.linalg.norm(q[22:27].mean(axis=0)-eye_centers[1]))/2
    pairs=[(0,16),(1,15),(2,14),(3,13),(4,12),(5,11),(6,10),(7,9),
           (17,26),(18,25),(19,24),(20,23),(21,22),(31,35),(32,34),
           (36,45),(37,44),(38,43),(39,42),(40,47),(41,46),
           (48,54),(49,53),(50,52),(55,59),(56,58),(60,64),(61,63),(65,67)]
    sym=float(np.mean([math.hypot(q[a,0]+q[b,0],q[a,1]-q[b,1]) for a,b in pairs]))
    va=q[4]-q[8]; vb=q[12]-q[8]
    angle=math.acos(float(np.clip(np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb)), -1,1)))/math.pi
    out=[height,d(39,42),ew,float(ear),float(browdist),d(31,35),d(27,33)/height,
         mw,d(51,57)/mw,d(62,66)/mw,(q[8,1]-q[33,1])/height,sym,d(4,12),angle,d(31,35)/mw]
    if not np.isfinite(out).all(): raise ValueError('nonfinite geometry')
    return np.array(out)

raw=list(csv.DictReader((DATA/'landmarks.csv').open(encoding='utf-8-sig',newline='')))
membership=collections.defaultdict(set)
for split in ['train','val','test']:
    for line in (DATA/f'{split}_2022.txt').read_text(encoding='utf-8-sig').splitlines():
        path,score=line.rsplit(' ',1)
        membership[Path(path).name].add(split)
groups=collections.defaultdict(list)
for row in raw: groups[Path(row['image']).name].append(row)
prepared=[]; skipped=[]; duplicate_groups=[]
for name,rows in groups.items():
    split=next((s for s in ['test','val','train'] if s in membership[name]),None)
    if split is None:
        skipped.append({'name':name,'reason':'not present in official split files'})
        continue
    if len(rows)>1: duplicate_groups.append({'name':name,'rows':len(rows),'memberships':sorted(membership[name]),'kept_split':split})
    try:
        coords=[float(x) for x in rows[0]['landmarks'].strip(',').split(',')]
        x=features68(coords)
    except Exception as e:
        skipped.append({'name':name,'reason':str(e)})
        continue
    scores=[float(r['score']) for r in rows]
    gender='female' if '/female/' in rows[0]['image'] else 'male' if '/male/' in rows[0]['image'] else 'unknown'
    prepared.append({'name':name,'split':split,'gender':gender,'score':float(np.mean(scores)),
                     'features':x.tolist(),'path':rows[0]['image'],'points':coords})

part={s:[r for r in prepared if r['split']==s] for s in ['train','val','test']}
X={s:np.array([r['features'] for r in records]) for s,records in part.items()}
Y={s:np.array([r['score'] for r in records]) for s,records in part.items()}
lower=np.quantile(X['train'],.01,axis=0); upper=np.quantile(X['train'],.99,axis=0)
clipped={s:np.clip(arr,lower,upper) for s,arr in X.items()}
mean=clipped['train'].mean(axis=0); std=clipped['train'].std(axis=0)
std=np.maximum(std,1e-8)
Z={s:(arr-mean)/std for s,arr in clipped.items()}
intercept=float(Y['train'].mean())

def metrics(y,p):
    return {'n':len(y),'mae':float(np.mean(np.abs(y-p))),'rmse':float(np.sqrt(np.mean((y-p)**2))),
            'pearson':float(np.corrcoef(y,p)[0,1]) if np.std(p)>1e-12 else None}

candidates=[]
for alpha in [0.,1.,3.,10.,30.,100.,300.,1000.]:
    coef=np.linalg.solve(Z['train'].T@Z['train']+alpha*np.eye(len(FEATURES)),Z['train'].T@(Y['train']-intercept))
    pred=np.clip(intercept+Z['val']@coef,1,10)
    candidates.append((metrics(Y['val'],pred)['mae'],alpha,coef))
_,alpha,coef=min(candidates,key=lambda item:item[0])
predictions={s:np.clip(intercept+Z[s]@coef,1,10) for s in part}
metric={s:metrics(Y[s],predictions[s]) for s in part}
baseline={s:metrics(Y[s],np.full(len(Y[s]),intercept)) for s in part}
group_test_metrics={}
for gender in ['male','female']:
    mask=np.array([r['gender']==gender for r in part['test']])
    group_test_metrics[gender]={'model':metrics(Y['test'][mask],predictions['test'][mask]),
                                'constant_baseline':metrics(Y['test'][mask],np.full(int(mask.sum()),intercept))}

references={}
for mode in ['general','male','female']:
    mask=np.ones(len(part['train']),dtype=bool) if mode=='general' else np.array([r['gender']==mode for r in part['train']])
    references[mode]={'n':int(mask.sum()),'score_mean':float(Y['train'][mask].mean()),
                      'feature_mean':clipped['train'][mask].mean(axis=0).tolist(),
                      'feature_std':np.maximum(clipped['train'][mask].std(axis=0),1e-8).tolist()}

model={'version':'mebeauty68-ridge-v1','score_min':1,'score_max':10,'date':'2026-11-05',
       'dataset':'MEBeauty original public release, landmarks.csv',
       'training_detector':'dlib-style 68-point annotations supplied by MEBeauty',
       'runtime_detector':'MediaPipe Face Landmarker 478, corresponding semantic points mapped to 68',
       'limitation':'Held-out metrics evaluate supplied 68-point annotations only. The cross-detector mapping has not been calibrated end-to-end on a representative phone image test set.',
       'features':[n for n,_ in FEATURES],'feature_labels':[n for _,n in FEATURES],
       'lower':lower.tolist(),'upper':upper.tolist(),'mean':mean.tolist(),'std':std.tolist(),
       'coefficients':coef.tolist(),'intercept':intercept,'alpha':alpha,'references':references,
       'metrics':metric,'baseline_metrics':baseline,'group_test_metrics':group_test_metrics,'split_counts':{s:len(v) for s,v in part.items()},
       'source_sha256':hashlib.sha256((DATA/'landmarks.csv').read_bytes()).hexdigest()}
(ASSETS/'score_model.json').write_text(json.dumps(model,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(OUT/'score_model.json').write_text(json.dumps(model,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
audit={'input_rows':len(raw),'unique_basenames':len(groups),'used_rows':len(prepared),'duplicates':duplicate_groups,
       'skipped':skipped,'split_policy':'Group by image basename; official test > val > train. Keep first landmark geometry and average repeated score labels inside the group. No basename crosses sets.',
       'normalization':'1st/99th percentile clipping and mean/std fitted on train only; alpha chosen using validation MAE; test evaluated once after selection.',
       'split_counts':model['split_counts'],'selected_alpha':alpha,'metrics':metric,'baseline_metrics':baseline,'group_test_metrics':group_test_metrics,
       'candidate_validation_mae':[{'alpha':a,'mae':m} for m,a,c in candidates]}
(OUT/'training_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
with (OUT/'held_out_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['filename','reference_group','human_mean_score','model_prediction','constant_baseline'])
    for r,p in zip(part['test'],predictions['test']):writer.writerow([r['name'],r['gender'],r['score'],float(p),intercept])
with (OUT/'split_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['filename','split','source_reference_group'])
    for r in prepared:writer.writerow([r['name'],r['split'],r['gender']])

text=f'''MEBeauty几何评分实验
数据来源：MEBeauty原始公开发布；完整68点表{len(raw)}行，按图片文件名保守去重后{len(groups)}张。
有效划分：训练{len(part['train'])}张、验证{len(part['val'])}张、独立测试{len(part['test'])}张。
重复文件按组隔离；同一文件名不会同时出现在训练、验证和测试中。
从68点提取{len(FEATURES)}个几何比例；标准化及截尾参数仅由训练集拟合。
模型：岭回归，验证集选择alpha={alpha:g}；评分范围1—10。
独立测试MAE：{metric['test']['mae']:.3f}分；RMSE：{metric['test']['rmse']:.3f}分；Pearson相关系数：{metric['test']['pearson']:.3f}。
训练集平均分常数基线MAE：{baseline['test']['mae']:.3f}分，RMSE：{baseline['test']['rmse']:.3f}分。
通用参考样本{references['general']['n']}张；男性参考{references['male']['n']}张；女性参考{references['female']['n']}张。分组来自数据标注，由使用者主动选择，不从照片推断。
测试子组：男性{group_test_metrics['male']['model']['n']}张，MAE {group_test_metrics['male']['model']['mae']:.3f}；女性{group_test_metrics['female']['model']['n']}张，MAE {group_test_metrics['female']['model']['mae']:.3f}。子组样本量和图像来源不同，不能据此判断人的属性或价值。
以上误差仅针对原始68点标注，不能视为手机端照片评分准确率。
手机端采用MediaPipe478点的语义对应点，存在跨检测器偏差；还需代表性真机照片进行端到端校准。
分数学习的是该数据集评分者的主观意见，不能代表客观美丑或个人价值。
'''
(ASSETS/'experiment_summary.txt').write_text(text,encoding='utf-8')
(OUT/'experiment_summary.txt').write_text(text,encoding='utf-8')
# Representative unmodified landmarks for the Java/Python feature parity check.
fixtures=[{'name':r['name'],'points':r['points'],'features':r['features']} for r in part['test'][:4]]
(OUT/'geometry_parity_fixtures.json').write_text(json.dumps(fixtures,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'split_counts':model['split_counts'],'alpha':alpha,'test':metric['test'],'baseline':baseline['test'],'skipped':len(skipped)},ensure_ascii=False))
