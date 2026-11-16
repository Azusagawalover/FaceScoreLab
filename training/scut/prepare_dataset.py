"""Audit SCUT mirror labels and create Android-compatible landmark face crops."""
from pathlib import Path
import argparse,csv, hashlib, io, json, math, shutil, time
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT=Path(__file__).resolve().parents[2]
arguments=argparse.ArgumentParser();arguments.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'work')
ROOT=arguments.parse_args().output.resolve()
DATA=ROOT/'dataset'; RAW=DATA/'images'; CROPS=DATA/'crops'
RAW.mkdir(exist_ok=True);CROPS.mkdir(exist_ok=True)
OFFICIAL=Path(__file__).resolve().parent/'official'

def labels(path):
    rename={'fty':'AF','mty':'AM','ftw':'CF','mtw':'CM'}
    result={}
    for line in path.read_text().splitlines():
        if not line.strip():continue
        name,score=line.split();normalized=rename[name[:3]]+name[3:]
        result[normalized]=float(score)
    return result

def face_rect(points,width,height):
    # Match OnnxLocalEngine.faceCrop, including Java float32 arithmetic.
    p=np.asarray(points,dtype=np.float32)
    left,top=p.min(axis=0);right,bottom=p.max(axis=0)
    size=np.float32(max(right-left,bottom-top)*np.float32(1.45))
    cx=np.float32((left+right)/np.float32(2));cy=np.float32((top+bottom)/np.float32(2)-size*np.float32(.06))
    return [max(0,math.floor(float(cx-size/2))),max(0,math.floor(float(cy-size/2))),
            min(width,math.ceil(float(cx+size/2))),min(height,math.ceil(float(cy+size/2)))]

def main():
    train=labels(OFFICIAL/'SCUT-train_1.txt');test=labels(OFFICIAL/'SCUT-test_1.txt')
    assert len(train)==4400 and len(test)==1100 and not train.keys()&test.keys()
    expected=train|test;rows={}
    for path in sorted(DATA.glob('*.parquet')):
        for group in pq.ParquetFile(path).iter_batches(batch_size=64):
            for entry in group.to_pylist():
                name=Path(entry['image_name']).name
                assert name in expected and name not in rows,name
                score=float(entry['beauty_score'])
                assert abs(score-expected[name])<1e-5,(name,score,expected[name])
                image=entry['image']['bytes']
                (RAW/name).write_bytes(image)
                im=Image.open(io.BytesIO(image)).convert('RGB')
                digest=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
                rows[name]={'name':name,'score':expected[name],
                            'official_split':'test' if name in test else 'train',
                            'pixel_sha256':digest,'width':im.width,'height':im.height,
                            'dataset_group':name[:2]}
    assert len(rows)==5500 and set(rows)==set(expected)
    cache_path=DATA/'crop_manifest.json'
    cache=json.loads(cache_path.read_text(encoding='utf8')) if cache_path.exists() else {}
    options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(PROJECT/'app/src/main/assets/face_landmarker.task')),
        running_mode=vision.RunningMode.IMAGE,num_faces=3,min_face_detection_confidence=.5,
        min_face_presence_confidence=.5,min_tracking_confidence=.5)
    start=time.monotonic()
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for index,(name,row) in enumerate(sorted(rows.items())):
            if name in cache and cache[name]['pixel_sha256']==row['pixel_sha256']:
                rows[name]=cache[name];continue
            image=np.asarray(Image.open(RAW/name).convert('RGB'))
            found=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=image))
            if not found.face_landmarks:
                row['detected']=False;row['reason']='MediaPipe face not detected'
            else:
                def coordinates(face):return np.asarray([(p.x*row['width'],p.y*row['height']) for p in face],dtype=np.float32)
                points=max((coordinates(f) for f in found.face_landmarks),key=lambda p:float(np.prod(p.max(0)-p.min(0))))
                rect=face_rect(points,row['width'],row['height'])
                row.update(detected=True,crop=rect,face_count=len(found.face_landmarks))
                (DATA/'landmarks').mkdir(exist_ok=True)
                np.save(DATA/'landmarks'/(name+'.npy'),points)
                Image.fromarray(image).crop(rect).resize((224,224),Image.Resampling.BILINEAR).save(CROPS/(name+'.png'))
            cache[name]=row
            if index%100==0:
                cache_path.write_text(json.dumps(cache,ensure_ascii=False,indent=2),encoding='utf8')
                print('crops',index+1,'/ 5500',round(time.monotonic()-start,1),'s',flush=True)
    cache_path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    # Exact duplicate pixels never cross the train/validation/test boundaries.
    test_hashes={r['pixel_sha256'] for r in rows.values() if r['official_split']=='test'}
    used=set();eligible=[];excluded=[]
    for row in sorted(rows.values(),key=lambda r:r['name']):
        if not row['detected']:
            row['split']='excluded_detection';excluded.append(row);continue
        if row['official_split']=='test':row['split']='test';continue
        digest=row['pixel_sha256']
        if digest in test_hashes or digest in used:
            row['split']='excluded_duplicate';excluded.append(row);continue
        used.add(digest);eligible.append(row)
    rng=np.random.default_rng(42)
    # Stratify by score bins only; demographics are never image-classification targets.
    bins={}
    for row in eligible:bins.setdefault(min(9,int(row['score']*2)),[]).append(row)
    for group in bins.values():
        rng.shuffle(group);validation_count=max(1,round(len(group)*.1))
        for index,row in enumerate(group):row['split']='validation' if index<validation_count else 'train'
    manifest=sorted(rows.values(),key=lambda r:r['name'])
    with (DATA/'split_manifest.csv').open('w',newline='',encoding='utf8') as f:
        keys=['name','score','official_split','split','pixel_sha256','dataset_group']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(manifest)
    counts={s:sum(r['split']==s for r in manifest) for s in sorted({r['split'] for r in manifest})}
    audit={'source':'https://huggingface.co/datasets/MnLgt/scut-fbp5500',
        'official_labels':'https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/tree/master/data/1',
        'all_5500_normalized_names_and_scores_match_official':True,'score_tolerance':1e-5,'counts':counts,
        'filename_normalization':{'fty':'AF','mty':'AM','ftw':'CF','mtw':'CM'},
        'official_train':4400,'official_test':1100,'seed':42,'excluded':excluded,
        'detector_sha256':hashlib.sha256((PROJECT/'app/src/main/assets/face_landmarker.task').read_bytes()).hexdigest(),
        'preprocessing':'Same MediaPipe task and faceCrop formula as Android, RGB, 224x224 bilinear, ImageNet normalization',
        'split_note':'Official fold 1 test is never used for fitting or model selection. Validation is 10% of official training. Exact pixel duplicates removed from training. Different photos of one person cannot be ruled out.'}
    (ROOT/'dataset_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(audit,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
