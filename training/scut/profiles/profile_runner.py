"""Two preregistered SCUT scoring profiles, manually selected in the App.

This runner never predicts a person's identity. Dataset filename groups only
define two training subsets. Train/validation select weights; the previously
observed official test fold is opened only after both exported candidates lock.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_STARTED = time.monotonic()
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
DEFAULT_RUN = Path(os.environ.get('FACESCORELAB_WORK_DIR') or PROJECT / 'training/work').expanduser()
APP_PREPROCESS = 'landmark478-crop145-up006-rgb-imagenet-bilinear224-v1'
SEED = 20261115


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def group(name):
    if name[:2] in ('AM', 'CM'):
        return 'male'
    if name[:2] in ('AF', 'CF'):
        return 'female'
    raise ValueError(f'Unknown SCUT filename prefix: {name}')


class Budget:
    def __init__(self):
        self.limit = 60 * 60

    def elapsed(self):
        return time.monotonic() - SCRIPT_STARTED

    def check(self):
        if self.elapsed() >= self.limit:
            raise RuntimeError('The registered 60-minute total wall-clock budget is exhausted.')


def load_base():
    path = ROOT.parent / 'optimization/runner.py'
    spec = importlib.util.spec_from_file_location('frozen_scut_optimization', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def register(args, base, base_path, budget):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'registered_protocol.json'
    if protocol_path.exists():
        raise FileExistsError('Preserve the registered run. Use a new output for another explicitly authorized batch.')
    checkpoint = Path(args.base_run) / 'best.pt'
    base_metrics = json.loads((Path(args.base_run) / 'metrics.json').read_text(encoding='utf-8'))
    base_export = json.loads((Path(args.base_run) / 'export_metadata.json').read_text(encoding='utf-8'))
    warm = base.load_checkpoint(checkpoint)
    dev = base.load_development_rows(args.dev_manifest)
    assert len(dev) == 4400 and warm['development_manifest_sha256'] == digest(args.dev_manifest)
    assert warm['test_evaluated'] is False and base_metrics['test_evaluated'] is False
    assert digest(checkpoint) == base_metrics['best_checkpoint_sha256'] == base_export['checkpoint_sha256']
    assert warm['config']['id'] == 'R4' and warm['config']['seed'] == 2026
    counts = {key: {split: 0 for split in ('train', 'validation', 'test')} for key in ('male', 'female')}
    test_names = []
    # Only name/split are inspected here. Test score text is never converted or used.
    with Path(args.full_manifest).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key, split = group(row['name']), row['split']
            counts[key][split] += 1
            if split == 'test':
                test_names.append({'name': row['name'], 'profile': key, 'split': split})
    assert counts == {'male': {'train':2035, 'validation':190, 'test':525},
                      'female': {'train':1925, 'validation':250, 'test':575}}
    for key in counts:
        write_csv(output / f'{key}_development.csv', [r for r in dev if group(r['name']) == key],
                  ['name', 'score', 'split', 'pixel_sha256'])
    write_csv(output / 'test_names_only.csv', test_names, ['name', 'profile', 'split'])
    protocol = {
        'format':'facescore-manual-score-profiles-v1', 'registered_at_utc':now(),
        'purpose':'Two dataset-group scoring regressors selected manually; no photo identity classifier.',
        'group_definition':{'male':['AM','CM'], 'female':['AF','CF']}, 'counts':counts,
        'base_checkpoint':str(checkpoint), 'base_checkpoint_sha256':digest(checkpoint),
        'base_model_onnx_sha256':base_export['onnx_sha256'],
        'warmstart_provenance':'R4 seed2026 gradients used original train only; original validation selected the checkpoint. Historical test was subsequently observed but never used for gradient updates.',
        'base_metrics_path':str(Path(args.base_run) / 'metrics.json'),
        'base_checkpoint_test_evaluated_flag':False,
        'development_manifest':str(Path(args.dev_manifest).resolve()), 'development_manifest_sha256':digest(args.dev_manifest),
        'full_split_manifest_sha256':digest(args.full_manifest), 'base_runner_sha256':digest(base_path),
        'this_runner_sha256':digest(__file__),
        'configuration':{'seed':SEED, 'backbone':'fully frozen selected R4 backbone including BN statistics',
                         'head':'warmstarted LayerNorm576 / Linear128 / Hardswish / Dropout0.2 / Linear1',
                         'target_mean':'unchanged general-model target_mean buffer',
                         'optimizer':'AdamW', 'learning_rate':1e-4, 'weight_decay':0.01,
                         'batch_size':64, 'maximum_epochs':60, 'early_stopping_patience':10,
                         'minimum_validation_mae_improvement':1e-5, 'loss':'MSE',
                         'augmentation':'none; unchanged prepared224 inputs', 'selection':'lowest group validation raw MAE among actual trained epochs >=1'},
        'validation_deployment_gate':{'minimum_improvement_over_general':0.005,
                                      'test_metrics_do_not_change_recommendation':True},
        'preprocess_version':APP_PREPROCESS, 'original_runner_preprocess_version':base.PREPROCESS_VERSION,
        'normalization':{'mean':[0.485,0.456,0.406], 'std':[0.229,0.224,0.225]},
        'input':{'name':'input','shape':['batch',3,224,224],'layout':'NCHW','dtype':'float32','color':'RGB'},
        'output':{'name':'score','shape':['batch',1],'scale':[1,5]},
        'limits':{'total_wall_minutes':60,'cpu_threads':4,'parallel_training_jobs':1,'training_jobs':2,'paid_compute':0,'download_new_assets':False},
        'test_scope':'Previously observed SCUT official fold1 historical benchmark; not a new blind test and no identity-disjointness established.',
        'test_targets_parsed_before_registration':False,
        'feature_cache_reuse':'Existing R4 prefix cache only after model-prefix, input-content, manifest and cache-byte hashes verify.'
    }
    write_json(protocol_path, protocol)
    print(json.dumps({'phase':'registered','counts':counts,'base_checkpoint_sha256':protocol['base_checkpoint_sha256']}), flush=True)
    budget.check()
    return output, protocol, warm, dev


def extract_development_features(args, output, base, model, dev, budget):
    metrics = json.loads((Path(args.base_run) / 'metrics.json').read_text(encoding='utf-8'))
    cache_path = Path(metrics['cache']['path'])
    cache_meta = json.loads((cache_path.parent / 'cache_manifest.json').read_text(encoding='utf-8'))
    spec = cache_meta['spec']
    assert spec['manifest_sha256'] == digest(args.dev_manifest)
    assert spec['rows_sha256'] == base.digest_json(dev)
    assert spec['early_weights_sha256'] == base.state_sha256(model.features[:9].state_dict())
    assert spec['preprocess'] == 'legacy_facecrop_v1'
    assert cache_meta['npy_sha256'] == digest(cache_path)
    for index, (row, cached) in enumerate(zip(dev, spec['inputs'])):
        budget.check()
        assert row['name'] == cached['name'] and row['split'] == cached['split']
        assert [digest(path) for path in base.image_sources(args.data_dir,row,model._base_config)] == cached['source_sha256']
        if index and index % 1000 == 0:
            print(json.dumps({'phase':'development_input_hash_audit','count':index}), flush=True)
    assert len(spec['inputs']) == len(dev)
    bank=np.load(cache_path,mmap_mode='r',allow_pickle=False)
    assert list(bank.shape)==cache_meta['shape'] and bank.dtype==np.float32
    features=np.empty((len(dev),576),dtype=np.float32)
    with torch.inference_mode():
        tail=model.features[9:].eval()
        for start in range(0,len(dev),64):
            budget.check()
            x=torch.from_numpy(np.array(bank[0,start:start+64],copy=True))
            features[start:start+len(x)]=model.pool(tail(x)).flatten(1).numpy()
    path=output/'development_features.npy'
    np.save(path,features,allow_pickle=False)
    write_json(output/'development_features_metadata.json',{'created_at_utc':now(),'source_cache':str(cache_path),
        'source_cache_sha256':cache_meta['npy_sha256'],'source_cache_view':0,'features_sha256':digest(path),
        'shape':list(features.shape),'dtype':str(features.dtype),'all_actual_input_hashes_verified':True,
        'base_prefix_state_sha256':spec['early_weights_sha256'],'test_images_opened':False})
    print(json.dumps({'phase':'development_features_ready','shape':list(features.shape),'elapsed_seconds':budget.elapsed()}),flush=True)
    return torch.from_numpy(features)


def head_predictions(model, features, indices):
    model.head.eval()
    with torch.inference_mode():
        return (model.head(features[indices]).flatten()+model.target_mean).numpy().astype(np.float64)


def checkpoint_payload(model, optimizer, profile, epoch, validation, protocol_sha, stale, shuffle_rng):
    return {'format':'facescore-manual-score-profile-checkpoint-v1','profile':profile,'epoch':epoch,
            'model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'optimizer_state':copy.deepcopy(optimizer.state_dict()),'validation':validation,
            'target_mean':float(model.target_mean.item()),'protocol_sha256':protocol_sha,
            'stale_epochs':stale,'torch_rng':torch.get_rng_state(),'numpy_rng':shuffle_rng.bit_generator.state,
            'python_rng':random.getstate(),'checkpoint_boundary':'completed epoch',
            'test_targets_used_for_training_or_selection':False}


def train_profile(key, output, protocol, warm, dev, features, base, budget):
    directory=output/key
    directory.mkdir(exist_ok=True)
    model=base.model_from_best(warm)
    for parameter in model.features.parameters():parameter.requires_grad_(False)
    model.features.eval()
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    shuffle=np.random.default_rng(SEED)
    train=np.array([i for i,row in enumerate(dev) if row['split']=='train' and group(row['name'])==key],dtype=np.int64)
    validation=np.array([i for i,row in enumerate(dev) if row['split']=='validation' and group(row['name'])==key],dtype=np.int64)
    targets=torch.tensor([row['score'] for row in dev],dtype=torch.float32)
    baseline=base.metrics(targets[validation].numpy(),head_predictions(model,features,validation))
    write_json(directory/'general_validation_baseline.json',baseline)
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=.01)
    loss_fn=nn.MSELoss()
    best_mae=float('inf');best_epoch=0;stale=0;logs=[]
    protocol_sha=digest(output/'registered_protocol.json')
    for epoch in range(1,61):
        budget.check();model.head.train();order=shuffle.permutation(train);loss_sum=0.0
        for start in range(0,len(order),64):
            budget.check();chosen=order[start:start+64]
            optimizer.zero_grad(set_to_none=True)
            prediction=model.head(features[chosen]).flatten()+model.target_mean
            loss=loss_fn(prediction,targets[chosen]);loss.backward();optimizer.step()
            loss_sum+=float(loss.item())*len(chosen)
        current=base.metrics(targets[validation].numpy(),head_predictions(model,features,validation))
        improved=current['mae'] < best_mae-1e-5
        stale=0 if improved else stale+1
        snapshot=checkpoint_payload(model,optimizer,key,epoch,current,protocol_sha,stale,shuffle)
        base.atomic_torch(directory/'last.pt',snapshot)
        if improved:
            best_mae=current['mae'];best_epoch=epoch
            base.atomic_torch(directory/'best.pt',snapshot)
        logs.append({'epoch':epoch,'train_mse':loss_sum/len(train),'validation':current,'improved':improved,
                     'stale_epochs':stale,'completed_at_utc':now(),'total_wall_seconds':budget.elapsed()})
        write_json(directory/'training_log.json',logs)
        if epoch==1 or improved or stale==10:
            print(json.dumps({'phase':'head_training','profile':key,'epoch':epoch,'validation_mae':current['mae'],
                              'best_mae':best_mae,'stale':stale,'wall_seconds':budget.elapsed()}),flush=True)
        if stale>=10:break
    selected=base.load_checkpoint(directory/'best.pt');model.load_state_dict(selected['model_state']);model.eval()
    predicted=head_predictions(model,features,validation)
    rows=[]
    for index,value in zip(validation,predicted):
        rows.append({'name':dev[int(index)]['name'],'target':dev[int(index)]['score'],'prediction':float(value),'split':'validation'})
    write_csv(directory/'validation_predictions.csv',rows,['name','target','prediction','split'])
    improvement=baseline['mae']-selected['validation']['mae']
    decision={'validation_general':baseline,'validation_profile':selected['validation'],
              'raw_mae_improvement':improvement,'minimum_gate':.005,'gate_passed':improvement>=.005,
              'recommendation':'optional_manual_profile_validation_gate_passed' if improvement>=.005 else 'experimental_profile_gate_not_met_keep_general_default',
              'decision_time_utc':now(),'test_targets_opened':False}
    write_json(directory/'validation_selection.json',decision)
    metrics={'profile':key,'status':'trained','counts':protocol['counts'][key],'best_epoch':best_epoch,
             'actual_epochs':len(logs),'validation':selected['validation'],'general_validation':baseline,
             'validation_recommendation':decision['recommendation'],'best_checkpoint_sha256':digest(directory/'best.pt'),
             'backbone_frozen':True,'base_checkpoint_sha256':protocol['base_checkpoint_sha256'],
             'test_evaluated':False,'test_targets_used_for_selection':False}
    write_json(directory/'metrics.json',metrics)
    return model,metrics,decision


def export_profile(key, model, metrics, decision, output, protocol, dev, args, base, budget):
    budget.check();path=output/f'beauty_{key}.onnx'
    torch.onnx.export(model,torch.zeros(1,3,224,224),str(path),input_names=['input'],output_names=['score'],
                      opset_version=17,dynamic_axes={'input':{0:'batch'},'score':{0:'batch'}},dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    parity=[]
    for row in [r for r in dev if r['split']=='validation' and group(r['name'])==key][:20]:
        budget.check();x=base.input_array(args.data_dir,row,model._base_config if hasattr(model,'_base_config') else protocol['_base_config'])[None]
        with torch.inference_mode():native=float(model(torch.from_numpy(x)).item())
        exported=float(session.run(['score'],{'input':x})[0].item());difference=abs(native-exported)
        assert difference<1e-4,(key,row['name'],difference)
        parity.append({'name':row['name'],'pytorch':native,'onnx':exported,'absolute_difference':difference})
    del session
    write_json(output/key/'export_parity.json',{'rows':parity,'max_absolute_difference':max(r['absolute_difference'] for r in parity),
        'split':'validation','test_targets_opened':False,'onnx_sha256':digest(path)})
    metadata={'file':path.name,'sha256':digest(path),'bytes':path.stat().st_size,
              'model_version':f'scut-mobilenetv3-small-r4-head-{key}-seed{SEED}-v1','profile':key,
              'profile_selection':'manual_by_user_not_inferred_from_photo',
              'counts':protocol['counts'][key],'preprocess_version':APP_PREPROCESS,
              'input':protocol['input'],'output':protocol['output'],'normalization':protocol['normalization'],
              'preprocessing':'Unchanged general R4 faceCrop v1, RGB224, ImageNet normalization; no new alignment or resizing algorithm.',
              'native_scale':[1,5],'display_scale':[1,10],'display_mapping':'1 + 9 * (clip(raw,1,5)-1) / 4',
              'base_checkpoint_sha256':protocol['base_checkpoint_sha256'],'checkpoint_sha256':metrics['best_checkpoint_sha256'],
              'training':'Two-group head-only adaptation of frozen selected R4 backbone; AdamW lr1e-4, batch64, validation raw MAE early stopping.',
              'validation':metrics['validation'],'general_validation':metrics['general_validation'],
              'validation_recommendation':decision['recommendation'],'deployment_decision_made_before_test':True,
              'historical_benchmark_evaluated':False,'new_unseen_test_evaluated':False,
              'limitation':protocol['test_scope']}
    write_json(output/f'{key}_metadata.json',metadata)
    return metadata


def locked_historical_evaluation(args, output, base, protocol, locked, budget):
    budget.check()
    assert digest(args.full_manifest)==protocol['full_split_manifest_sha256']
    for key in ('male','female'):
        assert digest(output/locked['profiles'][key]['file'])==locked['profiles'][key]['sha256']
    # Test targets are first converted below, only after the two candidate hashes and rules lock.
    rows=[]
    with Path(args.full_manifest).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split']=='test':
                rows.append({'name':row['name'],'split':'test','score':float(row['score']),'profile':group(row['name'])})
    assert len(rows)==1100
    options=ort.SessionOptions();options.intra_op_num_threads=4
    general_path=Path(args.base_run)/'beauty_mobilenetv3_small.onnx'
    assert digest(general_path)==protocol['base_model_onnx_sha256']
    sessions={'general':ort.InferenceSession(str(general_path),sess_options=options,providers=['CPUExecutionProvider'])}
    for key in ('male','female'):
        sessions[key]=ort.InferenceSession(str(output/locked['profiles'][key]['file']),sess_options=options,providers=['CPUExecutionProvider'])
    results={}
    for key in ('male','female'):
        chosen=[r for r in rows if r['profile']==key];target=np.array([r['score'] for r in chosen],np.float64)
        predictions={'general':[],'profile':[]}
        for start in range(0,len(chosen),32):
            budget.check();batch=chosen[start:start+32]
            x=np.stack([base.input_array(args.data_dir,row,protocol['_base_config']) for row in batch])
            predictions['general'].extend(sessions['general'].run(['score'],{'input':x})[0].reshape(-1).tolist())
            predictions['profile'].extend(sessions[key].run(['score'],{'input':x})[0].reshape(-1).tolist())
        general=np.array(predictions['general']);special=np.array(predictions['profile'])
        paired=np.abs(target-general)-np.abs(target-special)
        rng=np.random.default_rng(SEED);boot=[]
        for start in range(0,5000,250):
            budget.check();boot.extend(paired[rng.integers(0,len(paired),size=(min(250,5000-start),len(paired)))].mean(axis=1).tolist())
        report={'profile':key,'test_scope':protocol['test_scope'],'backend':'ONNX Runtime CPU, exactly exported models',
                'counts':protocol['counts'][key],'general':base.metrics(target,general),'specialized':base.metrics(target,special),
                'paired_mae_improvement':float(paired.mean()),'paired_image_bootstrap_95ci':np.percentile(boot,[2.5,97.5]).tolist(),
                'bootstrap_resamples':5000,'bootstrap_seed':SEED,'validation_recommendation_frozen':locked['profiles'][key]['validation_recommendation'],
                'test_used_to_select_weights_or_deployment_gate':False,
                'limitation':'Images, not identities, are bootstrap units; fixed-model interval does not cover training random variation or real-phone domain shift.'}
        write_json(output/key/'historical_test_evaluation.json',report)
        write_csv(output/key/'historical_test_predictions.csv',[
            {'name':r['name'],'target':float(y),'general_prediction':float(g),'profile_prediction':float(s),'paired_error_improvement':float(d)}
            for r,y,g,s,d in zip(chosen,target,general,special,paired)],
            ['name','target','general_prediction','profile_prediction','paired_error_improvement'])
        results[key]=report
        metadata=json.loads((output/f'{key}_metadata.json').read_text(encoding='utf-8'))
        metadata.update(historical_benchmark_evaluated=True,evaluation=report)
        write_json(output/f'{key}_metadata.json',metadata)
        print(json.dumps({'phase':'historical_evaluation','profile':key,'general_mae':report['general']['mae'],
                          'profile_mae':report['specialized']['mae'],'paired_mae_improvement':report['paired_mae_improvement']}),flush=True)
    write_json(output/'historical_comparison.json',{'completed_at_utc':now(),'profiles':results,'total_wall_seconds':budget.elapsed(),
               'general_default_remains_available':True,'test_did_not_change_validation_recommendations':True})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(DEFAULT_RUN/'score-profiles'),
                        help='New run directory; defaults under FACESCORELAB_WORK_DIR or project training/work')
    parser.add_argument('--base-run',default=str(PROJECT/'training/scut/optimization/results/r4-seed2026-v2'),
                        help='Selected R4 checkpoint, metrics and export metadata directory')
    parser.add_argument('--dev-manifest',default=str(DEFAULT_RUN/'development_manifest.csv'),
                        help='Frozen 4400-row development manifest from the base run')
    parser.add_argument('--full-manifest',default=str(PROJECT/'training/scut/results/split_manifest.csv'),
                        help='Original full train/validation/test split manifest')
    parser.add_argument('--data-dir',default=str(DEFAULT_RUN/'datasets/scut'),
                        help='Prepared SCUT inputs, including the crops directory; images are not bundled')
    args=parser.parse_args();budget=Budget();base,base_path=load_base();base.setup_threads(4)
    output,protocol,warm,dev=register(args,base,base_path,budget)
    # Runtime-only configuration is not inserted into the immutable preregistration file.
    protocol['_base_config']=warm['config']
    generic=base.model_from_best(warm);generic._base_config=warm['config']
    features=extract_development_features(args,output,base,generic,dev,budget)
    metadata={}
    for key in ('male','female'):
        model,metrics,decision=train_profile(key,output,protocol,warm,dev,features,base,budget)
        model._base_config=warm['config']
        metadata[key]=export_profile(key,model,metrics,decision,output,protocol,dev,args,base,budget)
    locked={'locked_at_utc':now(),'registered_protocol_sha256':digest(output/'registered_protocol.json'),
            'profiles':metadata,'selection_complete_before_test_targets_parsed':True,'test_selection_prohibited':True}
    write_json(output/'frozen_candidates.json',locked)
    print(json.dumps({'phase':'candidates_locked','sha256':digest(output/'frozen_candidates.json'),'test_targets_parsed':False}),flush=True)
    results=locked_historical_evaluation(args,output,base,protocol,locked,budget)
    registry={'format':'manual-score-profile-registry-v1','created_at_utc':now(),
              'profiles':{key:json.loads((output/f'{key}_metadata.json').read_text(encoding='utf-8')) for key in ('male','female')},
              'selection':'manual; this registry does not classify photos','general_model_sha256':protocol['base_model_onnx_sha256']}
    write_json(output/'score_profiles.json',registry)
    write_json(output/'completion.json',{'status':'completed','completed_at_utc':now(),'total_wall_seconds':budget.elapsed(),
               'training_jobs':2,'parallel_jobs':1,'paid_compute':0,'new_downloads':False,'test_evaluated_after_lock':True,
               'profile_files':{key:metadata[key]['file'] for key in metadata}})
    print(json.dumps({'phase':'complete','total_wall_seconds':budget.elapsed(),'models':{key:metadata[key]['sha256'] for key in metadata}}),flush=True)


if __name__=='__main__':
    main()
