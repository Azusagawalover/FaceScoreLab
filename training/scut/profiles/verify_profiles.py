"""Recompute saved report arithmetic and audit model/checkpoint provenance.

No model inference, training, or new test-label parsing is performed here.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import numpy as np
import torch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    work_root=Path(os.environ.get('FACESCORELAB_WORK_DIR') or Path(__file__).resolve().parents[3]/'training/work').expanduser()
    parser.add_argument('--output',default=str(work_root/'score-profiles'),
                        help='Existing profile run directory; defaults under FACESCORELAB_WORK_DIR or project training/work')
    args=parser.parse_args();output=Path(args.output)
    protocol=json.loads((output/'registered_protocol.json').read_text(encoding='utf-8'))
    frozen=json.loads((output/'frozen_candidates.json').read_text(encoding='utf-8'))
    baseline=torch.load(protocol['base_checkpoint'],map_location='cpu',weights_only=False)['model_state']
    assert frozen['registered_protocol_sha256']==sha(output/'registered_protocol.json')
    profiles={}
    for key in ('male','female'):
        checkpoint=output/key/'best.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['protocol_sha256']==sha(output/'registered_protocol.json')
        assert state['test_targets_used_for_training_or_selection'] is False
        features=[name for name in baseline if name.startswith('features.')]
        assert all(torch.equal(baseline[name],state['model_state'][name]) for name in features)
        assert torch.equal(baseline['target_mean'],state['model_state']['target_mean'])
        head=[name for name in baseline if name.startswith('head.')]
        changed=[name for name in head if not torch.equal(baseline[name],state['model_state'][name])]
        assert changed and state['optimizer_state']['state']
        delta=sum(float(((state['model_state'][name]-baseline[name]).double()**2).sum().item()) for name in head)**.5
        metadata=json.loads((output/f'{key}_metadata.json').read_text(encoding='utf-8'))
        assert metadata['sha256']==sha(output/metadata['file'])==frozen['profiles'][key]['sha256']
        assert metadata['checkpoint_sha256']==sha(checkpoint)
        validation=json.loads((output/key/'validation_selection.json').read_text(encoding='utf-8'))
        report=json.loads((output/key/'historical_test_evaluation.json').read_text(encoding='utf-8'))
        with (output/key/'historical_test_predictions.csv').open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        targets=np.array([float(row['target']) for row in rows])
        general=np.array([float(row['general_prediction']) for row in rows])
        trained=np.array([float(row['profile_prediction']) for row in rows])
        recomputed={'general_mae':float(np.abs(targets-general).mean()),
                    'profile_mae':float(np.abs(targets-trained).mean()),
                    'paired_mae_improvement':float((np.abs(targets-general)-np.abs(targets-trained)).mean())}
        assert abs(recomputed['general_mae']-report['general']['mae'])<1e-12
        assert abs(recomputed['profile_mae']-report['specialized']['mae'])<1e-12
        assert abs(recomputed['paired_mae_improvement']-report['paired_mae_improvement'])<1e-12
        assert report['validation_recommendation_frozen']==validation['recommendation']
        assert metadata['validation_recommendation']==validation['recommendation']
        profiles[key]={'passed':True,'backbone_state_tensors_unchanged':len(features),
                       'target_mean_unchanged':True,'head_state_tensors_changed':changed,'head_weight_change_l2':delta,
                       'selected_epoch':state['epoch'],'optimizer_and_rng_present':all(name in state for name in ('optimizer_state','torch_rng','numpy_rng','python_rng')),
                       'test_metrics_recomputed_from_saved_predictions':recomputed,
                       'validation_gate_passed':validation['gate_passed'],
                       'onnx_sha256':metadata['sha256']}
    assert profiles['male']['onnx_sha256']!=profiles['female']['onnx_sha256']
    result={'passed':True,'audit_kind':'No inference; saved output arithmetic and tensor provenance only',
            'source_runner_sha256_matches_registration':sha(Path(__file__).with_name('profile_runner.py'))==protocol['this_runner_sha256'],
            'profiles':profiles}
    assert result['source_runner_sha256_matches_registration']
    (output/'verification_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
