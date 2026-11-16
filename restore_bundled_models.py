"""检查本地运行模型，或从对应 APK 恢复缺失的模型文件。"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def check_local(folder, assets):
    errors = []
    for name, info in assets.items():
        path = folder / name
        if not path.is_file():
            errors.append('Missing model: ' + name)
        elif path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
            errors.append('Model size/hash mismatch: ' + name)
    return errors

def main():
    root=Path(__file__).resolve().parent
    meta=json.loads((root/'source_model_bundle_manifest.json').read_text(encoding='utf-8'))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apk',type=Path,help='用于恢复模型的配套 APK')
    parser.add_argument('--check',action='store_true',help='只校验，不写入文件；默认校验本地模型')
    args=parser.parse_args()
    assets=meta['assets']
    for name in assets:
        if Path(name).name!=name or Path(name).suffix.lower() not in {'.onnx','.task'}:
            raise SystemExit('Invalid model asset name: '+name)
    folder=root/'app/src/main/assets'
    local_errors=check_local(folder, assets)
    if args.apk is None and not local_errors:
        print('Verified '+str(len(assets))+' local model assets.')
        return
    if args.check and args.apk is None:
        raise SystemExit('\n'.join(local_errors))
    apk=args.apk
    if apk is None:
        apk=next((p/meta['apk_filename'] for p in (root,root.parent,root.parent.parent) if (p/meta['apk_filename']).is_file()),None)
    if apk is None:
        parser.error('\n'.join(local_errors)+'\nUse --apk PATH to restore missing models.')
    if not apk.is_file() or apk.stat().st_size!=meta['apk_bytes'] or digest(apk)!=meta['apk_sha256']:
        raise SystemExit('APK hash/size does not match this source release.')
    with zipfile.ZipFile(apk) as archive:
        # 校验全部文件后再恢复，已有的不同模型不覆盖。
        for name,info in assets.items():
            if Path(name).name!=name or Path(name).suffix.lower() not in {'.onnx','.task'}:
                raise SystemExit('Invalid model asset name: '+name)
            entry='assets/'+name
            if entry not in archive.namelist() or archive.getinfo(entry).file_size!=info['bytes']:
                raise SystemExit('Required model asset missing or wrong size: '+name)
            with archive.open(entry) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=info['sha256']:
                    raise SystemExit('Model hash mismatch: '+name)
            target=root/'app/src/main/assets'/name
            if not args.check and target.exists() and digest(target)!=info['sha256']:
                raise SystemExit('Existing different model preserved without overwrite: '+str(target))
        if not args.check:
            for name in assets:
                target=root/'app/src/main/assets'/name
                if target.exists():
                    continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open('assets/'+name) as source,target.open('xb') as destination:
                    shutil.copyfileobj(source,destination)
    print(('Verified ' if args.check else 'Restored/verified ')+str(len(assets))+' model assets from '+apk.name+'.')

if __name__=='__main__':main()
