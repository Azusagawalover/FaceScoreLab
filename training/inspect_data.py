"""Inspect the MEBeauty landmark table and original split lists without changing them."""
import argparse, csv, collections, importlib.util, json, os
from pathlib import Path


def main():
    project=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,
                        default=Path(os.environ.get('FACESCORELAB_MEBEAUTY_DATA', project/'training/data')),
                        help='Directory containing landmarks.csv and train/val/test_2022.txt; raw data is not bundled')
    args=parser.parse_args()
    root=args.data_dir.expanduser().resolve()
    with (root/'landmarks.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    print(json.dumps({'modules':{x:bool(importlib.util.find_spec(x)) for x in ['numpy','sklearn','pandas']},'rows':len(rows),'unique_basename':len(set(Path(r['image']).name for r in rows))},ensure_ascii=False))
    dups={k:v for k,v in collections.Counter(Path(r['image']).name for r in rows).items() if v>1}
    print('duplicates',json.dumps(dups,ensure_ascii=False))
    names={Path(r['image']).name for r in rows}
    for split in ['train','val','test']:
        items=[]
        for line in (root/f'{split}_2022.txt').read_text(encoding='utf-8-sig').splitlines():
            path,score=line.rsplit(' ',1)
            items.append(Path(path).name)
        print(split,len(items),len(set(items)),len(set(items)&names))


if __name__=='__main__':
    main()
