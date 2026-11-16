"""Read actual device contracts for visible measurements; no prediction accuracy is calculated."""
from pathlib import Path
import argparse
import json

REQUIRED={
    'visible geometry has known values and transformation invariance',
    'missing visible measurements remain unavailable',
    'eye and lip apertures stay visible measurements',
    'real blur changes measured texture without rewriting input',
    'visible measurement JSON carries definitions rather than classifications'
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True,help='Actual cache/android_contract_checks.json pulled from the tested app')
    args=parser.parse_args()
    report=json.loads(args.report.read_text(encoding='utf-8'))
    rows={row['name']:row for row in report['checks']}
    missing=REQUIRED-rows.keys()
    if missing:
        raise SystemExit('Missing real device checks: '+', '.join(sorted(missing)))
    failed=[name for name in REQUIRED if rows[name].get('status')!='PASS']
    if failed:
        raise SystemExit('Required checks did not pass: '+', '.join(sorted(failed)))
    print(json.dumps({'visible_measurement_contracts_passed':len(REQUIRED),'predictive_accuracy_evaluated':False,
                      'scope':'Known synthetic geometry, real bitmap transformations and JSON contracts; no clinical or personal-attribute prediction.'}))

if __name__=='__main__':main()
