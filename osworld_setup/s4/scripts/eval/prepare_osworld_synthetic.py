#!/usr/bin/env python3
"""Configure imported synthetic inputs and validate them without model/VM calls."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # Agent-S root
from agent_s4.synthetic_tasks import IDS, ensure_chrome_launch, validate_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/osworld_synthetic_30')
    parser.add_argument('--check', action='store_true', help='Read-only package and split check')
    parser.add_argument('--scheme', choices=['http', 'https'], default='https')
    parser.add_argument('--origin', action='append', default=[], metavar='APP=URL')
    args = parser.parse_args()
    package = args.root.resolve()
    manifest = validate_package(package)
    if not args.check:
        # The package configurator checks only for the existence of this file;
        # link to the public loader, never copy official/gated task definitions.
        loader = package / 'task_loader.py'
        if not loader.exists():
            loader.symlink_to(ROOT / 'OSWorldV2_Base/task_loader.py')
        spec = importlib.util.spec_from_file_location('configure_synthetic', package / 'configure_synthetic.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not os.environ.get('WEBSITE_HOST_SUFFIX'):
            # Read only the public routing key from the main repo .env.
            os.environ['WEBSITE_HOST_SUFFIX'] = module.deployment_suffix(ROOT)
        overrides = {}
        for pair in args.origin:
            app, sep, url = pair.partition('=')
            if not sep or not app or app in overrides:
                parser.error('--origin expects unique APP=URL values')
            overrides[app] = url
        if not module.configure(package, args.scheme, overrides, check_only=True)["configuration_matches"]:
            module.configure(package, args.scheme, overrides)
        repaired = ensure_chrome_launch(package)
        (ROOT / 'eval_sets/synthetic_train_30.json').write_text(json.dumps(IDS, indent=2) + '\n')
        manifest = validate_package(package)
    else:
        repaired = ensure_chrome_launch(package, apply=False)
        if repaired:
            raise ValueError(
                'Web tasks cannot reach Chrome over CDP because they never launch it: '
                + ', '.join(repaired)
                + '. Run this command without --check to repair the package.'
            )
    if not manifest.get('website_configuration', {}).get('configured'):
        raise ValueError('Website configuration missing; run this command without --check')
    train = json.loads((ROOT / 'eval_sets/synthetic_train_30.json').read_text())
    val = json.loads((ROOT / 'eval_sets/osworld_v2_val_15.json').read_text())
    canonical_val = json.loads((ROOT / 'eval_sets/osworld_v2_val_25.json').read_text())
    test = json.loads((ROOT / 'eval_sets/osworld_v2_test.json').read_text())
    if train != IDS or len(val) != 15 or len(set(val)) != 15:
        raise ValueError('Expected exactly 30 synthetic and 15 unique old validation IDs')
    if not set(val) <= set(canonical_val) or set(train) & set(val) or (set(train) | set(val)) & set(test):
        raise ValueError('Invalid validation provenance or train/val/test overlap')
    print(json.dumps({
        'package_root': str(package), 'train_tasks': len(train), 'validation_tasks': len(val),
        'synthetic_score': 'UNGRADED: constant zero, not a success metric',
        'websites': manifest['website_configuration']['origins'],
        'integrity': 'all manifest file hashes and asset references passed',
        'chrome_launch_repaired': repaired,
        'runtime_checks_pending': ['application availability', 'website/session reachability',
                                   'Chrome CDP 9222 reachable from the setup controller',
                                   'UTC setup/browser verification in the target VM', 's_028 fresh tab/cart timer'],
        'model_calls': 0, 'vm_calls': 0,
    }, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f'ERROR: {exc}')
