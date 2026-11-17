#!/usr/bin/env python3
"""Compare a baseline runner with current HSA on identical LongBench prompts."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_suite.datasets import load_evaluation_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--reuse-baseline', action='store_true', help='Reuse the saved baseline only if settings and inputs match')
    parser.add_argument('--task', action='append', help='Measure only this configured task; repeat to select several')
    parser.add_argument('--runner', default='./runner')
    parser.add_argument('--repeats', type=int, default=3, help='Includes one warmup per task')
    parser.add_argument('--output', default='results/hsa-parity-longbench.json')
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('--repeats must include a warmup and at least one measured run')
    cfg = json.loads((ROOT/'configs/default.json').read_text())
    tasks = json.loads((ROOT/'configs/longbench.json').read_text())['benchmark']['tasks']
    output = ROOT/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    batch = []
    for task in tasks:
        sample = load_evaluation_samples('longbench', task, cfg['benchmark']['context_len'], 1)[0]
        for repeat in range(args.repeats):
            batch.append({'sample_id': len(batch)+1, 'prompt': sample['prompt'], 'max_tokens': cfg['model']['max_tokens']})
    batch_path = output.with_suffix('.batch.json')
    if args.reuse_baseline and (not batch_path.exists() or json.loads(batch_path.read_text()) != batch):
        parser.error('Saved baseline batch does not match the requested inputs')
    batch_path.write_text(json.dumps(batch))
    result = {'settings': cfg, 'workload': 'longbench', 'tasks': tasks, 'repeats': args.repeats,
              'ignore_eos': True, 'hsa_enabled': True, 'baseline_binary_sha256': hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),
              'run_order': 'task_pairs_alternating',
              'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'runs': {}}
    if args.reuse_baseline:
        saved = json.loads(output.read_text())
        if any(saved.get(key) != result[key] for key in ('settings', 'tasks', 'repeats', 'baseline_commit', 'ignore_eos', 'hsa_enabled', 'baseline_binary_sha256', 'run_order')):
            parser.error('Saved baseline settings or commit do not match')
    measured_tasks = args.task or tasks
    if any(task not in tasks for task in measured_tasks):
        parser.error('--task must name a configured LongBench task')
    result['measured_tasks'] = measured_tasks
    runners = {'baseline': args.baseline, 'current': args.runner}
    result['runs'] = {'baseline': [], 'current': []}
    if args.reuse_baseline:
        result['runs']['baseline'] = json.loads(output.with_suffix('.baseline.json').read_text())
        print('Reusing matching saved baseline', flush=True)
    for task_index, task in enumerate(tasks):
        if task not in measured_tasks:
            continue
        ids = set(range(task_index*args.repeats+1, (task_index+1)*args.repeats+1))
        task_batch = output.with_suffix(f'.task-{task_index}.batch.json')
        task_batch.write_text(json.dumps([sample for sample in batch if sample['sample_id'] in ids]))
        labels = ('baseline', 'current') if task_index%2 == 0 else ('current', 'baseline')
        for label in labels:
            if label == 'baseline' and args.reuse_baseline:
                rows = [row for row in result['runs'][label] if row['sample_id'] in ids]
            else:
                runner = runners[label]
                task_output = output.with_suffix(f'.task-{task_index}.{label}.json')
                cmd = [str(Path(runner).resolve()), '--model', cfg['model']['path'], '--rocket', '1', '--hsa', '1', '--kv-type', 'f16', '--ignore-eos',
                       '--budget', str(cfg['attention']['budget']), '--n-ctx', str(cfg['model']['n_ctx']),
                       '--context-len', str(cfg['benchmark']['context_len']), '--window-size', str(cfg['attention']['window_size']),
                       '--kernel-size', str(cfg['attention']['kernel_size']), '--batch', str(task_batch), '--output', str(task_output)]
                print(f'Running {label} on {task}: {runner}', flush=True)
                subprocess.run(cmd, cwd=ROOT, check=True)
                rows = json.loads(task_output.read_text())
                result['runs'][label].extend(rows)
            if len(rows) != len(ids) or {row['sample_id'] for row in rows} != ids:
                parser.error(f'{label} returned an unexpected set of samples for {task}')
            if any(row['metrics'].get('ignore_eos') is not True or row['metrics']['decode_tokens'] != cfg['model']['max_tokens'] for row in rows):
                parser.error(f'{label} did not produce the fixed-length benchmark workload; check --ignore-eos support')
            result['runs'][label].sort(key=lambda row: row['sample_id'])
            output.with_suffix(f'.{label}.json').write_text(json.dumps(result['runs'][label], indent=2))
            output.write_text(json.dumps(result, indent=2))
    result['throughput'] = {}
    for task_index, task in enumerate(tasks):
        if task not in measured_tasks:
            continue
        measured_ids = set(range(task_index*args.repeats+2, (task_index+1)*args.repeats+1))
        speeds = []
        for label in ('baseline', 'current'):
            rows = [r['metrics'] for r in result['runs'][label] if r['sample_id'] in measured_ids]
            seconds = sum(r['decode_tokens']/r['tokens_per_sec'] for r in rows)
            speeds.append(sum(r['decode_tokens'] for r in rows)/seconds)
        result['throughput'][task] = {'baseline': speeds[0], 'current': speeds[1], 'change_pct': 100*(speeds[1]/speeds[0]-1)}
        print(f'{task}: baseline={speeds[0]:.2f}, current={speeds[1]:.2f} tok/s, change={100*(speeds[1]/speeds[0]-1):+.1f}%', flush=True)
    output.write_text(json.dumps(result, indent=2))
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()
