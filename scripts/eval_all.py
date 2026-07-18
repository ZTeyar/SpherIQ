#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import itertools
import pandas as pd
from cross_validate import evaluate, DATASET_CFG

DATASETS = sorted(DATASET_CFG.keys())
GRID_SIZE = 16

results = []
for trained_on, evaluate_on in itertools.product(DATASETS, repeat=2):
    try:
        pcc, srcc = evaluate(trained_on, evaluate_on, tta_yaw_angles=[0],
                             batch_size=1, cpu_workers=4, grid_size=GRID_SIZE)
        results.append({
            'trained_on': trained_on,
            'evaluate_on': evaluate_on,
            'PCC': round(pcc, 4),
            'SRCC': round(srcc, 4),
        })
    except Exception as e:
        results.append({
            'trained_on': trained_on,
            'evaluate_on': evaluate_on,
            'PCC': None,
            'SRCC': None,
        })
        print(f'  ERROR: {e}')

df = pd.DataFrame(results)
print('\n' + '=' * 60)
print('PCC Matrix')
print('=' * 60)
pcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='PCC')
print(pcc_mat.to_string())

print('\n' + '=' * 60)
print('SRCC Matrix')
print('=' * 60)
srcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='SRCC')
print(srcc_mat.to_string())

df.to_csv('cross_eval_results.csv', index=False)
print(f'\nSaved cross_eval_results.csv')
