import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from REINFORCE_CartPole import train

# name -> (reward_to_go, baseline)
settings = {
    'whole episode':           (False, False),
    'reward-to-go':            (True,  False),
    'reward-to-go + baseline': (True,  True),
}
SEEDS = [0, 1, 2]
EPOCHS = 100


def main():
    # 1. run all 9 experiments (training output is hidden to keep the console clean)
    results = {}
    for name, (rtg, base) in settings.items():
        for seed in SEEDS:
            print(f'running: {name}, seed {seed} ...')
            with redirect_stdout(io.StringIO()):
                returns = train(seed=seed, reward_to_go=rtg,
                                baseline=base, epochs=EPOCHS)
            results[(name, seed)] = returns

    # 2. save every run's per-epoch mean return to a CSV (900 rows)
    rows = []
    for (name, seed), returns in results.items():
        for epoch, r in enumerate(returns):
            rows.append({'setting': name, 'seed': seed,
                         'epoch': epoch, 'mean_return': r})
    pd.DataFrame(rows).to_csv('results.csv', index=False)

    # 3. summary numbers, for checking against your reference
    print('\nsetting                   seed  last-10 mean  first epoch >= 475')
    for (name, seed), returns in results.items():
        r = np.array(returns)
        hits = np.where(r >= 475)[0]
        first = int(hits[0]) if len(hits) else 'never'
        print(f'{name:25s} {seed:4d}  {r[-10:].mean():12.0f}  {first}')

    # 4. plot: mean over seeds per setting, with a min-max band
    x = np.arange(EPOCHS)
    plt.figure(figsize=(9, 5))
    for name in settings:
        arr = np.array([results[(name, s)] for s in SEEDS])
        plt.plot(x, arr.mean(axis=0), label=name)
        plt.fill_between(x, arr.min(axis=0), arr.max(axis=0), alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean episode return')
    plt.title('REINFORCE on CartPole-v1 (mean of 3 seeds, band = min-max)')
    plt.legend()
    plt.grid(True)
    plt.savefig('returns_comparison.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()