import click
import itertools
import json

from collections import defaultdict
from dataclasses import dataclass
from project import get_expansion_score

@dataclass
class Experiment:
    path: str
    name: str


def get_results_single_experiment(filename, attr):

    with open("/".join([filename,"properties"]), 'r') as f:
        props = json.load(f)

    data = defaultdict()

    for p, run in props.items():
        domain = run['domain']
        algo = run['algorithm']
        if data.get(domain) is None:
            data[domain] = defaultdict(int)
        if run.get(attr) is not None:
            data[domain][algo] += run.get(attr)

    simplified_data = defaultdict(list)
    for key, value in data.items():
        simplified_data[key] = [v for k, v in value.items()]

    return simplified_data

@click.command
@click.argument('eval_dir', type=str)
def main(eval_dir):
    ATTRIBUTE = 'coverage'

    exp = Experiment(eval_dir, 'exp')

    results = defaultdict(list)

    exp_results = get_results_single_experiment(exp.path, ATTRIBUTE)

    print('domain 1 5 10 15 20 25')
    total_res = [0, 0, 0, 0, 0, 0]
    for domain, results in exp_results.items():
        dom_res = []
        for K in [1, 5, 10, 15, 20, 25]:
            N = 25 * K
            if K == 25:
                N = 1
            all_combinations = list(itertools.combinations(results, K))
            coverage = []
            count = 0
            for comb in all_combinations:
                coverage.append(max(comb))
                count += 1
            dom_res.append(sum(coverage)/count)
        for idx, r in enumerate(dom_res):
            total_res[idx] += r
        str_res = [f"{x:.2f}" for x in dom_res]
        print(f"{domain} {' '.join(str_res)}")

    print("==================")
    final_res = [f"{x:.2f}" for x in total_res]
    print(f"TOTAL {' '.join(final_res)}")



if __name__ == '__main__':
    main()
