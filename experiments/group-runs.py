import json
import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from project import get_expansion_score

@dataclass
class Experiment:
    path: str
    name: str


ATTRIBUTES = ['coverage']


def get_results_single_experiment(filename):

    with open("/".join([filename,"properties"]), 'r') as f:
        props = json.load(f)

    data = defaultdict()
    for attr in ATTRIBUTES:
        data[attr] = defaultdict(list)

    for p, run in props.items():
        domain = run['domain']
        run = get_expansion_score(run)
        algo = run['algorithm']
        for attr in ATTRIBUTES:
            if data[attr].get(domain) is None:
                data[attr][domain] = defaultdict(int)
            if run.get(attr) is not None:
                if run.get(attr) == 1:
                    if run.get('search_time') <= 300:
                        data[attr][domain][algo] += run.get(attr)
                    #else:
                    #    data[attr][domain][algo] += 0
                #else:
                #    data[attr][domain][algo] += run.get(attr)

    summed_stdev = 0
    result = defaultdict()
    best = 0
    worst = 0
    for attr in ATTRIBUTES:
        attr_data = data[attr]
        if result.get(attr) is None:
            result[attr] = defaultdict(list)
        for domain in attr_data:
            values = []
            d = attr_data[domain]
            total = 0
            for _, value in attr_data[domain].items():
                total += value
                values.append(value)
            non_zero_heuristics = len(attr_data[domain].keys())
            if len(attr_data[domain].keys()) != 0:
                result[attr][domain] = total/len(attr_data[domain].keys())
            else:
                result[attr][domain] = 0
            print(f"{domain} {total} {np.mean(values)} {np.std(values)}")
            best += max(values + [0])
            worst += min(values) if len(values) > 0 else 0
            summed_stdev += np.std(values)
        print(f"@@@ Summed stdev: {summed_stdev}")
        print(f"@@@ Best: {best}")
        print(f"@@@ Worst: {worst}")
    return result

def main():
    experiments = [
                   Experiment('data/2025-05-05-C-ablation-dependent-heuristics-eval/', 'no-heuristics'),
                   Experiment('data/2025-05-05-G-ablation-planner-code-eval/', 'no-planner-code'),
                   Experiment('data/2025-05-05-H-ablation-state-eval/', 'no-state'),
                   Experiment('data/2025-05-05-J-ablation-checklistno6-eval/', 'no-checklistno6'),
                   Experiment('data/2025-05-05-D-ablation-description-eval/', 'no-description'),
                   Experiment('data/2025-05-05-I-ablation-static-eval/', 'no-static'),
                   Experiment('data/2025-05-05-K-ablation-dependent-heuristics-and-plans-eval/', 'with-plans'),
                   Experiment('data/2025-05-05-L-ablation-heuristics-nocomment-eval/', 'no-comments'),
                   Experiment('data/2025-05-05-M-ablation-independent-heuristics-eval/', 'domain-indep-heuristics'),
                   Experiment('data/2025-05-05-A-ablation-checklist-eval/', 'no-checklist'),
                   Experiment('data/2025-05-05-E-ablation-domain-eval/', 'no-domain'),
                   Experiment('data/2025-05-05-F-ablation-instances-eval/', 'no-instances'),
                   Experiment('data/2025-05-05-B-ablation-complete-eval/', 'complete'),
                   Experiment('data/2025-05-16-A-deepseek-r1-full-testing-eval/', 'deepseek-r1'),
                   ]

    results = defaultdict(list)
    names = [exp.name for exp in experiments]

    for exp in experiments:
        print(f"@@@@ Experiment: {exp.name}")
        exp_results = get_results_single_experiment(exp.path)
        for attr in ATTRIBUTES:
            if results.get(attr) is None:
                results[attr] = defaultdict(list)
            for domain, value in exp_results[attr].items():
                if results[attr].get(domain) is None:
                    results[attr][domain] = defaultdict(list)
                results[attr][domain][exp.name] = value

    for attr in ATTRIBUTES:
        print('\n\n', attr)
        print("domain", " ".join(names))
        acc = [0.0 for exp in experiments]
        for domain, values in results[attr].items():
            r = " ".join([f"{values[n]:.2f}" for n in names])
            acc = [x + float(values[y]) for x, y in zip(acc, values)]
            print(domain, r)
        print('------------')
        print('TOTAL', ' '.join([f"{a:.2f}" for a in acc]))
        print('============')



if __name__ == '__main__':
    main()
