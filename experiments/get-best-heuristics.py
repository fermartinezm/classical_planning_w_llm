import click
import json

from collections import defaultdict

from project import get_agile_score, get_expansion_score

@click.command()
@click.argument('filename')
@click.option(
    "--evaluator",
    default="expansion",
    type=click.Choice(["expansion", "agile"]),
    help="Score used to classify heuristic.",
)
def main(filename, evaluator):
    def get_sum_score(s):
        return sum(s)

    with open(filename, 'r') as f:
        props = json.load(f)


    main_evaluator = evaluator
    if main_evaluator == 'expansion':
        secondary_evaluator = 'agile'
    elif main_evaluator == 'agile':
        secondary_evaluator = 'expansion'

    domains = set()
    coverage = dict()
    scores = dict()
    second_scores = dict()

    for p, k in props.items():
        alg = k['algorithm']
        run = get_expansion_score(k)
        run = get_agile_score(k)
        score = run.get(f'score_{main_evaluator}')
        second_score = run.get(f'score_{secondary_evaluator}')
        d = k['domain']
        domains.add(d)
        if d not in coverage:
            coverage[d] = defaultdict(int)
            scores[d] = defaultdict(list)
            second_scores[d] = defaultdict(list)
        if k.get('coverage') == 1:
            coverage[d][alg] += 1
            scores[d][alg].append(score)
            second_scores[d][alg].append(second_score)

    best = dict()

    for d in domains:
        for alg in coverage[d]:
            if best.get(d) is None:
                best[d] = (alg, coverage[d][alg], get_sum_score(scores[d][alg]), get_sum_score(second_scores[d][alg]))
            else:
                best_alg, best_coverage, best_score, best_second_score = best[d]
                score = get_sum_score(scores[d][alg])
                second_score = get_sum_score(second_scores[d][alg])
                cover = coverage[d][alg]
                if cover > best_coverage:
                    best[d] = (alg, cover, score, second_score)
                elif cover == best_coverage:
                    if score > best_score:
                        best[d] = (alg, cover, score, second_score)
                    elif score == best_score and second_score > best_second_score:
                        best[d] = (alg, cover, score, second_score)

    print(f"domain algorithm coverage score_{main_evaluator} score_{secondary_evaluator}")
    for dom, (alg, cover, score, second_score) in sorted(best.items()):
        print(f"{dom} {alg} {cover} {score} {second_score}")


    prefix='gemini-2.0-ft-'
    for dom, (alg, cover, score, second_score) in sorted(best.items()):
        heur = alg.replace(prefix,'')
        print(f"'{dom}' : '{heur}',")

if __name__ == '__main__':
    main()
