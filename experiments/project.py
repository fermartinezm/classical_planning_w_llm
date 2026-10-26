import math
import os
import shutil
import tarfile
import subprocess
import sys


from pathlib import Path

from downward.experiment import FastDownwardExperiment
from downward.reports.absolute import AbsoluteReport
from downward.reports.compare import ComparativeReport
from downward.reports.scatter import ScatterPlotReport
from downward.reports.taskwise import TaskwiseReport
from lab import tools
from lab.environments import BaselSlurmEnvironment, TetralithEnvironment, LocalEnvironment
from lab.experiment import ARGPARSER
from lab.reports import Attribute, geometric_mean
from lab.reports.filter import FilterReport

SCRIPT = Path(sys.argv[0]).resolve()

AUTOSCALE_DOMAINS = ["depots",
                     "driverlog",
                     "freecell",
                     "grid",
                     "pipesworld-notankage",
                     "satellite",
                     "storage",
                     "tpp",
                     "visitall",
                     "zenotravel"]

IPC2023_DOMAINS = ["blocksworld",
                   "childsnack",
                   "floortile",
                   "miconic",
                   "rovers",
                   "sokoban",
                   "spanner",
                   "transport"]


# def parse_args():
#     ARGPARSER.add_argument("--tex", action="store_true", help="produce LaTeX output")
#     ARGPARSER.add_argument(
#         "--relative", action="store_true", help="make relative scatter plots"
#     )
#     return ARGPARSER.parse_args()

TEX = True
RELATIVE = False

def get_autoscale_domains(run):
    # This is used to filter out some domains from the reports because the
    # 'AUTOSCALE_DOMAINS' originally used some domains that we no longer care
    # about---basically because they were too hard for the baselines---, but
    # some of the baselines were run on it.
    return run['domain'] in AUTOSCALE_DOMAINS


def make_suite(dom, bench, training=False):
    suite = []
    for d in dom:
        path = os.path.join(bench, d)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        files.sort()
        idx = 0
        for f in files:
            if training and idx == 20:
                break
            if f.endswith('.pddl') and 'domain' not in f:
                suite.append((d, os.path.join(path, f)))
                idx += 1

    return suite

def sample_suite_in_intervals(dom, bench, k=25):
    suite = []
    for d in dom:
        path = os.path.join(bench, d)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.pddl') and 'domain' not in f]
        files.sort()
        for i in range(0, len(files), k):
            suite.append((d, os.path.join(path, files[i])))
            suite.append((d, os.path.join(path, files[len(files)-1])))
    return suite



def is_remote():
    #return re.fullmatch(r"login12|ic[ab]\d\d", platform.node())
    return TetralithEnvironment.is_present() or BaselSlurmEnvironment.is_present()


def get_agile_score(run):
    if run['coverage'] == 0:
        run['score_agile'] = 0
    else:
        time = run.get('total_time')
        if time is None: return run
        if time <= 1.00:
            run['score_agile'] = 1
        else:
            run['score_agile'] = 1.00 - (math.log(time) / math.log(1800))
    return run


def get_expansion_score(run):
    upper_bound = 1e6
    if run['coverage'] == 0:
        run['score_expansion'] = 0
    else:
        expansions = run.get('expansions')
        if not expansions:
            run['score_expansion'] = None
            return run
        if expansions <= 100:
            run['score_expansion'] = 1
        elif expansions > upper_bound:
            run['score_expansion'] = 0
        else:
            run['score_expansion'] = 1 - (math.log(expansions) / math.log(upper_bound))
    return run


def remove_explained_errors(run):
    explained_messages = ["out of memory", "MemoryError"]
    errors = run.get("unexplained_errors")
    if errors:
        run["unexplained_errors"] = [
            error for error in errors
            if all(msg not in error for msg in explained_messages)]
    return True


def add_absolute_report(exp, *, name=None, outfile=None, **kwargs):
    report = AbsoluteReport(**kwargs)
    if name and not outfile:
        outfile = f"{name}.{report.output_format}"
    elif outfile and not name:
        name = Path(outfile).name
    elif not name and not outfile:
        name = f"{exp.name}-abs"
        outfile = f"{name}.{report.output_format}"

    if not Path(outfile).is_absolute():
        outfile = Path(exp.eval_dir) / outfile

    exp.add_report(report, name=name, outfile=outfile)
    if not is_remote():
        exp.add_step(f"open-{name}", subprocess.call, ["xdg-open", outfile])
    #exp.add_step(f"publish-{name}", subprocess.call, ["publish", outfile])


def fetch_algorithm(exp, expname, algo, *, new_algo=None):
    """Fetch (and possibly rename) a single algorithm from *expname*."""
    new_algo = new_algo or algo

    def rename_and_filter(run):
        if run["algorithm"] == algo:
            run["algorithm"] = new_algo
            run["id"][0] = new_algo
            return run
        return False

    exp.add_fetcher(
        f"data/{expname}-eval",
        filter=rename_and_filter,
        name=f"fetch-{new_algo}-from-{expname}",
        merge=True,
    )


def fetch_algorithms(exp, expname, *, algos=None, name=None, filters=None):
    """
    Fetch multiple or all algorithms.
    """
    assert not expname.rstrip("/").endswith("-eval")
    algos = set(algos or [])
    filters = filters or []
    if algos:

        def algo_filter(run):
            return run["algorithm"] in algos

        filters.append(algo_filter)

    exp.add_fetcher(
        f"data/{expname}-eval",
        filter=filters,
        name=name or f"fetch-from-{expname}",
        merge=True,
    )


def add_scatter_plot_reports(exp, algorithm_pairs, attributes, *, filter=None):
    suffix = "-relative" if RELATIVE else ""
    for algo1, algo2 in algorithm_pairs:
        for attribute in attributes:
            exp.add_report(
                ScatterPlotReport(
                    relative=RELATIVE,
                    get_category=lambda run1, run2: run1["domain"],
                    attributes=[attribute],
                    filter_algorithm=[algo1, algo2],
                    filter = tools.make_list(filter),
                    format="tex" if TEX else "png",
                ),
                name=f"{exp.name}-{algo1}-{algo2}-{attribute}{suffix}",
            )


def get_repo_base() -> Path:
    """Get base directory of the repository, as an absolute path.

    Search upwards in the directory tree from the main script until a
    directory with a subdirectory named ".git" is found.

    Abort if the repo base cannot be found."""
    path = Path(SCRIPT)
    while path.parent != path:
        if (path / ".git").is_dir():
            return path
        path = path.parent
    sys.exit("repo base could not be found")


def _get_exp_dir_relative_to_repo():
    repo_name = get_repo_base().name
    script = Path(SCRIPT)
    script_dir = script.parent
    rel_script_dir = script_dir.relative_to(get_repo_base())
    expname = script.stem
    return repo_name / rel_script_dir / "data" / expname


def add_compress_exp_dir_step(exp):
    def compress_exp_dir():
        tar_file_path = Path(exp.path).parent / f"{exp.name}.tar.xz"
        exp_dir_path = Path(exp.path)

        with tarfile.open(tar_file_path, mode="w:xz", dereference=True) as tar:
            for file in exp_dir_path.rglob("*"):
                relpath = file.relative_to(exp_dir_path.parent)
                print(f"Adding {relpath}")
                tar.add(file, arcname=relpath)

        shutil.rmtree(exp_dir_path)

    exp.add_step("compress-exp-dir", compress_exp_dir)


def add_scp_step(exp, login, repos_dir, name="scp-eval-dir"):
    remote_exp = Path(repos_dir) / _get_exp_dir_relative_to_repo()
    exp.add_step(
        name,
        subprocess.call,
        [
            "rsync",
            "-Pavz",
            f"{login}:{remote_exp}-eval/",
            f"{exp.path}-eval/",
        ],
    )
