#!/usr/bin/env python3

import click
import logging
import os
from pathlib import Path
import sys

from src.llm_heuristics import models
from src.llm_heuristics.suites import DomainSuite, SUITES
from src.llm_heuristics.prompt import create_prompt


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s :: %(message)s",
)
logger = logging.getLogger(__name__)


def validate_heuristic_name(ctx, param, value):
    if not value.endswith("Heuristic"):
        raise click.BadParameter("The heuristic name must end with 'Heuristic'.")
    return value


def validate_heuristic_file(ctx, param, value):
    if not value.endswith(".py"):
        raise click.BadParameter("The heuristic name must end with '.py'.")
    return value


def validate_temperature(ctx, param, value):
    if value > 2 or value < 0:
        raise click.BadParameter("The temperature must be in the interval [0,2].")
    return value

def validate_top_p(ctx, param, value):
    if value > 1 or value < 0:
        raise click.BadParameter("The top-P must be in the interval [0,1].")
    return value


@click.command()
@click.option(
    "--domain",
    "-d",
    required=True,
    type=click.Choice(SUITES.keys()),
    help="Name of the domain used. Everything else is inferred.",
)
@click.option(
    "--framework",
    default="gemini",
    type=click.Choice(["gemini", "deepseek", "openai", "openai-compatible"]),
    help="Framework used to run LLMs.",
)
@click.option(
    "--base-url",
    default=None,
    help="Optional base URL for openai-compatible endpoints (e.g. http://localhost:11434/v1 for Ollama, https://openrouter.ai/api/v1).",
)
@click.option(
    "--suite-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Optional custom training directory (e.g. for L1/L2/L3 obfuscated domain containing domain.pddl, p01.pddl, p99.pddl, example-state.out, example-static.out).",
)
@click.option(
    "--num-candidates",
    "-k",
    default=1,
    type=int,
    help="Number of candidate heuristics to generate (use 10 for the fast protocol).",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory to write generated candidate files (<domain><i_10>.py) when --num-candidates > 1.",
)
@click.option(
    "--model",
    "-m",
    default="gemini-2.0-flash-thinking",
    help="LLM Model used.",
)
@click.option(
    "--heuristic-name",
    "-n",
    default="NewDomainDependentHeuristic",
    callback=validate_heuristic_name,
    help="Name of the heuristic and of its class. Name must end with 'Heuristic'. \
    NOTE: This impacts how you will call it from pyperplan!",
)
@click.option(
    "--heuristic-file",
    "-f",
    default="new-heuristic.py",
    callback=validate_heuristic_file,
    help="File where the learnt heuristic is stored. It must be a Python file.",
)
@click.option(
    "--prompt-format",
    default="neurips",
    type=click.Choice(["base", "checklist", "goalcount", "new", "pyperplan", "reordered", "neurips"]),
    help="Format of the prompt passed to the heuristic.",
)
@click.option(
    "--temperature",
    "-t",
    default=1.0,
    type=float,
    callback=validate_temperature,
    help="Model temperature.",
)
@click.option(
    "--top-p",
    default=0.5,
    type=float,
    callback=validate_top_p,
    help="Model top-P.",
)
@click.option(
    "--ablation",
    default="false",
    type=click.Choice(["false", "complete", "description_simple", "domain", "instances", "dependent-heuristics", "state-representation", "static-representation",
                       "planner-code", "checklist", "checklist-no6", "independent-heuristics", "heuristics-nocomment", "dependent-heuristics-and-plans"]),
    help="Ablation options for reordered prompt.",
)
def main(domain, model, framework, base_url, suite_dir, num_candidates, output_dir, heuristic_name, heuristic_file, prompt_format, temperature, top_p, ablation):
    logging.info(f"Python version: {sys.version}.")
    logging.info(f"Using suite {domain}.")
    if suite_dir:
        sdir = Path(suite_dir)
        suite = DomainSuite(
            name="secret_domain",
            domain=str(sdir / "domain.pddl"),
            instance1=str(sdir / "p01.pddl"),
            instance2=str(sdir / "p99.pddl"),
            state=str(sdir / "example-state.out"),
            static=str(sdir / "example-static.out"),
        )
    else:
        suite = SUITES[domain]
    logging.info(f"Using model {model}.")
    logging.info(f"Using temperature {temperature}.")
    logging.info(f"Using top-P {top_p}.")
    logging.info(f"Using ablation option {ablation}")
    logging.info(f"Generating prompt with format {prompt_format}")
    prompt = create_prompt(suite, heuristic_name, prompt_format, ablation)
    logging.info("Final prompt: ")
    print(prompt)

    logging.info(f"Using model {model} with framework {framework} (num_candidates={num_candidates}).")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for idx in range(1, num_candidates + 1):
        target_dir = output_dir or str(Path(heuristic_file).parent or Path("."))
        target_file = (
            os.path.join(target_dir, f"{domain}{idx}_{num_candidates}.py")
            if (output_dir or num_candidates > 1)
            else heuristic_file
        )
        if output_dir and os.path.exists(target_file) and os.path.getsize(target_file) > 0:
            logging.info(f"Skipping existing candidate {target_file}")
            continue

        if framework == "gemini":
            answer = models.run_gemini(model, prompt, temperature, top_p)
        elif framework == "deepseek":
            answer = models.run_deepseek(model, prompt, temperature, top_p)
        elif framework == "openai":
            answer = models.run_openai(model, prompt, temperature, top_p)
        elif framework == "openai-compatible":
            answer = models.run_openai_compatible(model, prompt, temperature, top_p, base_url=base_url)

        logging.info(f"LLM Answer (candidate {idx}/{num_candidates}):")
        print(answer)

        if not answer:
            logging.error(f"LLM returned an empty string for candidate {idx}!")
            if num_candidates == 1:
                raise ValueError("LLM answer is empty.")
            continue

        code = models.sanitize_llm_answer(answer)
        logging.info("Code extracted:")
        print(code)

        if not code:
            logging.error(f"LLM returned no Python code for candidate {idx}!")
            if num_candidates == 1:
                raise ValueError("LLM answer has no Python code.")
            continue

        logging.info(f"Saving code to {target_file}.")
        Path(target_file).parent.mkdir(parents=True, exist_ok=True)
        with open(target_file, "w") as f:
            f.write(code)

    logging.info("Finished correctly.")


if __name__ == "__main__":
    main()
