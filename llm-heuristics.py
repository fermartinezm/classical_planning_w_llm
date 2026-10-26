#!/usr/bin/env python3

import click
import logging
import sys

from src.llm_heuristics import models
from src.llm_heuristics.suites import SUITES
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
    type=click.Choice(["gemini", "deepseek", "openai"]),
    help="Framework used to run LLMs.",
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
def main(domain, model, framework, heuristic_name, heuristic_file, prompt_format, temperature, top_p, ablation):
    logging.info(f"Python version: {sys.version}.")
    logging.info(f"Using suite {domain}.")
    suite = SUITES[domain]
    logging.info(f"Using model {model}.")
    logging.info(f"Using temperature {temperature}.")
    logging.info(f"Using top-P {top_p}.")
    logging.info(f"Using ablation option {ablation}")
    logging.info(f"Generating prompt with format {prompt_format}")
    prompt = create_prompt(suite, heuristic_name, prompt_format, ablation)
    logging.info("Final prompt: ")
    print(prompt)

    logging.info(f"Using model {model} with framework {framework}.")

    if framework == "gemini":
        answer = models.run_gemini(model, prompt, temperature, top_p)
    elif framework == "deepseek":
        answer = models.run_deepseek(model, prompt, temperature, top_p)
    elif framework == "openai":
        answer = models.run_openai(model, prompt, temperature, top_p)

    logging.info("LLM Answer:")
    print(answer)

    if not answer:
        logging.error("LLM returned an empty string!")
        raise ValueError("LLM answer is empty.")

    code = models.sanitize_llm_answer(answer)
    logging.info("Code extracted:")
    print(code)

    if not code:
        logging.error("LLM returned no Python code!")
        raise ValueError("LLM answer has no Python code.")

    logging.info(
        f"Saving code to {heuristic_file}."
    )
    with open(heuristic_file, "w") as f:
        f.write(code)
        f.close()

    logging.info("Finished correctly.")


if __name__ == "__main__":
    main()
