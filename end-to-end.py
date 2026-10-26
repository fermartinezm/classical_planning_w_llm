#!/usr/bin/env python3

import click
import logging
import subprocess
import sys

from src.llm_heuristics.models import *
from src.llm_heuristics.suites import SUITES
from src.llm_heuristics.prompt import create_end_to_end_prompt


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s :: %(message)s",
)
logger = logging.getLogger(__name__)


def extract_plan(answer):
    match = re.search(r"<plan>(.*?)</plan>", answer, re.DOTALL)
    if match:
        return match.group(1)
    return None


def validate_plan(domain, instance, plan_file):
    command = ["validate", "-v", domain, instance, plan_file]
    result = subprocess.run(command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)

    logging.info("VAL output:")
    print(result.stdout)

    logging.info("VAL errors:")
    print(result.stderr)

    return "Plan executed successfully" in result.stdout


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
    "--instance",
    "-i",
    required=True,
    type=str,
    help="Instance used.",
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
    "--plan-file",
    "-f",
    default="plan",
    type=str,
    help="File where the plan is stored.",
)
@click.option(
    "--temperature",
    "-t",
    default=0.1,
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
def main(domain, instance, model, framework, plan_file, temperature, top_p):
    logging.info(f"Python version: {sys.version}.")
    logging.info(f"Using suite {domain}.")
    suite = SUITES[domain]
    logging.info(f"Using model {model}.")
    logging.info(f"Using temperature {temperature}.")
    logging.info(f"Using top-P {top_p}.")

    prompt = create_end_to_end_prompt(suite, instance)
    logging.info("Final prompt: ")
    print(prompt)

    logging.info(f"Using model {model} with framework {framework}.")

    if framework == "gemini":
        answer = run_gemini(model, prompt, temperature, top_p)
    elif framework == "deepseek":
        answer = run_deepseek(model, prompt, temperature, top_p)
    elif framework == "openai":
        answer = run_openai(model, prompt, temperature, top_p)

    logging.info("LLM Answer:")
    print(answer)

    if len(answer) == 0:
        logging.error("LLM returned an empty string!")
        raise ValueError("LLM answer is empty.")

    plan = extract_plan(answer)
    if len(plan) == 0:
         logging.error("LLM returned no plan!")
         raise ValueError("LLM answer has no plan.")

    logging.info("Plan found:")
    print(plan)

    logging.info(
        f"Saving code to {plan_file}."
    )
    with open(plan_file, "w") as f:
        f.write(plan)
        f.close()

    val = validate_plan(suite.domain, instance, plan_file)
    if val:
        logging.info("Valid plan: 1")
    else:
        logging.info("Valid plan: 0")

    logging.info("Finished correctly.")


if __name__ == "__main__":
    main()
