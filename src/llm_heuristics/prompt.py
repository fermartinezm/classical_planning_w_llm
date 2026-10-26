from src.llm_heuristics.templates import *


def create_prompt(suite, heuristic_name, prompt_format, ablation):
    data = {}

    data["name"] = suite.name
    data["heuristic_name"] = heuristic_name

    h1 = "src/pyperplan/heuristics/blind.py"
    h2 = "src/pyperplan/heuristics/relaxation.py"
    if prompt_format in ['checklist', 'goalcount', 'new', 'reordered', 'neurips']:
        h1 = "src/llm_heuristics/templates/gripper.py"
        h2 = "src/llm_heuristics/templates/logistics.py"
    if ablation == 'heuristics-nocomment':
        h1 = "src/llm_heuristics/templates/gripper_nocomment.py"
        h2 = "src/llm_heuristics/templates/logistics_nocomment.py"

    # TODO: Probably we want better error handling here
    with open(suite.domain, "r") as f:
        lines = f.readlines()
        code = [line for line in lines if not line.strip().startswith(';')]
        data["domain"] = ''.join(code)
    with open(suite.instance1, "r") as f:
        lines = f.readlines()
        code = [line for line in lines if not line.strip().startswith(';')]
        data["instance1"] = ''.join(code)
    with open(suite.instance2, "r") as f:
        lines = f.readlines()
        code = [line for line in lines if not line.strip().startswith(';')]
        data["instance2"] = ''.join(code)
    with open(suite.state, "r") as f:
        data["state"] = f.read()
    with open(suite.static, "r") as f:
        data["static"] = f.read()

    with open(h1, "r") as f:
        lines = f.readlines()
        code = [line for line in lines]
        data["heuristic1"] = ''.join(code)
    with open(h2, "r") as f:
        lines = f.readlines()
        code = [line for line in lines]
        data["heuristic2"] = ''.join(code)
    with open("src/pyperplan/task.py", "r") as f:
        lines = f.readlines()
        code = [line for line in lines]
        data["task"] = ''.join(code)

    if ablation == "false":
        if prompt_format == 'pyperplan':
            template = PYPERPLAN_PROMPT
        elif prompt_format == 'new':
            template = NEW_PROMPT
        elif prompt_format == 'checklist':
            template = CHECKLIST_PROMPT
        elif prompt_format == 'goalcount':
            data["heuristic_goalcount"] = read_goalcount_file()
            template = GOALCOUNT_PROMPT
        elif prompt_format == 'reordered' or prompt_format == 'neurips':
            template = REORDERED_PROMPT
        elif prompt_format == 'ideas':
            template = IDEAS_PROMPT
        else:
            template = BASE_PROMPT
    else:
        data["heuristic_goalcount"] = read_goalcount_file()
        data["heuristic_relaxation"] = read_relaxation_file()
        template = create_ablation_prompt(ablation)

    prompt = template.substitute(data)
    return prompt

def create_ablation_prompt(ablation_option):
    if ablation_option == 'complete' or ablation_option == 'heuristics-nocomment':
        prompt = DESCRIPTION + "\n" + DOMAIN + "\n" + INSTANCES + "\n" + DEPENDENT_HEURISTICS + "\n" + STATE_REPRESENTATION + "\n" + STATIC_REPRESENTATION + "\n" + PLANNER_CODE + "\n" + CHECKLIST
    else:
        if ablation_option == 'description_simple':
            prompt = DESCRIPTION_SIMPLE
        else:
            prompt = DESCRIPTION

        if ablation_option != 'domain':
            prompt = prompt + "\n" + DOMAIN
        if ablation_option != 'instances':
            prompt = prompt + "\n" + INSTANCES

        if ablation_option == 'independent-heuristics':
            prompt = prompt + "\n" + INDEPENDENT_HEURISTICS
        elif ablation_option == 'dependent-heuristics-and-plans':
            prompt = prompt + "\n" + DEPENDENT_HEURISTICS_AND_PLANS
        elif ablation_option != 'dependent-heuristics':
            prompt = prompt + "\n" + DEPENDENT_HEURISTICS

        if ablation_option != 'state-representation':
            prompt = prompt + "\n" + STATE_REPRESENTATION
        if ablation_option != 'static-representation':
            prompt = prompt + "\n" + STATIC_REPRESENTATION
        if ablation_option != 'planner-code':
            prompt = prompt + "\n" + PLANNER_CODE

        if ablation_option == 'checklist-no6':
            prompt = prompt + "\n" + CHECKLIST_NO6
        elif ablation_option != 'checklist':
            prompt = prompt + "\n" + CHECKLIST

    return Template(prompt)

def read_goalcount_file():
    with open("src/pyperplan/heuristics/goalcount.py", "r") as f:
        lines = f.readlines()
        code = [line for line in lines if not line.strip().startswith('#')]
    return ''.join(code)

def read_relaxation_file():
    with open("src/pyperplan/heuristics/relaxation.py", "r") as f:
        lines = f.readlines()

        code = []
        past_initial_comments = False
        for line in lines:
            if not past_initial_comments and line.strip().startswith('#'):
                continue
            else:
                past_initial_comments = True
                code.append(line)

    return ''.join(code)

def create_end_to_end_prompt(suite, instance):
    data = {}

    data["name"] = suite.name
    with open(suite.domain, "r") as f:
        lines = f.readlines()
        code = [line for line in lines if not line.strip().startswith(';')]
        data["domain"] = ''.join(code)
    with open(instance, "r") as f:
        data["instance"] = f.read()

    template = END_TO_END_PROMPT

    prompt = template.substitute(data)
    return prompt
