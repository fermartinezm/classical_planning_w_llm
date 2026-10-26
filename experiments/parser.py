from lab.parser import Parser

def make_parser():
    def solved(content, props):
        props["coverage"] = int("plan_length" in props)

    def add_raw_memory(content, props):
        if "memory" in props:
            props["raw_memory"] = props["memory"]

    def error(content, props):
        if props["coverage"]:
            props["error"] = "solved"
        else:
            props["error"] = f"exitcode-{props.get('planner_exit_code', 'unknown')}"

    parser = Parser()
    parser.add_pattern(
        "node", r"node: (.+)\n", type=str, file="driver.log", required=True
    )
    parser.add_pattern(
        "planner_exit_code", r"plan exit code: (.+)\n", type=int, file="driver.log"
    )
    parser.add_pattern(
        "planner_wall_clock_time", r"plan wall-clock time: (.+)s", type=float, file="driver.log"
    )
    parser.add_pattern("plan_length", r"Plan length: (\d+)\n", type=int)
    parser.add_pattern("expansions", r"(\d+) Nodes expanded\n", type=int, flags="I")
    parser.add_pattern("max_open_list_size", r"Max open list size: (\d+)\n", type=int)
    parser.add_pattern("initial_h_value", r"Initial h value: (.+)", type=str)
    parser.add_pattern("search_time", r"Search time: (.+)", type=float)
    parser.add_pattern("total_time", r"Total time: (.+)", type=float)
    parser.add_pattern("memory", r"Peak memory: (.+) KB", type=int)
    parser.add_function(add_raw_memory)
    parser.add_function(solved)
    parser.add_function(error)
    return parser
