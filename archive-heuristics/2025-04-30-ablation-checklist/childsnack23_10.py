from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists and contains the Heuristic class
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL fact strings with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions required to serve all children.
    The heuristic is the sum of estimated costs for reaching different stages
    of the process for all unserved children:
    1. Making sandwiches: Number of sandwiches of each type (GF/Reg) that still need to be made.
    2. Putting on tray: Number of needed sandwiches (GF/Reg) that exist but are not yet on a tray.
    3. Moving trays: Number of distinct locations where unserved children are waiting, but no tray is present.
    4. Serving: Number of unserved children (representing the final 'serve' action for each).

    This heuristic is not admissible but aims to guide a greedy search efficiently
    by prioritizing states where more prerequisites for serving children are met.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals # Goal conditions, e.g., frozenset({'(served child1)', '(served child2)'})
        self.static_facts = task.static # Static facts, e.g., frozenset({'(allergic_gluten child1)', '(waiting child1 table1)'})

        # Pre-process static facts for quick lookup
        self.allergic_children = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "allergic_gluten", "*")}
        # Map child to waiting place. Filter out children not in waiting facts if any.
        self.waiting_info = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static_facts if match(fact, "waiting", "*", "*")}

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify unserved children and their types
        unserved_children = set()
        unserved_allergic_count = 0
        unserved_non_allergic_count = 0

        # Goal facts are typically (served childX)
        for goal_fact in self.goals:
            if match(goal_fact, "served", "*"):
                child = get_parts(goal_fact)[1]
                if goal_fact not in state:
                    unserved_children.add(child)
                    if child in self.allergic_children:
                        unserved_allergic_count += 1
                    else:
                        unserved_non_allergic_count += 1

        num_unserved = len(unserved_children)

        if num_unserved == 0:
            return 0 # Goal reached

        # 2. Count existing sandwiches by type and location stage
        # Sandwiches at_kitchen_sandwich
        kitchen_gf = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*") and f"(no_gluten_sandwich {get_parts(fact)[1]})" in state)
        kitchen_reg = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*") and f"(no_gluten_sandwich {get_parts(fact)[1]})" not in state)

        # Sandwiches ontray
        ontray_gf = sum(1 for fact in state if match(fact, "ontray", "*", "*") and f"(no_gluten_sandwich {get_parts(fact)[1]})" in state)
        ontray_reg = sum(1 for fact in state if match(fact, "ontray", "*", "*") and f"(no_gluten_sandwich {get_parts(fact)[1]})" not in state)

        # Total existing sandwiches (at_kitchen or ontray)
        existing_gf = kitchen_gf + ontray_gf
        existing_reg = kitchen_reg + ontray_reg

        # 3. Calculate heuristic components based on stages

        # Cost 1: Sandwiches needing to be made
        # Number of needed GF/Reg sandwiches that don't exist yet.
        make_cost = max(0, unserved_allergic_count - existing_gf) + \
                    max(0, unserved_non_allergic_count - existing_reg)

        # Cost 2: Sandwiches needing to be put on tray
        # Number of needed GF/Reg sandwiches that are not yet on a tray.
        # This counts how many more GF/Reg sandwiches need to reach the 'ontray' state.
        put_cost = max(0, unserved_allergic_count - ontray_gf) + \
                   max(0, unserved_non_allergic_count - ontray_reg)

        # Cost 3: Trays needing to be moved
        # Count distinct locations where unserved children wait, and no tray is present.
        # Only consider children whose waiting location is known from static facts.
        waiting_places_with_unserved = {self.waiting_info[c] for c in unserved_children if c in self.waiting_info}
        tray_locations = {get_parts(fact)[2] for fact in state if match(fact, "at", "tray*", "*")}

        needed_tray_locations = waiting_places_with_unserved - tray_locations
        # Trays typically start at the kitchen. Moving *to* the kitchen is not a delivery step for a child waiting elsewhere.
        # If a child is waiting *at* the kitchen, no tray move is needed for that location.
        if 'kitchen' in needed_tray_locations:
             needed_tray_locations.remove('kitchen')

        move_cost = len(needed_tray_locations)

        # Cost 4: Children needing to be served
        # Each unserved child represents a final 'serve' action needed.
        serve_cost = num_unserved

        # Total heuristic is the sum of costs for each stage
        total_cost = make_cost + put_cost + move_cost + serve_cost

        return total_cost
