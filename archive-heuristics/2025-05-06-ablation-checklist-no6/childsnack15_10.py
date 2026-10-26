# childsnackHeuristic.py

from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all children.
    Heuristic components:
    1. Number of unserved children (needs 'serve' action).
    2. Number of sandwiches that need to be made (needs 'make' action).
    3. Number of sandwiches in the kitchen that need to be put on trays (needs 'put_on_tray' action).
    4. Number of distinct non-kitchen locations with unserved children that need a tray moved there (needs 'move_tray' action).

    The heuristic value is calculated as:
    h = N_unserved + 2 * N_to_make + N_kitchen_sandwich + N_locations_need_tray

    Where:
    - N_unserved: Number of children not yet served.
    - N_to_make: Number of sandwiches (considering gluten type) that are needed but not currently in the kitchen or on a tray.
    - N_kitchen_sandwich: Number of sandwiches currently in the kitchen.
    - N_locations_need_tray: Number of distinct non-kitchen places with unserved children where no tray is currently located.

    This heuristic is 0 if and only if all children are served (N_unserved = 0), which implies N_to_make = 0 and N_locations_need_tray = 0.
    It is not admissible but aims to guide the search effectively.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and all objects.
        """
        self.goals = task.goals # Goal conditions are needed to identify all children
        static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.gf_bread = set()
        self.gf_content = set()

        # Extract static properties from static facts
        for fact_str in static_facts:
            parts = get_parts(fact_str)
            predicate = parts[0]
            if predicate == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif predicate == "not_allergic_gluten":
                self.not_allergic_children.add(parts[1])
            elif predicate == "no_gluten_bread":
                self.gf_bread.add(parts[1])
            elif predicate == "no_gluten_content":
                self.gf_content.add(parts[1])

        # Extract all objects of each type from task.facts
        # This assumes task.facts contains all possible ground atoms, implicitly listing all objects.
        self.children = set()
        self.trays = set()
        self.places = set()
        self.sandwiches = set()
        self.bread = set()
        self.content = set()

        # Map predicate arguments to types based on domain definition
        # This mapping is derived from the provided PDDL domain file.
        arg_types = {
            "at_kitchen_bread": {0: "bread-portion"},
            "at_kitchen_content": {0: "content-portion"},
            "at_kitchen_sandwich": {0: "sandwich"},
            "no_gluten_bread": {0: "bread-portion"},
            "no_gluten_content": {0: "content-portion"},
            "ontray": {0: "sandwich", 1: "tray"},
            "no_gluten_sandwich": {0: "sandwich"},
            "allergic_gluten": {0: "child"},
            "not_allergic_gluten": {0: "child"},
            "served": {0: "child"},
            "waiting": {0: "child", 1: "place"},
            "at": {0: "tray", 1: "place"},
            "notexist": {0: "sandwich"},
            "make_sandwich_no_gluten": {0: "sandwich", 1: "bread-portion", 2: "content-portion"},
            "make_sandwich": {0: "sandwich", 1: "bread-portion", 2: "content-portion"},
            "put_on_tray": {0: "sandwich", 1: "tray"},
            "serve_sandwich_no_gluten": {0: "sandwich", 1: "child", 2: "tray", 3: "place"},
            "serve_sandwich": {0: "sandwich", 1: "child", 2: "tray", 3: "place"},
            "move_tray": {0: "tray", 1: "place", 2: "place"},
        }

        type_sets = {
            "child": self.children,
            "bread-portion": self.bread,
            "content-portion": self.content,
            "sandwich": self.sandwiches,
            "tray": self.trays,
            "place": self.places,
        }

        # Populate object sets by inspecting arguments in all possible ground facts
        for fact_str in task.facts:
             parts = get_parts(fact_str)
             predicate = parts[0]
             args = parts[1:]

             if predicate in arg_types:
                 for i, arg_type in arg_types[predicate].items():
                     if i < len(args):
                         type_sets[arg_type].add(args[i])

        # Ensure 'kitchen' constant is in places as it's defined in the domain
        self.places.add("kitchen")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Count unserved children ---
        unserved_children = {c for c in self.children if f"(served {c})" not in state}
        n_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached. Heuristic is 0.
        if n_unserved == 0:
            return 0

        # --- Count sandwiches in different states and their gluten status ---
        kitchen_sandwiches = {s for s in self.sandwiches if f"(at_kitchen_sandwich {s})" in state}
        ontray_sandwiches = {s for s in self.sandwiches if any(f"(ontray {s} {t})" in state for t in self.trays)}

        kitchen_sandwiches_gf = {s for s in kitchen_sandwiches if f"(no_gluten_sandwich {s})" in state}
        ontray_sandwiches_gf = {s for s in ontray_sandwiches if f"(no_gluten_sandwich {s})" in state}

        # --- Count needed sandwiches by type for unserved children ---
        n_needed_gf = sum(1 for c in unserved_children if c in self.allergic_children)
        n_needed_reg = sum(1 for c in unserved_children if c in self.not_allergic_children)

        # --- Calculate sandwiches that still need to be made (by type) ---
        # Available GF sandwiches (kitchen + ontray)
        n_avail_gf = len(kitchen_sandwiches_gf) + len(ontray_sandwiches_gf)
        # Available Regular sandwiches (kitchen + ontray)
        n_avail_reg = (len(kitchen_sandwiches) - len(kitchen_sandwiches_gf)) + (len(ontray_sandwiches) - len(ontray_sandwiches_gf))

        n_make_gf = max(0, n_needed_gf - n_avail_gf)
        n_make_reg = max(0, n_needed_reg - n_avail_reg)
        n_to_make = n_make_gf + n_make_reg

        # --- Count sandwiches in kitchen ---
        n_kitchen_sandwich = len(kitchen_sandwiches)

        # --- Count locations needing a tray ---
        places_with_unserved_children = {
            p for c in unserved_children for fact in state if match(fact, "waiting", c, p)
        }
        tray_locations = {
            p for t in self.trays for fact in state if match(fact, "at", t, p)
        }

        # Non-kitchen locations with unserved children that currently have no tray
        n_locations_need_tray = sum(
            1 for p in places_with_unserved_children if p != "kitchen" and p not in tray_locations
        )

        # --- Calculate heuristic value ---
        # h = N_unserved (serve)
        #   + N_to_make (make)
        #   + (N_kitchen_sandwich + N_to_make) (put_on_tray for all sandwiches that need it)
        #   + N_locations_need_tray (move_tray)
        # h = N_unserved + 2 * N_to_make + N_kitchen_sandwich + N_locations_need_tray

        heuristic_value = n_unserved + (2 * n_to_make) + n_kitchen_sandwich + n_locations_need_tray

        return heuristic_value
