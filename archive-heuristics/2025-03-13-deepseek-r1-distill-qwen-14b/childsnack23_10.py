from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children by:
    1. Making necessary sandwiches.
    2. Putting sandwiches on trays.
    3. Moving trays to children's locations if not already present.
    4. Serving each child.

    # Assumptions:
    - Each child requires exactly one sandwich.
    - Sandwiches can be made with or without gluten based on the child's allergy.
    - Trays must be moved to a child's location if not already there before serving.

    # Heuristic Initialization
    - Extracts static facts about children's allergies and waiting places.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are not yet served.
    2. Determine how many sandwiches need to be made.
    3. Determine how many sandwiches need to be placed on trays.
    4. For each group of children waiting at the same place, check if a tray is already there. If not, add one move action per group.
    5. Sum all necessary actions to estimate the total cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts about children and their properties."""
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract information about children
        self.children = set()
        self.waiting_place = {}
        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                parts = get_parts(fact)
                child = parts[1]
                self.children.add(child)
                # Default waiting place if not specified (should be specified in static facts)
                self.waiting_place[child] = "kitchen"  # placeholder
            elif match(fact, "waiting", "*", "*", "*"):
                parts = get_parts(fact)
                child = parts[1]
                place = parts[3]
                self.children.add(child)
                self.waiting_place[child] = place

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        static_facts = node.task.static

        # Count the number of unserved children
        unserved_children = []
        for child in self.children:
            served = False
            for fact in state:
                if match(fact, "served", child):
                    served = True
                    break
            if not served:
                unserved_children.append(child)
        num_unserved = len(unserved_children)

        if num_unserved == 0:
            return 0

        # Count the number of made sandwiches
        num_made_sandwiches = 0
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                num_made_sandwiches += 1

        # Count the number of sandwiches on trays
        num_on_tray = 0
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                num_on_tray += 1

        # Calculate the number of actions needed for making and putting on trays
        make_needed = max(0, num_unserved - num_made_sandwiches)
        put_needed = max(0, num_made_sandwiches - num_on_tray)

        # Group unserved children by their waiting place
        groups = {}
        for child in unserved_children:
            place = self.waiting_place[child]
            if place not in groups:
                groups[place] = 0
            groups[place] += 1

        # Determine which places already have trays
        trays_at_place = {}
        for fact in state:
            if match(fact, "at", "tray*", "*"):
                parts = get_parts(fact)
                place = parts[2]
                trays_at_place[place] = True

        # Calculate the number of moves needed
        move_needed = 0
        for place in groups:
            if place not in trays_at_place:
                move_needed += 1

        # Total heuristic is the sum of all needed actions
        total_cost = make_needed + put_needed + move_needed + num_unserved

        return total_cost
