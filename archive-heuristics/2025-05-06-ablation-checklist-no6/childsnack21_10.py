from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure fact has at least as many parts as args for a potential match
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all children.
    The heuristic is the sum of estimated costs for distinct stages:
    1. Making sandwiches that are not yet made.
    2. Putting sandwiches onto trays if they are in the kitchen.
    3. Moving trays to locations where unserved children are waiting but no tray is present.
    4. Serving the sandwiches to the unserved children.

    This heuristic is not admissible but aims to guide a greedy best-first search
    efficiently by estimating progress towards satisfying the goal conditions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal information and static facts.

        Args:
            task: The planning task object containing goals and static facts.
        """
        # Identify all children that need to be served from the goal state
        self.children_to_serve = set()
        for goal in task.goals:
            # Goal facts are typically of the form (served ?child)
            if match(goal, "served", "*"):
                parts = get_parts(goal)
                if len(parts) > 1:
                    self.children_to_serve.add(parts[1])

        # Extract static information about children's waiting places and allergies
        # This information does not change during planning
        self.waiting_places_map = {} # Maps child name to their waiting place
        self.allergy_status = {}     # Maps child name to 'allergic' or 'not_allergic'

        for fact in task.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                if len(parts) > 2:
                    child, place = parts[1], parts[2]
                    self.waiting_places_map[child] = place
            elif match(fact, "allergic_gluten", "*"):
                parts = get_parts(fact)
                if len(parts) > 1:
                    child = parts[1]
                    self.allergy_status[child] = 'allergic'
            elif match(fact, "not_allergic_gluten", "*"):
                parts = get_parts(fact)
                if len(parts) > 1:
                    child = parts[1]
                    self.allergy_status[child] = 'not_allergic'

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The current state node in the search tree.

        Returns:
            An integer estimate of the remaining cost to reach a goal state.
        """
        state = node.state

        # 1. Count unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = [c for c in self.children_to_serve if c not in served_children]
        unserved_children_count = len(unserved_children)

        # If all children who need serving are served, the heuristic is 0
        if unserved_children_count == 0:
            return 0

        # 2. Count sandwiches at different stages of the process
        at_kitchen_sandwich_count = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*"))
        ontray_count = sum(1 for fact in state if match(fact, "ontray", "*", "*"))
        # notexist_count = sum(1 for fact in state if match(fact, "notexist", "*")) # We don't need this directly, we need total needed

        # 3. Estimate actions needed for sandwiches
        # We need unserved_children_count sandwiches in total to serve everyone.
        # Sandwiches needing to be made: total needed - total already made (at_kitchen or ontray)
        # This is a lower bound on the number of 'make_sandwich' actions required.
        make_cost = max(0, unserved_children_count - (at_kitchen_sandwich_count + ontray_count))

        # Sandwiches needing to be put on a tray: total needed on tray - total already on tray
        # This is a lower bound on the number of 'put_on_tray' actions required.
        # Sandwiches must be at_kitchen_sandwich to be put on a tray.
        # This count represents sandwiches that are either currently at_kitchen_sandwich
        # or will be made and then need to be put on a tray.
        put_on_tray_cost = max(0, unserved_children_count - ontray_count)


        # 4. Estimate actions needed for tray movement
        # Identify the set of places where unserved children are waiting.
        waiting_places_with_unserved = set(
            self.waiting_places_map[c] for c in unserved_children if c in self.waiting_places_map
        )

        # Identify the set of places where trays are currently located.
        places_with_trays = set(get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*"))

        # Count the number of places with unserved children that currently do not have a tray.
        # Each such place will require at least one tray movement action to bring a tray there.
        places_needing_tray_move = waiting_places_with_unserved - places_with_trays
        move_tray_cost = len(places_needing_tray_move)

        # 5. Estimate actions needed for serving
        # Each unserved child needs one 'serve_sandwich' action.
        serve_cost = unserved_children_count

        # The total heuristic is the sum of estimated actions for each necessary stage.
        # This sum provides a non-admissible estimate of the remaining steps.
        total_cost = make_cost + put_on_tray_cost + move_tray_cost + serve_cost

        return total_cost

