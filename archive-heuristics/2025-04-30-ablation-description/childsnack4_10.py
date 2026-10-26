from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, predicate, *args):
    """
    Check if a PDDL fact matches a given predicate and pattern for arguments.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `predicate`: The expected predicate name (string).
    - `args`: The expected pattern for arguments (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if not parts or parts[0] != predicate:
        return False
    # Match arguments (parts[1:]) against args
    fact_args = parts[1:]
    if len(fact_args) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(fact_args, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve each unserved child.
    It sums up the estimated cost for each child, where the cost for a child depends on the
    most advanced state of an appropriate sandwich for that child.

    # Assumptions
    - The primary goal is to serve all children specified in the task goals.
    - Resource availability (ingredients, trays) is not strictly modeled; the heuristic assumes
      that if a sandwich needs to be made or put on a tray, the necessary resources and
      vehicles (trays) will eventually be available with a fixed cost.
    - The cost of moving a tray is assumed to be 1 action, regardless of distance.
    - Each child requires exactly one sandwich.
    - Tray names start with 'tray' (based on provided examples).

    # Heuristic Initialization
    - Identify all children who need to be served based on the task goals.
    - Extract static information: which children are allergic to gluten and where each child is waiting.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of the estimated costs for each child who is not yet served.
    For each unserved child `c` waiting at place `p`:

    1.  Determine if the child `c` is allergic to gluten. This dictates whether a regular or
        a gluten-free sandwich is required.
    2.  Look for an "appropriate" sandwich `s` in the current state. An appropriate sandwich
        is one that is gluten-free if the child is allergic, or any sandwich otherwise.
    3.  Estimate the minimum number of actions required to get an appropriate sandwich to
        the child and serve it, based on the *most advanced* state of any appropriate sandwich:
        *   **Cost 1 (Serve):** If there is an appropriate sandwich `s` that is already
            `ontray s t` and the tray `t` is `at t p` (at the child's location).
        *   **Cost 2 (Move Tray + Serve):** Else, if there is an appropriate sandwich `s`
            that is `ontray s t` but the tray `t` is `at t p_current` where `p_current != p`.
            (Requires moving the tray).
        *   **Cost 3 (Put on Tray + Move Tray + Serve):** Else, if there is an appropriate
            sandwich `s` that is `at_kitchen_sandwich`. (Requires putting it on a tray,
            moving the tray). Assumes a tray is available in the kitchen.
        *   **Cost 4 (Make + Put on Tray + Move Tray + Serve):** Else (no appropriate
            sandwich exists or is available in a useful state). (Requires making the
            sandwich, putting it on a tray, moving the tray). Assumes ingredients
            and a tray are available.
    4.  The total heuristic value is the sum of these minimum costs for all unserved children.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children and static facts.
        """
        # Children who need to be served (from task goals)
        self.children_to_serve = {
            get_parts(goal)[1] for goal in task.goals if match(goal, "served", "*")
        }

        # Map children to their waiting places (from static facts)
        self.child_to_place = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in task.static
            if match(fact, "waiting", "*", "*")
        }

        # Set of children who are allergic to gluten (from static facts)
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in task.static
            if match(fact, "allergic_gluten", "*")
        }

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to serve
        all unserved children.
        """
        state = node.state

        # Set of children already served in the current state
        served_children_in_state = {
            get_parts(fact)[1] for fact in state if match(fact, "served", "*")
        }

        total_heuristic_cost = 0

        # Find trays and their current locations
        tray_locations = {}
        # Find all existing sandwiches and their properties
        existing_sandwiches = set()
        kitchen_sandwiches = set()
        ontray_sandwiches = {} # Map sandwich -> tray
        sandwich_is_gf = set() # Set of GF sandwich names

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any

            predicate = parts[0]
            args = parts[1:]

            # Assuming tray names start with 'tray' based on examples
            if predicate == 'at' and len(args) == 2 and args[0].startswith('tray'):
                 tray_locations[args[0]] = args[1]
            elif predicate == 'at_kitchen_sandwich' and len(args) == 1:
                 s = args[0]
                 existing_sandwiches.add(s)
                 kitchen_sandwiches.add(s)
            elif predicate == 'ontray' and len(args) == 2:
                 s, t = args
                 existing_sandwiches.add(s)
                 ontray_sandwiches[s] = t
            elif predicate == 'no_gluten_sandwich' and len(args) == 1:
                 s = args[0]
                 sandwich_is_gf.add(s)

        # Calculate cost for each unserved child
        for child in self.children_to_serve:
            if child in served_children_in_state:
                continue # This child is already served

            child_place = self.child_to_place.get(child)
            # Should not happen in valid problems, but defensive check
            if child_place is None:
                 # If a child needs serving but isn't waiting, this indicates an invalid state.
                 # Assign a high cost to discourage reaching such states.
                 total_heuristic_cost += 1000
                 continue

            needs_gluten_free = child in self.allergic_children

            min_child_cost = 4 # Default cost: needs making

            # Check for appropriate sandwiches at different stages
            found_level1 = False

            # Check Level 1: On tray at child's location?
            for s in existing_sandwiches:
                is_appropriate = (not needs_gluten_free) or (s in sandwich_is_gf)
                if is_appropriate and s in ontray_sandwiches:
                    t = ontray_sandwiches[s]
                    if tray_locations.get(t) == child_place:
                        min_child_cost = 1
                        found_level1 = True
                        break # Found the best case for this child

            if found_level1:
                total_heuristic_cost += min_child_cost
                continue # Move to the next child

            # If not Level 1, check Level 2 and 3
            found_level2_or_3 = False
            for s in existing_sandwiches:
                 is_appropriate = (not needs_gluten_free) or (s in sandwich_is_gf)

                 if is_appropriate:
                     # Check Level 2: On tray elsewhere?
                     if s in ontray_sandwiches:
                         # Location doesn't match child_place (already checked by not finding Level 1)
                         min_child_cost = min(min_child_cost, 2)
                         found_level2_or_3 = True
                     # Check Level 3: In kitchen?
                     elif s in kitchen_sandwiches:
                         min_child_cost = min(min_child_cost, 3)
                         found_level2_or_3 = True

            # If found_level2_or_3 is True, min_child_cost is either 2 or 3.
            # If it's still False, min_child_cost remains 4.
            total_heuristic_cost += min_child_cost

        return total_heuristic_cost
