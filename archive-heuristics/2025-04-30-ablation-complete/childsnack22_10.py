from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: '(at tray1 kitchen)' -> ['at', 'tray1', 'kitchen']
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children (representing the final 'serve' action),
    the number of locations needing a tray (representing 'move_tray'), and the number
    of sandwiches that need to be made and/or put on a tray to satisfy the demand
    from unserved children (representing 'make_sandwich' and 'put_on_tray').

    # Assumptions
    - Sufficient bread, content, and sandwich slots (`notexist`) are available
      in the kitchen to make any required sandwiches.
    - Sufficient trays are available to put sandwiches on and move to locations.
    - Each action (make, put_on_tray, move_tray, serve) costs 1.
    - The heuristic is non-admissible and aims to guide a greedy best-first search.

    # Heuristic Initialization
    - Identify all children that are part of the goal (i.e., need to be served).
    - Identify which children are allergic to gluten from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify all children who are currently waiting (`waiting ?c ?p`) but
        are not yet served (`served ?c` is not in state). Count the total number
        of such unserved children. This count contributes to the heuristic as
        each child requires a final 'serve' action.
    2.  Group the unserved children by their waiting location (`?p`). Identify all
        distinct locations where unserved children are waiting.
    3.  For each distinct location identified in step 2, check if there is currently
        any tray present at that location (`at ?t ?p`). Count the number of locations
        that have unserved children but no tray. This count contributes to the
        heuristic as each such location requires a 'move_tray' action.
    4.  Determine the total number of gluten-free (GF) and regular sandwiches
        needed by the unserved children based on their allergy status.
    5.  Count the number of GF and regular sandwiches that are currently available
        and on a tray (`ontray ?s ?t`).
    6.  Calculate the number of GF and regular sandwiches that are needed by
        unserved children but are *not* currently on a tray. These sandwiches
        must either be made or are currently in the kitchen (`at_kitchen_sandwich`).
    7.  Count the number of GF and regular sandwiches that are currently in the
        kitchen (`at_kitchen_sandwich`).
    8.  Estimate the cost to get the needed-but-not-on-tray sandwiches onto trays:
        - For needed sandwiches that are currently in the kitchen: Each requires
          a 'put_on_tray' action (cost 1). Prioritize using these first.
        - For needed sandwiches that are not in the kitchen (must be made): Each
          requires a 'make_sandwich' (cost 1) followed by a 'put_on_tray' (cost 1),
          totaling cost 2.
    9.  Sum the costs from steps 1, 3, and 8. This sum represents the estimated
        number of actions remaining to reach the goal state. The heuristic is 0
        if and only if all children are served.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children and allergy status.
        """
        # Identify all children that are part of the goal (i.e., need to be served)
        self.goal_children = {get_parts(goal)[1] for goal in task.goals if match(goal, "served", "*")}

        # Extract static facts about allergies
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in task.static
            if match(fact, "allergic_gluten", "*")
        }
        # Note: not_allergic_gluten is also static, but we only need the allergic list.

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # --- Step 1 & 2: Identify unserved children and their locations ---
        unserved_children_at_place = {}
        served_children = set()

        # First, find all served children
        for fact in state:
            if match(fact, "served", "*"):
                served_children.add(get_parts(fact)[1])

        # Then, find unserved children and their locations, only considering goal children
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                if child in self.goal_children and child not in served_children:
                    if place not in unserved_children_at_place:
                        unserved_children_at_place[place] = []
                    unserved_children_at_place[place].append(child)

        unserved_children = [c for children_list in unserved_children_at_place.values() for c in children_list]
        n_unserved = len(unserved_children)

        # If no children need serving, the goal is reached.
        if n_unserved == 0:
            return 0

        # --- Step 3: Count locations needing trays ---
        places_with_unserved = set(unserved_children_at_place.keys())
        trays_at_place = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                 # Check if the first argument is a tray (heuristic assumes this structure)
                 # A more robust check would involve parsing types from the task,
                 # but for a domain-dependent heuristic, we rely on domain structure.
                 parts = get_parts(fact)
                 if len(parts) == 3 and parts[1].startswith('tray'): # Simple check for tray type
                     trays_at_place.add(parts[2])

        move_tray_cost = len(places_with_unserved - trays_at_place)

        # --- Step 4: Count needed sandwiches by type ---
        needed_gf_count = sum(1 for c in unserved_children if c in self.allergic_children)
        needed_reg_count = n_unserved - needed_gf_count

        # --- Step 5: Count available sandwiches on trays by type ---
        sandwiches_ontray = set()
        gf_sandwiches_made = set() # Sandwiches with (no_gluten_sandwich s)

        # Find all sandwiches on trays and identify GF ones
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwiches_ontray.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_sandwich", "*"):
                 gf_sandwiches_made.add(get_parts(fact)[1])

        n_gf_avail_ontray = len({s for s in sandwiches_ontray if s in gf_sandwiches_made})
        n_reg_avail_ontray = len(sandwiches_ontray) - n_gf_avail_ontray

        # --- Step 6: Needed sandwiches not yet on tray ---
        # These are the sandwiches needed by unserved children MINUS those already on trays.
        # We only care about the *number* needed, not specific sandwiches.
        n_gf_from_kitchen_or_make = max(0, needed_gf_count - n_gf_avail_ontray)
        n_reg_from_kitchen_or_make = max(0, needed_reg_count - n_reg_avail_ontray)

        # --- Step 7: Count sandwiches in kitchen by type ---
        sandwiches_at_kitchen = set()
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                sandwiches_at_kitchen.add(get_parts(fact)[1])

        n_gf_kitchen = len({s for s in sandwiches_at_kitchen if s in gf_sandwiches_made})
        n_reg_kitchen = len(sandwiches_at_kitchen) - n_gf_kitchen

        # --- Step 8: Calculate cost to get needed sandwiches onto trays ---
        # Cost for needed GF sandwiches:
        # Prioritize using existing kitchen GF sandwiches (cost 1: put_on_tray)
        cost_gf_put_on_tray = min(n_gf_kitchen, n_gf_from_kitchen_or_make)
        # Remaining needed GF sandwiches must be made (cost 2: make + put_on_tray)
        cost_gf_make_and_put = max(0, n_gf_from_kitchen_or_make - n_gf_kitchen) * 2

        # Cost for needed Regular sandwiches:
        # Prioritize using existing kitchen Reg sandwiches (cost 1: put_on_tray)
        cost_reg_put_on_tray = min(n_reg_kitchen, n_reg_from_kitchen_or_make)
        # Remaining needed Reg sandwiches must be made (cost 2: make + put_on_tray)
        cost_reg_make_and_put = max(0, n_reg_from_kitchen_or_make - n_reg_kitchen) * 2

        put_on_tray_and_make_cost = cost_gf_put_on_tray + cost_gf_make_and_put + cost_reg_put_on_tray + cost_reg_make_and_put

        # --- Step 9: Sum all costs ---
        # Total heuristic = Cost for serving + Cost for moving trays + Cost for getting sandwiches onto trays (from kitchen or by making)
        total_heuristic = n_unserved + move_tray_cost + put_on_tray_and_make_cost

        return total_heuristic
