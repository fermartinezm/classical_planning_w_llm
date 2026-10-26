from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a string like '(predicate arg1 arg2)'
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children, the estimated cost to make and
    prepare necessary sandwiches that are not already on trays, and the estimated
    cost to move trays to locations where unserved children are waiting.

    # Assumptions
    - Sufficient bread and content portions are available in the kitchen to make
      any required new sandwiches.
    - Sufficient 'notexist' sandwich objects are available to represent new sandwiches.
    - Trays have infinite capacity for sandwiches.
    - An available tray can always be moved to a required location if one is needed there.
    - The cost of making a sandwich is 1 action.
    - The cost of putting a sandwich on a tray is 1 action.
    - The cost of moving a tray between any two places is 1 action.
    - The cost of serving a child is 1 action.

    # Heuristic Initialization
    The heuristic extracts the set of all children that need to be served (from the goal),
    and identifies which children are allergic to gluten based on static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children that need to be served based on the goal state.
    2. From the current state, determine which of these children are not yet served.
    3. Count the total number of unserved children (`N_unserved`). If `N_unserved` is 0, the goal is reached, and the heuristic value is 0.
    4. Separate the unserved children into those who are allergic to gluten (`N_allergic_unserved`) and those who are not (`N_non_allergic_unserved`), using the static information.
    5. Count the number of gluten-free sandwiches currently on trays (`S_gf_ontray`) and the number of regular sandwiches currently on trays (`S_reg_ontray`).
    6. Calculate the number of *new* sandwiches that need to be made and put on trays:
       - First, determine how many new gluten-free sandwiches are needed for allergic children, considering the available `S_gf_ontray`.
       - Then, determine how many *additional* sandwiches (can be regular or surplus GF) are needed for non-allergic children, considering available `S_reg_ontray` and any surplus `S_gf_ontray` not used by allergic children.
       - The total number of new sandwiches needed (`N_new_sandwiches_needed`) is the sum of these two counts.
    7. Estimate the cost for making and putting these new sandwiches on trays as `N_new_sandwiches_needed * 2` (1 for make, 1 for put_on_tray).
    8. Identify all distinct places where unserved children are waiting.
    9. Identify all distinct places where trays are currently located.
    10. Count the number of places identified in step 8 that do *not* have a tray present according to step 9. This is the estimated number of tray movements needed (`cost_move_tray`).
    11. The total heuristic value is the sum of:
        - The number of unserved children (`N_unserved`), representing the minimum number of `serve` actions.
        - The estimated cost for preparing needed sandwiches.
        - The estimated cost for moving trays.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children and static facts
        about allergies.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract children that need to be served from the goal facts.
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Extract allergic and non-allergic children from static facts.
        self.allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")}
        self.not_allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1-3. Identify unserved children and count them.
        unserved_children = {c for c in self.goal_children if "(served {})".format(c) not in state}
        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0  # Goal reached

        # 4. Separate unserved children by allergy status.
        N_allergic_unserved = sum(1 for c in unserved_children if c in self.allergic_children)
        N_non_allergic_unserved = N_unserved - N_allergic_unserved # Assumes all children are either allergic or not_allergic

        # 5. Count available sandwiches on trays.
        S_gf_ontray = 0
        S_reg_ontray = 0
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "ontray", "*", "*"):
                 sandwich = parts[1]
                 if "(no_gluten_sandwich {})".format(sandwich) in state:
                     S_gf_ontray += 1
                 else:
                     S_reg_ontray += 1

        # 6-7. Calculate cost for making and putting new sandwiches on trays.
        needed_gf = N_allergic_unserved
        needed_any = N_non_allergic_unserved

        # How many GF sandwiches from trays are used by allergic children?
        used_gf_ontray_for_allergic = min(needed_gf, S_gf_ontray)
        # How many GF sandwiches from trays remain?
        remaining_S_gf_ontray = S_gf_ontray - used_gf_ontray_for_allergic

        # How many new GF sandwiches must be made?
        new_gf_to_make = needed_gf - used_gf_ontray_for_allergic # This is max(0, needed_gf - S_gf_ontray)

        # How many 'any' sandwiches (regular or surplus GF) are needed for non-allergic children?
        # These can be satisfied by remaining GF on trays or regular on trays.
        available_for_non_allergic = remaining_S_gf_ontray + S_reg_ontray
        new_any_to_make = max(0, needed_any - available_for_non_allergic)

        N_new_sandwiches_needed = new_gf_to_make + new_any_to_make
        cost_make_put = N_new_sandwiches_needed * 2 # 1 for make, 1 for put_on_tray

        # 8-10. Calculate cost for tray movements.
        places_with_waiting_children = set()
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "waiting", "*", "*"):
                child, place = parts[1], parts[2]
                if child in unserved_children:
                    places_with_waiting_children.add(place)

        places_with_trays = set()
        for fact in state:
             parts = get_parts(fact)
             # Assuming any fact starting with 'at' and having a tray object as the first argument
             # indicates a tray location. Tray objects start with 'tray'.
             if match(fact, "at", "tray*", "*"):
                  place = parts[2]
                  places_with_trays.add(place)

        cost_move_tray = sum(1 for place in places_with_waiting_children if place not in places_with_trays)

        # 11. Total heuristic value.
        # Base cost is 1 per unserved child (the serve action itself).
        # Add cost for preparing needed sandwiches.
        # Add cost for moving trays to needed locations.
        total_heuristic = N_unserved + cost_make_put + cost_move_tray

        return total_heuristic
