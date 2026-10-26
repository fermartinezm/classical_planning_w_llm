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
    # Ensure the number of parts matches the number of args if args are not wildcards
    # This simple match works for fixed-arity predicates
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It calculates the deficit of suitable sandwiches at different stages of the process:
    made, on trays, and delivered to the child's location. The total heuristic is the
    sum of the estimated actions for making, putting on trays, moving trays, and serving.

    # Assumptions
    - Each unserved child needs one suitable sandwich (gluten-free for allergic, any for non-allergic).
    - The heuristic assumes sufficient bread, content, and 'notexist' sandwich objects exist
      to make any required sandwiches.
    - The heuristic assumes sufficient trays exist and can be moved to the kitchen for
      'put_on_tray' actions when needed.
    - The cost for 'move_tray' is estimated by counting the deficit of trays at locations
      where unserved children are waiting, assuming one tray is needed per child at that location.
      This is a simplification; one tray can carry multiple sandwiches. The calculation
      `max(0, N_children_at_loc - N_trays_at_loc)` summed over locations estimates the
      number of tray movements needed to bring enough trays to the children's locations.

    # Heuristic Initialization
    - Extract the set of children that need to be served (from task goals).
    - Extract the set of children who are allergic to gluten (from static facts and initial state).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost `h = 0`.
    2. Identify all unserved children (`U`) who are currently waiting, distinguishing between
       allergic (`U_allergic`) and non-allergic (`U_non_allergic`). Record the current
       waiting location for each unserved child.
    3. Count the number of suitable sandwiches that are currently made (either
       `at_kitchen_sandwich` or `ontray`):
       - `N_made_gf`: Count GF sandwiches that are made.
       - `N_made_regular`: Count regular sandwiches that are made.
    4. Count the number of suitable sandwiches that are currently on trays (`ontray`):
       - `N_ontray_gf`: Count GF sandwiches that are on trays.
       - `N_ontray_regular`: Count regular sandwiches that are on trays.
    5. Count the number of trays currently at each location.
    6. Add cost for the final `serve` action: `h += |U|`.
    7. Add cost for `make_sandwich` actions: Estimate the number of suitable sandwiches
       that still need to be made. This is the total number needed minus those already made.
       `h += max(0, len(unserved_allergic) - N_made_gf) + max(0, len(unserved_non_allergic) - N_made_regular)`.
    8. Add cost for `put_on_tray` actions: Estimate the number of suitable sandwiches
       that need to be moved from the 'made' state (kitchen or newly made) onto a tray.
       This is the total number needed on trays minus those already on trays.
       `h += max(0, len(unserved_allergic) - N_ontray_gf) + max(0, len(unserved_non_allergic) - N_ontray_regular)`.
    9. Add cost for `move_tray` actions: For each location where unserved children
       are waiting, estimate how many additional trays are needed there. Sum
       `max(0, N_children_at_loc - N_trays_at_loc)` over all such locations.
       `h += sum(max(0, N_children_at_loc - N_trays_at_loc))` for relevant locations.
    10. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal children (those who need to be served).
        - Static facts about gluten allergies.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Identify children who need to be served from the goals.
        self.goal_children = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "served", "*")
        }

        # Identify which children are allergic to gluten from static facts and initial state
        # (allergy status is static)
        self.allergic_children = set()
        # Look in both static and initial state for allergy facts
        for fact in static_facts.union(initial_state):
             if match(fact, "allergic_gluten", "*"):
                 self.allergic_children.add(get_parts(fact)[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # --- Step 2: Identify unserved children and their locations ---
        unserved_children = set()
        unserved_allergic = set()
        unserved_non_allergic = set()
        child_current_location = {} # Map child -> location

        # Find all children currently waiting and their locations
        waiting_children_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "waiting" and len(parts) == 3 and parts[1].startswith("child"):
                child, place = parts[1], parts[2]
                waiting_children_in_state.add(child)
                child_current_location[child] = place

        # Identify which waiting children are unserved goals
        for child in waiting_children_in_state:
             if child in self.goal_children and f"(served {child})" not in state:
                 unserved_children.add(child)
                 if child in self.allergic_children:
                     unserved_allergic.add(child)
                 else:
                     unserved_non_allergic.add(child)

        # If no children need serving or all are served, heuristic is 0
        if not unserved_children:
            return 0

        # --- Step 3, 4, 5: Count available sandwiches and trays ---
        trays_at_location = {}
        sandwiches_ontray = set()
        sandwiches_kitchen = set()
        sandwich_is_gf = {} # Map sandwich name to boolean (is_gluten_free)

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3 and parts[1].startswith("tray"):
                tray, place = parts[1], parts[2]
                trays_at_location.setdefault(place, set()).add(tray)
            elif parts[0] == "ontray" and len(parts) == 3 and parts[1].startswith("sandw"):
                sandwich, tray = parts[1], parts[2]
                sandwiches_ontray.add((sandwich, tray))
            elif parts[0] == "at_kitchen_sandwich" and len(parts) == 2 and parts[1].startswith("sandw"):
                sandwich = parts[1]
                sandwiches_kitchen.add(sandwich)
            elif parts[0] == "no_gluten_sandwich" and len(parts) == 2 and parts[1].startswith("sandw"):
                sandwich = parts[1]
                sandwich_is_gf[sandwich] = True

        # Count made sandwiches (kitchen or ontray)
        made_sandwiches = sandwiches_kitchen.union({s for s, t in sandwiches_ontray})
        N_made_gf = sum(1 for s in made_sandwiches if sandwich_is_gf.get(s, False))
        N_made_regular = len(made_sandwiches) - N_made_gf # Assume non-GF if not explicitly marked GF

        # Count ontray sandwiches
        N_ontray_gf = sum(1 for s, t in sandwiches_ontray if sandwich_is_gf.get(s, False))
        N_ontray_regular = len(sandwiches_ontray) - N_ontray_gf

        # --- Step 6: Cost for serving ---
        h = len(unserved_children)

        # --- Step 7: Cost for making sandwiches ---
        needed_make_gf = max(0, len(unserved_allergic) - N_made_gf)
        needed_make_regular = max(0, len(unserved_non_allergic) - N_made_regular)
        h += needed_make_gf + needed_make_regular

        # --- Step 8: Cost for putting on trays ---
        # Number of suitable sandwiches that need to transition from 'made' to 'ontray'.
        # This is the total number needed on trays minus those already on trays.
        needed_ontray_gf = max(0, len(unserved_allergic) - N_ontray_gf)
        needed_ontray_regular = max(0, len(unserved_non_allergic) - N_ontray_regular)
        h += needed_ontray_gf + needed_ontray_regular

        # --- Step 9: Cost for moving trays ---
        locations_with_unserved = set(child_current_location[c] for c in unserved_children)
        move_cost = 0
        for loc in locations_with_unserved:
            N_children_at_loc = sum(1 for c in unserved_children if child_current_location[c] == loc)
            N_trays_at_loc = len(trays_at_location.get(loc, set()))
            # Estimate: need one tray per child at a location, minus those already there.
            # This is a simplification; one tray can carry multiple sandwiches.
            # A better estimate might be ceil(N_children_at_loc / capacity) - N_trays_at_loc
            # but capacity is not specified. Let's stick to 1 child per tray deficit.
            move_cost += max(0, N_children_at_loc - N_trays_at_loc)

        h += move_cost

        return h
