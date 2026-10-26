from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assuming valid PDDL fact strings like "(predicate arg1 arg2)"
    if not fact or fact[0] != '(' or fact[-1] != ')':
         # Handle unexpected format, though standard PDDL facts should fit
         return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts in the fact must match the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all unserved children.
    The heuristic counts the "stages" a sandwich needs to go through for each
    unserved child, from not-yet-made to served. It prioritizes using existing
    sandwiches that are closer to being served (e.g., already on a tray at the
    correct location).

    Stages (actions needed to reach 'served' from this stage):
    4: Not yet made (make + put + move + serve)
    3: Made, in kitchen (put + move + serve)
    2: On tray, anywhere (move + serve)
    1: On tray, at child's location (serve)
    0: Served (goal reached for this child)

    The heuristic sums the minimum actions needed for each unserved child,
    greedily assigning available sandwiches to children.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        self.goals = task.goals # Goal conditions (served children)
        self.static_facts = task.static # Static facts (allergies, waiting places)

        # Map child to waiting place
        self.child_waiting_place = {}
        # Map child to allergy status
        self.child_allergy = {} # True for allergic, False for not_allergic

        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                self.child_waiting_place[child] = place
            elif parts[0] == 'allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = True
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = False

        # Identify all children that need to be served (from goals)
        self.children_to_serve = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'served'}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Identify children already served in the current state
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Identify children who still need to be served
        unserved_children = self.children_to_serve - served_children_in_state

        # If all children are served, the heuristic is 0
        if not unserved_children:
            return 0

        heuristic_cost = 0

        # Keep track of sandwiches we assume will be used to satisfy a child's need
        used_sandwiches = set()

        # --- Collect available sandwiches in different stages ---

        # Sandwiches currently on trays and their locations/types
        # {sandwich: (tray, location, is_gf)}
        sandwiches_on_trays = {}
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwich, tray = get_parts(fact)[1], get_parts(fact)[2]
                # Find tray location
                tray_location = None
                for loc_fact in state:
                    if match(loc_fact, "at", tray, "*"):
                        tray_location = get_parts(loc_fact)[2]
                        break
                # Only consider sandwiches on trays that are located somewhere
                if tray_location:
                    # Check if the sandwich is gluten-free
                    is_gf = "(no_gluten_sandwich {})".format(sandwich) in state
                    sandwiches_on_trays[sandwich] = (tray, tray_location, is_gf)

        # Sandwiches currently in the kitchen and their types
        # {sandwich: is_gf}
        sandwiches_in_kitchen = {}
        for fact in state:
             if match(fact, "at_kitchen_sandwich", "*"):
                sandwich = get_parts(fact)[1]
                # Check if the sandwich is gluten-free
                is_gf = "(no_gluten_sandwich {})".format(sandwich) in state
                sandwiches_in_kitchen[sandwich] = is_gf

        # --- Estimate cost for each unserved child ---
        # Process children to assign available sandwiches greedily

        for child in unserved_children:
            child_place = self.child_waiting_place[child]
            needs_gluten_free = self.child_allergy[child]

            # Base cost is 1 for the final 'serve' action
            cost_for_child = 1

            # Check if a suitable unused sandwich is available at the child's location (Stage 1: serve)
            found_stage_1 = False
            for sandwich, (tray, location, is_gf) in sandwiches_on_trays.items():
                if sandwich not in used_sandwiches and location == child_place and is_gf == needs_gluten_free:
                    used_sandwiches.add(sandwich)
                    found_stage_1 = True
                    break

            if found_stage_1:
                heuristic_cost += cost_for_child # Only serve action needed
                continue

            # If not found at location, need to move a tray (or use one already there) (Cost +1)
            cost_for_child += 1

            # Check if a suitable unused sandwich is available on any tray (Stage 2: move + serve)
            found_stage_2 = False
            for sandwich, (tray, location, is_gf) in sandwiches_on_trays.items():
                 if sandwich not in used_sandwiches and is_gf == needs_gluten_free:
                    used_sandwiches.add(sandwich)
                    found_stage_2 = True
                    break

            if found_stage_2:
                heuristic_cost += cost_for_child # Serve + Move actions needed
                continue

            # If not found on any tray, need to put one on a tray (Cost +1)
            cost_for_child += 1

            # Check if a suitable unused sandwich is available in the kitchen (Stage 3: put + move + serve)
            found_stage_3 = False
            for sandwich, is_gf in sandwiches_in_kitchen.items():
                if sandwich not in used_sandwiches and is_gf == needs_gluten_free:
                    used_sandwiches.add(sandwich)
                    found_stage_3 = True
                    break

            if found_stage_3:
                heuristic_cost += cost_for_child # Serve + Move + Put actions needed
                continue

            # If not found in kitchen, need to make one (Cost +1) (Stage 4: make + put + move + serve)
            cost_for_child += 1
            # We don't check if resources (bread, content, notexist) are available to make it.
            # This is an optimistic estimate assuming resources exist for the required make action.

            heuristic_cost += cost_for_child # Serve + Move + Put + Make actions needed

        return heuristic_cost
