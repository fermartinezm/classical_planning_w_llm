from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    This heuristic estimates the number of actions needed to serve all waiting children.
    It uses an additive approach, counting the number of children not yet served
    and adding costs based on how many necessary sandwiches (total and gluten-free)
    are "behind schedule" in the preparation and delivery pipeline (not made,
    in kitchen, on tray elsewhere, on tray at the child's location).

    The stages considered for a sandwich needed by a waiting child are:
    1. Not yet made.
    2. Made, in the kitchen.
    3. On a tray, tray in the kitchen.
    4. On a tray, tray at a location where a child is waiting.
    5. Served to the child.

    The heuristic sums the number of items (representing required servings) that
    need to transition through stages 1->2 (make), 2->3 (put on tray), and 3->4 (move tray),
    plus the final 4->5 (serve) action for each child. It does this for the total
    number of sandwiches needed and additionally for the number of gluten-free
    sandwiches specifically required by allergic children.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children, allergic status,
        and waiting locations from the task's static information and goals.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal children (those who need to be served)
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "served":
                self.goal_children.add(parts[1])

        # Extract allergic children and waiting locations from static facts
        self.allergic_children = set()
        self.waiting_locations = {} # {child: place}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif parts[0] == "waiting":
                child, place = parts[1], parts[2]
                self.waiting_locations[child] = place

        # Create a set of all places where children are waiting
        self.waiting_places_set = set(self.waiting_locations.values())

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions from the current state.
        """
        state = node.state

        # 1. Count children needing service
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        waiting_children_list = [c for c in self.goal_children if c not in served_children]
        N_waiting_total = len(waiting_children_list)

        # If no children are waiting, the goal is reached.
        if N_waiting_total == 0:
            return 0

        # Count allergic children who are still waiting
        N_allergic_waiting_total = sum(1 for c in waiting_children_list if c in self.allergic_children)

        # 2. Count sandwiches by type and current stage/location
        sandwiches_on_trays = {} # {sandwich: tray}
        tray_locations = {} # {tray: place}
        kitchen_sandwiches = set() # {sandwich}
        gf_sandwiches = set() # {sandwich}
        # num_notexist = 0 # Not directly used in this heuristic calculation

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ontray":
                s, t = parts[1], parts[2]
                sandwiches_on_trays[s] = t
            elif parts[0] == "at":
                t, p = parts[1], parts[2]
                tray_locations[t] = p
            elif parts[0] == "at_kitchen_sandwich":
                kitchen_sandwiches.add(parts[1])
            elif parts[0] == "no_gluten_sandwich":
                gf_sandwiches.add(parts[1])
            # elif parts[0] == "notexist":
            #     num_notexist += 1

        # Count sandwiches at different stages
        N_any_ontray_total = len(sandwiches_on_trays)
        N_gf_ontray_total = sum(1 for s in sandwiches_on_trays if s in gf_sandwiches)

        N_any_kitchen = len(kitchen_sandwiches)
        N_gf_kitchen = sum(1 for s in kitchen_sandwiches if s in gf_sandwiches)

        N_any_at_waiting_loc = 0
        N_gf_at_waiting_loc = 0

        for s, t in sandwiches_on_trays.items():
            if t in tray_locations and tray_locations[t] in self.waiting_places_set:
                 N_any_at_waiting_loc += 1
                 if s in gf_sandwiches:
                     N_gf_at_waiting_loc += 1

        # Total made sandwiches (kitchen + on trays)
        N_any_made_total = N_any_kitchen + N_any_ontray_total
        N_gf_made_total = N_gf_kitchen + N_gf_ontray_total

        # 3. Calculate heuristic based on deficits at each stage, prioritizing GF needs
        # The heuristic is the sum of the number of items (servings) that still need
        # to pass through each stage of the pipeline.

        # Cost Layer 1: Serve action
        # Each waiting child needs a serve action.
        h = N_waiting_total

        # Cost Layer 2: Move tray to location
        # We need N_waiting_total sandwiches to reach the 'at waiting location' stage.
        # We have N_any_at_waiting_loc already there.
        # The deficit needs at least one 'move_tray' action per sandwich.
        h += max(0, N_waiting_total - N_any_at_waiting_loc)

        # We specifically need N_allergic_waiting_total GF sandwiches to reach the 'at waiting location' stage.
        # We have N_gf_at_waiting_loc already there.
        # The GF deficit needs at least one 'move_tray' action per sandwich.
        h += max(0, N_allergic_waiting_total - N_gf_at_waiting_loc)


        # Cost Layer 3: Put on tray
        # We need N_waiting_total sandwiches to reach the 'on tray' stage (anywhere).
        # We have N_any_ontray_total already on trays.
        # The deficit needs at least one 'put_on_tray' action per sandwich.
        h += max(0, N_waiting_total - N_any_ontray_total)

        # We specifically need N_allergic_waiting_total GF sandwiches to reach the 'on tray' stage (anywhere).
        # We have N_gf_ontray_total already on trays.
        # The GF deficit needs at least one 'put_on_tray' action per sandwich.
        h += max(0, N_allergic_waiting_total - N_gf_ontray_total)


        # Cost Layer 4: Make sandwich
        # We need N_waiting_total sandwiches to reach the 'made' stage.
        # We have N_any_made_total already made.
        # The deficit needs at least one 'make_sandwich' action per sandwich.
        h += max(0, N_waiting_total - N_any_made_total)

        # We specifically need N_allergic_waiting_total GF sandwiches to reach the 'made' stage.
        # We have N_gf_made_total already made.
        # The GF deficit needs at least one 'make_sandwich_no_gluten' action per sandwich.
        h += max(0, N_allergic_waiting_total - N_gf_made_total)

        return h
