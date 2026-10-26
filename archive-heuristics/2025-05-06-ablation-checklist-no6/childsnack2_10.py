from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    parts = fact[1:-1].split()
    return parts

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

# Heuristic class
class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all waiting children.
    The heuristic counts the sum of:
    1. Sandwiches that need to be made.
    2. Sandwiches that need to be put on trays (those in kitchen or newly made).
    3. Locations where children are waiting but no tray is present.
    4. Children who are waiting to be served.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts about children's allergies
        and identifying the set of children that need to be served according to the goal.
        """
        self.goals = task.goals # Goal conditions

        self.allergic_children = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                child = parts[1]
                self.allergic_children.add(child)

        # Get the set of children that are part of the goal
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'served'}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state # Current world state.

        # 1. Identify served children (only considering those in the goal)
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        served_goal_children = self.goal_children.intersection(served_children_in_state)

        # If all goal children are served, goal reached.
        if len(served_goal_children) == len(self.goal_children):
             return 0 # Goal reached

        # 2. Identify waiting goal children and their locations
        waiting_goal_children_info = {} # child -> place
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                if child in self.goal_children and child not in served_children_in_state:
                    waiting_goal_children_info[child] = place

        N_wait = len(waiting_goal_children_info)

        # 3. Count needed sandwiches by type for waiting goal children
        needed_gf = 0
        needed_reg = 0
        for child in waiting_goal_children_info:
            if child in self.allergic_children:
                needed_gf += 1
            else: # Assume not_allergic_gluten if not allergic_gluten
                needed_reg += 1

        # 4. Count available sandwiches by type (in kitchen or on trays)
        at_kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        ontray_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}
        available_sandwiches = at_kitchen_sandwiches | ontray_sandwiches

        no_gluten_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        available_gf_count = sum(1 for s in available_sandwiches if s in no_gluten_sandwiches_in_state)
        # Sandwiches without the no_gluten predicate are regular
        available_reg_count = sum(1 for s in available_sandwiches if s not in no_gluten_sandwiches_in_state)

        # 5. Sandwiches that must be made to meet the demand of waiting goal children
        N_make_gf = max(0, needed_gf - available_gf_count)
        N_make_reg = max(0, needed_reg - available_reg_count)

        # 6. Sandwiches currently in the kitchen that need to be put on trays
        N_kitchen_sandwiches = len(at_kitchen_sandwiches)

        # 7. Places needing a tray move for waiting goal children
        waiting_places = set(waiting_goal_children_info.values())
        # Find all locations where trays are currently present
        tray_locations = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray")}
        places_needing_tray_move_count = len(waiting_places - tray_locations)

        # 8. Calculate heuristic based on the sum of required steps/items
        # h = (make_actions) + (put_actions) + (move_actions) + (serve_actions)
        # make_actions = N_make_gf + N_make_reg
        # put_actions = N_kitchen_sandwiches + N_make_gf + N_make_reg # Sandwiches in kitchen + newly made sandwiches
        # move_actions = places_needing_tray_move_count
        # serve_actions = N_wait

        # Summing these gives:
        # h = (N_make_gf + N_make_reg) + (N_kitchen_sandwiches + N_make_gf + N_make_reg) + places_needing_tray_move_count + N_wait

        heuristic_value = (N_make_gf + N_make_reg) + \
                          (N_kitchen_sandwiches + N_make_gf + N_make_reg) + \
                          places_needing_tray_move_count + \
                          N_wait

        return heuristic_value
