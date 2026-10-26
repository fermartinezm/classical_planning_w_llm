from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assuming valid PDDL fact strings from planner
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
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

    Estimates the number of actions needed to serve all waiting children.
    The heuristic is the sum of:
    1. Number of children not yet served (lower bound on serve actions).
    2. Number of sandwiches that need to be made (lower bound on make actions).
    3. Number of sandwiches that need to be put on trays (lower bound on put_on_tray actions).
    4. Number of sandwich-tray deliveries that need a tray movement to the child's location (lower bound on move_tray actions).

    Suitability of sandwiches for children is considered when counting needed items at each stage,
    prioritizing gluten-free sandwiches for allergic children.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        self.goals = task.goals # Goal conditions (e.g., (served child1))
        self.static_facts = task.static # Facts that are true in all states

        # Extract static properties of children (allergies)
        self.allergic_children_set = {
            get_parts(fact)[1] for fact in self.static_facts if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children_set = {
            get_parts(fact)[1] for fact in self.static_facts if match(fact, "not_allergic_gluten", "*")
        }

        # Extract all child objects from goals (assuming all children in goals are the ones to be served)
        # The goal is typically (and (served child1) (served child2) ...)
        self.all_children_in_goal = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify waiting children and their locations
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        # Waiting children are those in the goal who are not yet served
        waiting_children = {c for c in self.all_children_in_goal if c not in served_children}
        N_waiting = len(waiting_children)

        # If all children are served, the heuristic is 0
        if N_waiting == 0:
            return 0

        # Classify waiting children by allergy
        allergic_waiting = {c for c in waiting_children if c in self.allergic_children_set}
        non_allergic_waiting = {c for c in waiting_children if c in self.not_allergic_children_set}
        N_allergic = len(allergic_waiting)
        N_non_allergic = len(non_allergic_waiting)

        # Get dynamic facts about sandwiches and trays
        at_kitchen_sandwich_set = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        ontray_dict = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "ontray", "*", "*")} # {sandwich: tray}
        at_tray_dict = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith('tray')} # {tray: place}
        no_gluten_sandwich_set = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        
        # Get waiting locations for current waiting children
        waiting_children_loc = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "waiting", "*", "*") and get_parts(fact)[1] in waiting_children}


        # Classify available sandwiches by type and location stage

        # Sandwiches available (kitchen or ontray)
        all_sandwiches_available = at_kitchen_sandwich_set | set(ontray_dict.keys())
        gf_sandwiches_available = {s for s in all_sandwiches_available if s in no_gluten_sandwich_set}
        reg_sandwiches_available = all_sandwiches_available - gf_sandwiches_available

        # Sandwiches on trays
        gf_sandwiches_ontray = {s for s in ontray_dict.keys() if s in no_gluten_sandwich_set}
        reg_sandwiches_ontray = {s for s in ontray_dict.keys() if s not in no_gluten_sandwich_set}

        # Sandwiches on trays that are at a waiting child's location
        gf_sandwiches_ontray_at_loc = set()
        reg_sandwiches_ontray_at_loc = set()

        # Map locations to waiting children at that location
        waiting_children_by_loc = {}
        for child in waiting_children:
            loc = waiting_children_loc.get(child) # Use .get for safety, though child should be in dict
            if loc:
                if loc not in waiting_children_by_loc:
                    waiting_children_by_loc[loc] = []
                waiting_children_by_loc[loc].append(child)

        # Find sandwiches on trays at waiting locations that are suitable for *any* child at that location
        sandwiches_on_trays_at_waiting_loc = set()
        for s, t in ontray_dict.items():
            if t in at_tray_dict:
                tray_loc = at_tray_dict[t]
                if tray_loc in waiting_children_by_loc: # Check if this tray location has waiting children
                    # Check if this sandwich is suitable for any child waiting at tray_loc
                    is_suitable_for_any_child_at_loc = False
                    for child in waiting_children_by_loc[tray_loc]:
                         is_gf_sandwich = s in no_gluten_sandwich_set
                         is_allergic_child = child in self.allergic_children_set

                         if (is_allergic_child and is_gf_sandwich) or (not is_allergic_child):
                             is_suitable_for_any_child_at_loc = True
                             break # Found a child this sandwich is suitable for at this location

                    if is_suitable_for_any_child_at_loc:
                         if s in no_gluten_sandwich_set:
                             gf_sandwiches_ontray_at_loc.add(s)
                         else:
                             reg_sandwiches_ontray_at_loc.add(s)


        # 2. Calculate needed_make actions
        # Count how many children can be satisfied by available sandwiches (kitchen or ontray)
        A_avail_gf = len(gf_sandwiches_available)
        A_avail_reg = len(reg_sandwiches_available)

        served_allergic_by_avail = min(N_allergic, A_avail_gf)
        rem_avail_gf = A_avail_gf - served_allergic_by_avail
        served_non_allergic_by_avail = min(N_non_allergic, rem_avail_gf + A_avail_reg)
        total_satisfied_by_avail = served_allergic_by_avail + served_non_allergic_by_avail

        needed_make = max(0, N_waiting - total_satisfied_by_avail)

        # 3. Calculate needed_put actions
        # Count how many children can be satisfied by sandwiches already on trays
        A_ontray_gf = len(gf_sandwiches_ontray)
        A_ontray_reg = len(reg_sandwiches_ontray)

        served_allergic_by_ontray = min(N_allergic, A_ontray_gf)
        rem_ontray_gf = A_ontray_gf - served_allergic_by_ontray
        served_non_allergic_by_ontray = min(N_non_allergic, rem_ontray_gf + A_ontray_reg)
        total_satisfied_by_ontray = served_allergic_by_ontray + served_non_allergic_by_ontray

        needed_put = max(0, N_waiting - total_satisfied_by_ontray)

        # 4. Calculate needed_move actions
        # Count how many children can be satisfied by sandwiches on trays at their location
        A_ontray_at_loc_gf = len(gf_sandwiches_ontray_at_loc)
        A_ontray_at_loc_reg = len(reg_sandwiches_ontray_at_loc)

        served_allergic_by_ontray_at_loc = min(N_allergic, A_ontray_at_loc_gf)
        rem_ontray_at_loc_gf = A_ontray_at_loc_gf - served_allergic_by_ontray_at_loc
        served_non_allergic_by_ontray_at_loc = min(N_non_allergic, rem_ontray_at_loc_gf + A_ontray_at_loc_reg)
        total_satisfied_by_ontray_at_loc = served_allergic_by_ontray_at_loc + served_non_allergic_by_ontray_at_loc

        needed_move = max(0, N_waiting - total_satisfied_by_ontray_at_loc)

        # Total heuristic is the sum of required actions for each stage + the final serve action
        # The N_waiting term covers the final serve action for each child.
        heuristic_value = N_waiting + needed_make + needed_put + needed_move

        return heuristic_value
