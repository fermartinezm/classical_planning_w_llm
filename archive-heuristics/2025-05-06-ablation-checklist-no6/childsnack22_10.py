from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential non-string facts if necessary, though PDDL facts are strings
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
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

    Estimates the number of actions needed to serve all children.
    Calculates the minimum steps required for each unserved child independently
    and sums these minimums.

    Steps per child (if not served, waiting at place P):
    - If P is KITCHEN:
        - Suitable sandwich on tray at KITCHEN: 1 (serve)
        - Suitable sandwich on tray elsewhere: 2 (move_tray + serve)
        - Suitable sandwich in kitchen: 2 (put_on_tray + serve)
        - No suitable sandwich exists: 3 (make + put + serve)
    - If P is NOT KITCHEN:
        - Suitable sandwich on tray at P: 1 (serve)
        - Suitable sandwich on tray elsewhere: 2 (move_tray + serve)
        - Suitable sandwich in kitchen: 3 (put_on_tray + move_tray + serve)
        - No suitable sandwich exists: 4 (make + put + move + serve)

    - Cannot make suitable sandwich (missing ingredients): Large penalty (e.g., 1000)
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract goal children
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'served':
                self.goal_children.add(parts[1])

        # Extract static information about children (allergy, waiting place)
        self.child_info = {} # {child_name: {'allergic': bool, 'waiting_place': place_name}}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            if parts[0] == 'allergic_gluten':
                child_name = parts[1]
                if child_name not in self.child_info:
                    self.child_info[child_name] = {}
                self.child_info[child_name]['allergic'] = True
            elif parts[0] == 'not_allergic_gluten':
                 child_name = parts[1]
                 if child_name not in self.child_info:
                    self.child_info[child_name] = {}
                 self.child_info[child_name]['allergic'] = False
            elif parts[0] == 'waiting':
                child_name = parts[1]
                place_name = parts[2]
                if child_name not in self.child_info:
                    self.child_info[child_name] = {}
                self.child_info[child_name]['waiting_place'] = place_name

        # Extract static information about gluten-free ingredients
        self.gf_bread_static = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "no_gluten_bread", "*")}
        self.gf_content_static = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "no_gluten_content", "*")}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all children in the goal.
        """
        state = node.state
        total_heuristic_cost = 0

        # Pre-calculate ingredient availability in the current state
        bread_in_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        content_in_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}

        any_bread_in_kitchen = len(bread_in_kitchen) > 0
        any_content_in_kitchen = len(content_in_kitchen) > 0
        any_gf_bread_in_kitchen = len(bread_in_kitchen.intersection(self.gf_bread_static)) > 0
        any_gf_content_in_kitchen = len(content_in_kitchen.intersection(self.gf_content_static)) > 0


        # Find all existing sandwiches and their properties/locations
        existing_sandwiches = {} # {sandwich_name: {'is_gf': bool, 'location': 'kitchen' or tray_name}}
        sandwiches_on_trays = {} # {sandwich_name: tray_name}
        trays_at_places = {} # {tray_name: place_name}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at_kitchen_sandwich':
                s_name = parts[1]
                existing_sandwiches[s_name] = {'location': 'kitchen'}
            elif parts[0] == 'ontray':
                s_name = parts[1]
                t_name = parts[2]
                existing_sandwiches[s_name] = {'location': t_name}
                sandwiches_on_trays[s_name] = t_name
            elif parts[0] == 'at' and parts[1].startswith('tray'): # Assuming anything starting with 'tray' is a tray object
                 t_name = parts[1]
                 p_name = parts[2]
                 trays_at_places[t_name] = p_name

        # Add gluten-free property to existing sandwiches
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'no_gluten_sandwich':
                 s_name = parts[1]
                 if s_name in existing_sandwiches:
                     existing_sandwiches[s_name]['is_gf'] = True

        # Assume non-gluten-free if property is missing (sandwiches made with make_sandwich)
        for s_name in existing_sandwiches:
            if 'is_gf' not in existing_sandwiches[s_name]:
                existing_sandwiches[s_name]['is_gf'] = False

        # Iterate through each child that needs to be served
        for child_name in self.goal_children:
            # Check if child is already served
            if f'(served {child_name})' in state:
                continue # This child is served, cost is 0 for this child

            # Child is not served, calculate cost for this child
            child_cost = 0

            # Get child info
            info = self.child_info.get(child_name)
            if not info:
                 # Should not happen in valid problems where goal children are waiting
                 # print(f"Warning: Missing info for child {child_name}")
                 total_heuristic_cost += 1000 # Penalty for child that cannot be served
                 continue

            is_allergic = info.get('allergic', False)
            waiting_place = info.get('waiting_place', 'kitchen') # Default to kitchen if info missing


            # Determine the minimum stage cost for this child
            # Initialize with a high cost assuming needs making
            # Base cost for "needs make" stage depends on waiting place
            if waiting_place == 'kitchen':
                 min_stage_cost = 3 # make + put + serve
            else:
                 min_stage_cost = 4 # make + put + move + serve


            # Check existing sandwiches to find the best stage
            suitable_sandwich_found_at_any_stage = False
            for s_name, s_info in existing_sandwiches.items():
                # Check if sandwich is suitable for the child
                if is_allergic and not s_info['is_gf']:
                    continue # Allergic child needs GF, this is not GF

                # Sandwich is suitable. Check its location/stage.
                suitable_sandwich_found_at_any_stage = True
                s_loc = s_info['location']

                if s_loc != 'kitchen': # Sandwich is on a tray
                    t_name = s_loc # Tray name is the location value
                    t_place = trays_at_places.get(t_name)

                    if t_place == waiting_place:
                        # Suitable sandwich on tray at child's location
                        min_stage_cost = min(min_stage_cost, 1) # Serve is 1 action
                    elif t_place is not None:
                         # Suitable sandwich on tray elsewhere
                         # Cost is 2 (move + serve) regardless of waiting_place
                         min_stage_cost = min(min_stage_cost, 2)
                    # else: tray location unknown? Assume valid states.
                else: # Sandwich is in the kitchen
                    # Suitable sandwich in the kitchen
                    if waiting_place == 'kitchen':
                        # Cost is 2 (put + serve)
                        min_stage_cost = min(min_stage_cost, 2)
                    else:
                        # Cost is 3 (put + move + serve)
                        min_stage_cost = min(min_stage_cost, 3)


            # If no suitable sandwich exists at all (neither in kitchen nor on tray),
            # check if one can be made.
            if not suitable_sandwich_found_at_any_stage:
                 can_make_suitable = False
                 if is_allergic:
                     if any_gf_bread_in_kitchen and any_gf_content_in_kitchen:
                         can_make_suitable = True
                 else: # Not allergic
                     if any_bread_in_kitchen and any_content_in_kitchen:
                         can_make_suitable = True

                 if can_make_suitable:
                     # min_stage_cost is already initialized to the correct "needs make" cost
                     pass
                 else:
                     # Cannot make the required sandwich type with available ingredients.
                     # This child is stuck. Add a large penalty.
                     min_stage_cost = 1000 # Use a large fixed penalty

            # Add the determined minimum cost for this child
            total_heuristic_cost += min_stage_cost

        return total_heuristic_cost
