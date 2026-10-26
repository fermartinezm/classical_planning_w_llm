from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at child1 table1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check element-wise match using fnmatch, stopping at the end of the shorter sequence
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It calculates the minimum number of steps needed for each unserved child
    independently, based on the current state of available sandwiches and trays,
    and sums these minimum costs.

    # Assumptions
    - Each child requires exactly one sandwich.
    - Gluten-allergic children require gluten-free sandwiches.
    - Any available suitable sandwich/tray can be conceptually used for any child
      needing it in this relaxed calculation (resource contention is simplified).
    - Ingredients and sandwich slots are implicitly available if needed (simplified).
    - Trays can be moved to the kitchen if needed for putting sandwiches on (simplified).

    # Heuristic Initialization
    - Identify the set of children who are initially waiting, as these are the ones
      that need to be served according to the goal structure.
    - Identify all trays and places mentioned in the initial state, goals, or static facts.
    - Identify which children are allergic to gluten (static fact).

    # Step-By-Step Thinking for Computing Heuristic
    For each child that is currently waiting but not served:
    1. Determine if the child is allergic to gluten.
    2. Identify the place where the child is waiting.
    3. Estimate the minimum number of actions to get a suitable sandwich served to this child:
       - Cost 1: If a suitable sandwich is already on a tray at the child's location. (Serve action)
       - Cost 2: If a suitable sandwich is on a tray but not at the child's location. (Move tray + Serve actions)
       - Cost 3: If a suitable sandwich is in the kitchen. (Put on tray + Move tray + Serve actions)
       - Cost 4: If no suitable sandwich exists anywhere. (Make sandwich + Put on tray + Move tray + Serve actions)
    4. The heuristic value is the sum of these minimum costs for all unserved children.
    5. If all children who were initially waiting are now served (goal state), the heuristic is 0.

    The heuristic prioritizes finding an existing sandwich/tray closer to the child's
    needs (location and type) to determine the minimum cost for that child.
    Resource availability (like multiple children wanting the same sandwich/tray,
    or limited ingredients/trays) is simplified; the heuristic assumes that if
    a resource type exists that allows a lower-cost stage, that stage is reachable
    for the child in this relaxed calculation.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and relevant objects.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Identify children who are initially waiting (these are the ones in the goal)
        initial_waiting_children = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "waiting", "*", "*")}
        self.all_children_to_serve = initial_waiting_children # These are the children that will appear in the goal

        # Identify all trays and places mentioned in initial state or goals or static facts
        all_objects_in_facts = set()
        for fact in task.initial_state | task.goals | self.static_facts:
             parts = get_parts(fact)
             # Add all arguments as potential objects, skipping the predicate name
             all_objects_in_facts.update(parts[1:])

        self.all_trays = {obj for obj in all_objects_in_facts if obj.startswith('tray')}
        # Assuming places are objects starting with 'table' or the constant 'kitchen'
        self.all_places = {obj for obj in all_objects_in_facts if obj.startswith('table') or obj == 'kitchen'}


        # Identify allergic children (static fact)
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static_facts if match(fact, "allergic_gluten", "*")
        }

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Check if goal is reached (all children who were initially waiting are served)
        # The goal is typically a conjunction of (served child) for all initial waiting children.
        # We check if all children we identified as needing service are indeed served.
        goal_reached = True
        for child in self.all_children_to_serve:
            if "(served {})".format(child) not in state:
                goal_reached = False
                break

        if goal_reached:
            return 0

        # Identify unserved children and their locations/allergy status
        unserved_children_info = {} # {child_name: (place_name, is_allergic)}
        waiting_children = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "waiting", "*", "*")}
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for child in self.all_children_to_serve: # Only consider children who need serving
             if child in waiting_children and child not in served_children:
                 place = waiting_children[child]
                 is_allergic = (child in self.allergic_children) # Use pre-computed static info
                 unserved_children_info[child] = (place, is_allergic)

        # Identify available sandwiches and their status
        available_sandwiches = {} # {sandwich_name: (is_gluten_free, location_type, location_name)}
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        ontray_sandwiches = {} # {sandwich_name: tray_name}
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:]
                ontray_sandwiches[s] = t

        gluten_free_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        for s in kitchen_sandwiches:
            is_gf = (s in gluten_free_sandwiches)
            available_sandwiches[s] = (is_gf, 'kitchen', 'kitchen')

        for s, t in ontray_sandwiches.items():
            is_gf = (s in gluten_free_sandwiches)
            available_sandwiches[s] = (is_gf, 'tray', t)

        # Identify tray locations
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1] in self.all_trays}


        total_heuristic = 0

        # Calculate cost for each unserved child
        for child, (child_place, is_allergic) in unserved_children_info.items():
            # Default cost if sandwich needs making and subsequent steps
            best_sandwich_cost = 4

            # Check existing sandwiches to find the minimum cost stage
            for s_name, (s_is_gf, s_loc_type, s_loc_name) in available_sandwiches.items():
                # Check if sandwich is suitable (GF if needed)
                if is_allergic and not s_is_gf:
                    continue # Not suitable

                # Calculate cost based on sandwich location
                current_sandwich_cost = 4 # Default if not one of the known locations/stages

                if s_loc_type == 'tray':
                    tray_name = s_loc_name
                    if tray_name in tray_locations: # Ensure tray location is known
                        tray_place = tray_locations[tray_name]
                        if tray_place == child_place:
                            # Stage 3: Sandwich on tray at child's place (needs 1 action: serve)
                            current_sandwich_cost = 1
                        else:
                            # Stage 2: Sandwich on tray elsewhere (needs 1 action: move tray + 1 action: serve)
                            current_sandwich_cost = 2
                    # else: tray location unknown? This case implies an inconsistency if s is ontray but tray location is not in state.
                    # We default to cost 4, assuming it's hard to get.
                elif s_loc_type == 'kitchen':
                    # Stage 1: Sandwich in kitchen (needs 1 action: put on tray + 1 action: move tray + 1 action: serve)
                    current_sandwich_cost = 3

                # Update best cost found so far for this child
                best_sandwich_cost = min(best_sandwich_cost, current_sandwich_cost)

            # Add the minimum cost for this child to the total heuristic
            total_heuristic += best_sandwich_cost

        return total_heuristic
