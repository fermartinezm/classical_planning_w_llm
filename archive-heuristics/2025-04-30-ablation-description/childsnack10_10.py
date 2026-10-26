from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-fact strings defensively
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
         return []
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
    This heuristic estimates the total number of actions required to serve all
    unserved children. It calculates the cost for each unserved child independently
    based on the current state of suitable sandwiches and trays.

    # Assumptions
    - The goal is to serve all children specified in the task goals.
    - Necessary ingredients (bread, content) are available in the kitchen if a sandwich needs to be made.
    - A tray is available in the kitchen if a sandwich needs to be put on a tray there.
    - A tray can always be moved to the required location.
    - The heuristic for a child is based on the "most ready" suitable sandwich found.

    # Heuristic Initialization
    - Extracts the set of goal facts, which specify which children need to be served.
    - Extracts static facts, particularly child allergy information and waiting locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children that need to be served based on the task goals.
    2. For each child that is not yet marked as `(served ?c)` in the current state:
       a. Determine the child's waiting location `p` and allergy status (allergic or not) using static facts.
       b. Find the "best" state among suitable sandwiches (if any exist) in the current state. A sandwich `s` is suitable if it meets the child's allergy requirement. The states are prioritized as follows:
          - **State 1 (Best):** A suitable sandwich `s` is `ontray t` AND tray `t` is `at p`.
          - **State 2 (Good):** A suitable sandwich `s` is `ontray t` BUT tray `t` is NOT `at p`.
          - **State 3 (Okay):** A suitable sandwich `s` is `at_kitchen_sandwich`.
          - **State 4 (Needs Work):** No suitable sandwich exists anywhere in the required category (gluten-free if allergic, any otherwise).
       c. Calculate the estimated cost for this child based on the best state found:
          - If State 1: Cost = 1 (serve).
          - If State 2: Cost = 1 (move tray) + 1 (serve) = 2.
          - If State 3: Cost = 1 (put on tray) + (1 if child is not at kitchen else 0) (move tray) + 1 (serve). Total = 3 (if child not at kitchen), 2 (if child at kitchen).
          - If State 4: Cost = 1 (make sandwich) + 1 (put on tray) + (1 if child is not at kitchen else 0) (move tray) + 1 (serve). Total = 4 (if child not at kitchen), 3 (if child at kitchen).
       d. Sum the costs for all unserved children.
    3. The total sum is the heuristic value for the current state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Set of goal facts, e.g., {..., '(served child1)', ...}
        self.static = task.static # Set of static facts, e.g., {..., '(allergic_gluten child4)', '(waiting child4 table3)', ...}

        # Extract static information for quick lookup
        self.child_allergy = {} # Maps child name to True if allergic, False otherwise
        self.child_waiting_place = {} # Maps child name to their waiting place

        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == "allergic_gluten" and len(parts) == 2:
                child_name = parts[1]
                self.child_allergy[child_name] = True
            elif parts[0] == "not_allergic_gluten" and len(parts) == 2:
                 child_name = parts[1]
                 self.child_allergy[child_name] = False
            elif parts[0] == "waiting" and len(parts) == 3:
                 child_name, place_name = parts[1:]
                 self.child_waiting_place[child_name] = place_name

        # Identify all children that need to be served from the goals
        self.children_to_serve = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip invalid goals

            if parts[0] == "served" and len(parts) == 2:
                child_name = parts[1]
                self.children_to_serve.add(child_name)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        total_heuristic_cost = 0

        # Pre-process state facts for faster lookup
        state_facts_dict = {}
        tray_locations = {}
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             args = tuple(parts[1:])
             if predicate not in state_facts_dict:
                 state_facts_dict[predicate] = set()
             state_facts_dict[predicate].add(args)

             # Populate served children and tray locations
             if predicate == "at" and len(args) == 2:
                 obj, loc = args
                 # Check if the object is a tray (simple check based on name starting with 'tray')
                 # A more robust way would be to parse types from domain, but name check is usually sufficient for examples
                 if obj.startswith("tray"): # Assuming tray names start with 'tray'
                     tray_locations[obj] = loc


        # Identify children already served in the current state using the pre-processed dict
        served_children_in_state = {child for child, in state_facts_dict.get("served", set())}


        # Iterate through all children that need to be served according to the goal
        for child in self.children_to_serve:
            # If the child is already served, they contribute 0 to the heuristic
            if child in served_children_in_state:
                continue

            # Child is unserved. Calculate cost for this child.
            child_cost = 0
            child_place = self.child_waiting_place.get(child) # Get waiting place from static info
            is_allergic = self.child_allergy.get(child, False) # Default to not allergic if not specified

            # This should not happen in valid problems based on the example structure,
            # but handle defensively if waiting place isn't in static.
            if child_place is None:
                 # Cannot serve a child whose waiting place is unknown from static facts.
                 # This state might be unsolvable or malformed according to the problem definition.
                 # In a real planner, this might indicate an unreachable state or problem error.
                 # For a heuristic, we can assign a high cost or skip, assuming valid problems.
                 # Skipping means this child doesn't contribute to the heuristic, which is fine
                 # if they truly cannot be served based on static info.
                 continue


            # --- Find the "best" state of a suitable sandwich for this child ---
            # We prioritize states that are closer to being served.
            # State 1: Suitable sandwich on tray at child's location?
            found_state1 = False
            # State 2: Suitable sandwich on tray not at child's location?
            found_state2 = False
            # State 3: Suitable sandwich at kitchen?
            found_state3 = False
            # State 4: Need to make sandwich (no suitable sandwich anywhere)

            # Check for sandwiches on trays first (State 1 and 2)
            if "ontray" in state_facts_dict:
                for s, t in state_facts_dict["ontray"]:
                    # Check if sandwich s is suitable for child
                    s_is_gluten_free = (("no_gluten_sandwich", s) in state_facts_dict.get("no_gluten_sandwich", set()))
                    is_suitable = (not is_allergic) or (is_allergic and s_is_gluten_free)

                    if is_suitable:
                        # Check where tray t is
                        tray_loc = tray_locations.get(t)
                        if tray_loc == child_place:
                            found_state1 = True
                            break # Found the best state for this child
                        elif tray_loc is not None: # Tray is somewhere, but not at child's place
                             found_state2 = True
                             # Don't break yet, maybe there's a tray closer (State 1)

            # If we didn't find State 1, check for sandwiches at the kitchen (State 3)
            if not found_state1 and "at_kitchen_sandwich" in state_facts_dict:
                 for s, in state_facts_dict["at_kitchen_sandwich"]:
                     s_is_gluten_free = (("no_gluten_sandwich", s) in state_facts_dict.get("no_gluten_sandwich", set()))
                     is_suitable = (not is_allergic) or (is_allergic and s_is_gluten_free)
                     if is_suitable:
                         found_state3 = True
                         break # Found a suitable sandwich at kitchen

            # --- Calculate cost based on the best state found ---
            if found_state1:
                # State 1: Suitable sandwich on tray at child's location
                child_cost = 1 # serve
            elif found_state2:
                # State 2: Suitable sandwich on tray not at child's location
                child_cost = 1 + 1 # move tray + serve
            elif found_state3:
                # State 3: Suitable sandwich at kitchen
                child_cost = 1 # put on tray
                if child_place != "kitchen":
                    child_cost += 1 # move tray from kitchen
                child_cost += 1 # serve
            else: # State 4: No suitable sandwich exists anywhere
                # Need to make one
                child_cost = 1 # make sandwich
                # The made sandwich is conceptually at_kitchen_sandwich
                child_cost += 1 # put on tray
                if child_place != "kitchen":
                    child_cost += 1 # move tray from kitchen
                child_cost += 1 # serve

            total_heuristic_cost += child_cost

        return total_heuristic_cost
