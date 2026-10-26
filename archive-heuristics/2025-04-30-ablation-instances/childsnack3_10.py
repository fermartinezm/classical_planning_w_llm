from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The fact must have at least as many parts as the pattern arguments
    if len(parts) < len(args):
        return False
    # Check if the first len(args) parts match the pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It sums the minimum estimated actions needed for each unserved child independently.
    The estimate for a child is based on the current state of the most readily available suitable sandwich.

    # Assumptions
    - Each unserved child needs one suitable sandwich.
    - Resources (bread, content, trays) are eventually available to make and deliver sandwiches if needed.
    - The cost for a child is the minimum cost among the available suitable sandwiches.
    - The costs are:
        - 1: Suitable sandwich is on a tray at the child's location. (Serve)
        - 2: Suitable sandwich is on a tray elsewhere. (Move tray, Serve)
        - 3: Suitable sandwich is in the kitchen. (Put on tray, Move tray, Serve)
        - 4: Suitable sandwich does not exist yet. (Make, Put on tray, Move tray, Serve)
    - The heuristic assumes that if a suitable sandwich is needed and doesn't exist, it can be made (cost 4 path is possible).

    # Heuristic Initialization
    The heuristic extracts static information:
    - Which children are allergic and where they are waiting.
    - The set of all children (specifically, those who are waiting), sandwiches, and trays defined in the problem instance.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost `h = 0`.
    2. Identify all children who are not yet served based on the current state.
    3. For each unserved child:
        a. Determine the child's waiting location and allergy status (from static info).
        b. Identify the set of sandwiches suitable for this child (all if not allergic, only gluten-free if allergic). Gluten-free status of sandwiches is checked in the current state.
        c. Determine the minimum cost to get a suitable sandwich to this child's location and serve them, based on the current state of the *most available* suitable sandwich. The checks are performed in increasing order of cost:
            - Check if any suitable sandwich is currently on a tray at the child's waiting location. If yes, the minimum cost for this child is 1.
            - Else, check if any suitable sandwich is currently on a tray at *any* location (which must be different from the child's location if the previous check failed). If yes, the minimum cost for this child is 2.
            - Else, check if any suitable sandwich is currently in the kitchen. If yes, the minimum cost for this child is 3.
            - Else, if any suitable sandwich object exists but is currently marked as `notexist`, the minimum cost for this child is 4. This is the default cost if none of the above conditions are met, assuming a suitable sandwich can always be made if needed.
        d. Add this minimum cost for the child to the total heuristic cost `h`.
    4. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children,
        sandwiches, and trays.
        """
        self.goals = task.goals
        static_facts = task.static

        self.child_allergies = {}
        self.child_locations = {}
        self.all_children = set()
        self.all_sandwiches = set()
        self.all_trays = set()

        all_facts = task.initial_state | static_facts

        # Extract children and their static info (focus on waiting children)
        waiting_children_facts = {fact for fact in static_facts if match(fact, "waiting", "*", "*")}
        self.all_children = {get_parts(fact)[1] for fact in waiting_children_facts}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "waiting" and parts[1] in self.all_children:
                 self.child_locations[parts[1]] = parts[2]
            elif parts[0] == "allergic_gluten" and parts[1] in self.all_children:
                 self.child_allergies[parts[1]] = True
            elif parts[0] == "not_allergic_gluten" and parts[1] in self.all_children:
                 self.child_allergies[parts[1]] = False

        # Extract all sandwiches and trays mentioned in initial state or static facts
        for fact in all_facts:
             parts = get_parts(fact)
             if len(parts) > 1:
                 if parts[0] in ["at_kitchen_sandwich", "ontray", "no_gluten_sandwich", "notexist"]:
                     if len(parts) > 1: self.all_sandwiches.add(parts[1])
                 elif parts[0] in ["at", "ontray"]:
                      if parts[0] == "at" and len(parts) > 2:
                          self.all_trays.add(parts[1])
                      elif parts[0] == "ontray" and len(parts) > 2:
                           self.all_trays.add(parts[2])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached (all children in self.all_children are served)
        # The goals are typically (served child) for all waiting children.
        # Let's check against the explicit goals from the task.
        if self.goals <= state:
             return 0

        # Identify served children based on the state
        served_children_in_state = {parts[1] for fact in state if match(fact, "served", "*")}

        # We only care about children who are in our list of children (those who were waiting)
        unserved_children = {c for c in self.all_children if c not in served_children_in_state}

        total_cost = 0

        # Pre-compute sandwich and tray locations/statuses for faster lookup
        sandwich_location = {} # map sandwich -> 'kitchen', 'ontray_trayname', 'notexist'
        sandwich_ontray_map = {} # map sandwich -> trayname if on tray
        sandwich_is_gluten_free = set() # set of gluten free sandwich names
        tray_location = {} # map tray -> place

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at_kitchen_sandwich" and parts[1] in self.all_sandwiches:
                sandwich_location[parts[1]] = 'kitchen'
            elif parts[0] == "ontray" and parts[1] in self.all_sandwiches and parts[2] in self.all_trays:
                s, t = parts[1], parts[2]
                sandwich_location[s] = f'ontray_{t}'
                sandwich_ontray_map[s] = t
            elif parts[0] == "notexist" and parts[1] in self.all_sandwiches:
                 # Only add if we haven't seen it elsewhere (kitchen/ontray)
                 if parts[1] not in sandwich_location:
                     sandwich_location[parts[1]] = 'notexist'
            elif parts[0] == "no_gluten_sandwich" and parts[1] in self.all_sandwiches:
                sandwich_is_gluten_free.add(parts[1])
            elif parts[0] == "at" and parts[1] in self.all_trays:
                t, p = parts[1], parts[2]
                tray_location[t] = p

        for child in unserved_children:
            waiting_place = self.child_locations[child]
            is_allergic = self.child_allergies.get(child, False) # Default to not allergic if info missing

            min_child_cost = 4 # Default worst case: make, put, move, serve

            # Find suitable sandwiches based on allergy and existence as objects
            suitable_sandwiches = {s for s in self.all_sandwiches
                                   if (not is_allergic) or (s in sandwich_is_gluten_free)}

            # If no suitable sandwich objects exist at all (very unlikely in solvable problems),
            # the heuristic would be stuck. Let's assume suitable sandwich objects exist.
            # If suitable_sandwiches is empty, min_child_cost remains 4.

            # Check for the best available suitable sandwich
            found_best_sandwich = False

            # Cost 1: Suitable sandwich on tray at child's location
            for s in suitable_sandwiches:
                if s in sandwich_ontray_map:
                    t = sandwich_ontray_map[s]
                    if t in tray_location and tray_location[t] == waiting_place:
                        min_child_cost = 1
                        found_best_sandwich = True
                        break # Found cost 1, no need to check further for this child
            if found_best_sandwich:
                total_cost += min_child_cost
                continue # Move to next child

            # Cost 2: Suitable sandwich on tray elsewhere
            # Check if any suitable sandwich is on *any* tray *anywhere* (that wasn't at the waiting place)
            found_cost_2 = False
            if min_child_cost > 1: # Only check if cost 1 wasn't found
                for s in suitable_sandwiches:
                     if s in sandwich_ontray_map:
                         t = sandwich_ontray_map[s]
                         # Check if tray is anywhere *other* than the waiting place
                         if t not in tray_location or tray_location[t] != waiting_place:
                              min_child_cost = min(min_child_cost, 2)
                              found_cost_2 = True
                              break # Found cost 2 possibility

            # Cost 3: Suitable sandwich in kitchen
            found_cost_3 = False
            if min_child_cost > 2: # Only check if cost 1 or 2 wasn't found
                 for s in suitable_sandwiches:
                     if s in sandwich_location and sandwich_location[s] == 'kitchen':
                         min_child_cost = min(min_child_cost, 3)
                         found_cost_3 = True
                         break # Found cost 3 possibility

            # Cost 4: Suitable sandwich notexist
            # If min_child_cost is still 4, it means no suitable sandwich was found
            # in states corresponding to costs 1, 2, or 3. The default cost is 4,
            # assuming a suitable sandwich exists in the 'notexist' state and can be made.
            # No explicit check needed here, as min_child_cost is already 4.

            total_cost += min_child_cost

        return total_cost
