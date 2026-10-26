from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or raise an error
        # For robustness, return empty list or handle based on expected input
        # Assuming valid PDDL fact strings as input based on problem description
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all waiting children.
    The heuristic counts the number of unserved children, the number of
    sandwiches that need to be made, the number of sandwiches that need
    to be put on trays, and the number of locations that need tray deliveries.

    This heuristic is not admissible but aims to guide a greedy best-first
    search efficiently.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions (e.g., (served child1))
        static_facts = task.static # Facts that are true in every state

        # Extract static information: child allergy status and waiting location
        self.child_allergy = {} # Map child -> True if allergic, False otherwise
        self.child_waiting_place = {} # Map child -> waiting place
        self.goal_children = set() # Set of children that need to be served
        self.static_facts = static_facts # Store static facts if needed later (e.g., for ingredient types)

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'allergic_gluten':
                self.child_allergy[parts[1]] = True
            elif parts[0] == 'not_allergic_gluten':
                self.child_allergy[parts[1]] = False
            elif parts[0] == 'waiting':
                self.child_waiting_place[parts[1]] = parts[2]

        # Identify all children that are goals (i.e., need to be served)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'served':
                 self.goal_children.add(parts[1])

        # Store all possible places from static facts (kitchen and others)
        # This is useful for iterating through locations later
        self.all_places = {'kitchen'} # kitchen is a constant place
        for place in self.child_waiting_place.values():
             self.all_places.add(place)

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify unserved children
        unserved_children = set()
        unserved_gf_children = set()
        unserved_reg_children = set()
        unserved_children_at_place = {} # Map place -> set of unserved children at that place

        for child in self.goal_children:
            if f'(served {child})' not in state:
                unserved_children.add(child)
                place = self.child_waiting_place.get(child) # Get waiting place from static
                if place: # Child must have a waiting place
                    unserved_children_at_place.setdefault(place, set()).add(child)
                    if self.child_allergy.get(child, False): # Default to False if allergy status unknown
                        unserved_gf_children.add(child)
                    else:
                        unserved_reg_children.add(child)

        # If no unserved children, goal is reached
        if not unserved_children:
            return 0

        # Initialize heuristic cost
        h = 0

        # Cost Component 1: Serve actions (one per unserved child)
        # This is a lower bound on the number of serve actions needed.
        h += len(unserved_children)

        # 2. Count available sandwiches and trays
        ontray_sandwiches = {} # Map sandwich -> tray
        at_kitchen_sandwiches = set() # Sandwiches at kitchen, not on tray
        no_gluten_sandwiches = set()
        tray_locations = {} # Map tray -> place

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'ontray':
                ontray_sandwiches[parts[1]] = parts[2]
            elif parts[0] == 'at_kitchen_sandwich':
                at_kitchen_sandwiches.add(parts[1])
            elif parts[0] == 'no_gluten_sandwich':
                no_gluten_sandwiches.add(parts[1])
            elif parts[0] == 'at' and parts[1].startswith('tray'): # Assuming anything starting with 'tray' is a tray object
                 tray_locations[parts[1]] = parts[2]

        # Count available sandwiches by type and current state (on tray, at kitchen)
        ontray_gf_total = {s for s in ontray_sandwiches if s in no_gluten_sandwiches}
        ontray_reg_total = {s for s in ontray_sandwiches if s not in no_gluten_sandwiches}

        kitchen_gf_not_ontray = {s for s in at_kitchen_sandwiches if s in no_gluten_sandwiches}
        kitchen_reg_not_ontray = {s for s in at_kitchen_sandwiches if s not in no_gluten_sandwiches}

        # 3. Calculate sandwiches needed at each stage

        # Total GF/Reg sandwiches needed for unserved children
        needed_gf_total = len(unserved_gf_children)
        needed_reg_total = len(unserved_reg_children)

        # Total sandwiches needed on trays (anywhere)
        # These are the sandwiches required to serve all children, minus those already on trays
        needed_on_tray_total = max(0, needed_gf_total - len(ontray_gf_total)) + max(0, needed_reg_total - len(ontray_reg_total))

        # Cost Component 2: Make actions
        # These are needed sandwiches that are not already made (neither on tray nor at kitchen)
        # We count how many of the sandwiches needed on trays must come from being made
        num_kitchen_total_not_ontray = len(kitchen_gf_not_ontray) + len(kitchen_reg_not_ontray)
        needed_make = max(0, needed_on_tray_total - num_kitchen_total_not_ontray)
        h += needed_make

        # Cost Component 3: Put_on_tray actions
        # These are needed sandwiches that are at the kitchen (not on tray)
        # We count how many of the sandwiches needed on trays must come from the kitchen
        needed_put = min(num_kitchen_total_not_ontray, needed_on_tray_total)
        h += needed_put

        # 4. Calculate tray movements needed
        # Count sandwiches already on trays at each location
        ontray_gf_at_place = {p: set() for p in self.all_places}
        ontray_reg_at_place = {p: set() for p in self.all_places}

        for s, t in ontray_sandwiches.items():
            place = tray_locations.get(t)
            if place:
                ontray_gf_at_place[place].update({s for s in ontray_sandwiches if s in no_gluten_sandwiches and ontray_sandwiches[s] == t})
                ontray_reg_at_place[place].update({s for s in ontray_sandwiches if s not in no_gluten_sandwiches and ontray_sandwiches[s] == t})


        # Cost Component 4: Move_tray actions
        # Count locations (excluding kitchen) that need tray deliveries
        locations_needing_tray_move = 0
        for place, children_at_place in unserved_children_at_place.items():
            if place == 'kitchen': continue # Children don't wait at the kitchen

            num_unserved_gf_at_place = len({c for c in children_at_place if c in unserved_gf_children})
            num_unserved_reg_at_place = len({c for c in children_at_place if c in unserved_reg_children})

            num_ontray_gf_at_place = len(ontray_gf_at_place.get(place, set()))
            num_ontray_reg_at_place = len(ontray_reg_at_place.get(place, set()))

            # Number of sandwiches of each type needed at this place that are not already there on a tray
            needed_gf_arrive_at_place = max(0, num_unserved_gf_at_place - num_ontray_gf_at_place)
            needed_reg_arrive_at_place = max(0, num_unserved_reg_at_place - num_ontray_reg_at_place)

            # If any sandwiches are needed at this place, we need at least one tray move to bring them
            if needed_gf_arrive_at_place > 0 or needed_reg_arrive_at_place > 0:
                 locations_needing_tray_move += 1

        h += locations_needing_tray_move

        return h
