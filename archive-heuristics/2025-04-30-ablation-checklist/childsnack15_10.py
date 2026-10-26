from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    waiting children. It counts the deficit of necessary conditions:
    1. Children needing to be served.
    2. Sandwiches needing to be made (considering gluten requirements).
    3. Sandwiches needing to be put on trays.
    4. Trays needing to be moved to locations where children are waiting.

    # Heuristic Initialization
    - Extract static facts like which children are allergic to gluten.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are waiting but not yet served. Each such child
       requires a 'serve' action. Add the count of unserved children to the heuristic.
    2. For the unserved children, determine how many require gluten-free sandwiches
       and how many require regular sandwiches.
    3. Count the number of gluten-free and regular sandwiches that have already
       been made (either in the kitchen or on a tray).
    4. Calculate the deficit of made sandwiches of each type: `max(0, needed - made)`.
       Add the total deficit (GF + Regular) to the heuristic (estimating 'make_sandwich' actions).
       This assumes sufficient ingredients and 'notexist' slots are available, which is a relaxation.
    5. Count the number of sandwiches currently in the kitchen (`at_kitchen_sandwich`).
       Each of these needs a 'put_on_tray' action before it can be served. Add this count to the heuristic.
    6. Identify the distinct locations where unserved children are waiting.
    7. Count the number of trays that are currently located at one of these
       waiting places.
    8. Calculate the deficit of trays at needed locations: `max(0, num_needed_locations - num_trays_at_needed_locations)`.
       Add this deficit to the heuristic (estimating 'move_tray' actions). This assumes
       enough trays exist somewhere to be moved.
    9. The total heuristic value is the sum of costs from steps 1, 4, 5, and 8.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        self.goals = task.goals # Not strictly needed for this heuristic, but good practice
        static_facts = task.static

        # Identify which children are allergic to gluten from static facts
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "allergic_gluten", "*")
        }

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # --- Step 1: Count unserved children and add cost for 'serve' actions ---
        unserved_children = set()
        waiting_children_at_place = {} # Map child -> place
        served_children = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "waiting":
                child, place = parts[1], parts[2]
                waiting_children_at_place[child] = place
            elif parts[0] == "served":
                served_children.add(parts[1])

        for child in waiting_children_at_place:
            if child not in served_children:
                unserved_children.add(child)

        h = len(unserved_children) # Cost for the final 'serve' action for each child

        # If no children are unserved, the goal is reached.
        if not unserved_children:
            return 0

        # --- Step 2 & 3: Count needed vs made sandwiches (GF/Reg) ---
        unserved_gf_count = 0
        unserved_reg_count = 0
        for child in unserved_children:
            if child in self.allergic_children:
                unserved_gf_count += 1
            else:
                unserved_reg_count += 1

        made_gf_count = 0
        made_reg_count = 0
        sandwiches_on_trays = set()
        sandwiches_in_kitchen = set()
        gf_sandwiches = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at_kitchen_sandwich":
                sandwiches_in_kitchen.add(parts[1])
            elif parts[0] == "ontray":
                sandwiches_on_trays.add(parts[1])
            elif parts[0] == "no_gluten_sandwich":
                gf_sandwiches.add(parts[1])

        # Sandwiches that are made are either in the kitchen or on a tray
        made_sandwiches = sandwiches_in_kitchen | sandwiches_on_trays

        for s in made_sandwiches:
            if s in gf_sandwiches:
                made_gf_count += 1
            else:
                made_reg_count += 1

        # --- Step 4: Add cost for 'make_sandwich' actions ---
        needed_gf_to_make = max(0, unserved_gf_count - made_gf_count)
        needed_reg_to_make = max(0, unserved_reg_count - made_reg_count)
        h += needed_gf_to_make + needed_reg_to_make

        # --- Step 5: Add cost for 'put_on_tray' actions ---
        # Sandwiches currently in the kitchen need to be put on a tray
        h += len(sandwiches_in_kitchen)

        # --- Step 6, 7 & 8: Add cost for 'move_tray' actions ---
        # Identify distinct places where unserved children are waiting
        needed_places = {waiting_children_at_place[child] for child in unserved_children}

        # Count trays currently at one of these needed places
        trays_at_needed_places_count = 0
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[2] in needed_places:
                 # Check if the object is actually a tray type
                 # (This requires parsing types, which is complex.
                 # Assuming anything 'at' a place that isn't 'at-robby' in gripper
                 # or 'at' a package in logistics is a tray here, based on domain).
                 # A more robust way would involve parsing object types from the problem file.
                 # For this domain, only trays have 'at' location predicates.
                 trays_at_needed_places_count += 1

        # Add cost for moving trays to needed locations
        needed_tray_moves = max(0, len(needed_places) - trays_at_needed_places_count)
        h += needed_tray_moves

        return h

