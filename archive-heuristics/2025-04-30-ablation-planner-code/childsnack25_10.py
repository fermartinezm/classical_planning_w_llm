# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
import math # Import math for infinity

# Dummy Heuristic base class for standalone testing
# In a real planning environment, this would be provided.
class Heuristic:
    def __init__(self, task):
        self.task = task # Store task for access to initial_state, objects, static, goals
        pass
    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty fact strings
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Remove outer parentheses and split by whitespace
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

    Summary:
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the necessary actions in a relaxed way, summing up:
    1. The number of suitable sandwiches that need to be made.
    2. The number of suitable sandwiches that need to be put on trays.
    3. The number of tray movements required to bring trays to locations with
       waiting children.
    4. The number of children who are still waiting to be served.

    Assumptions:
    - Sufficient bread and content portions are available in the kitchen to make
      any required sandwiches (except limited by 'notexist' sandwich objects).
    - Sufficient trays exist to be moved to all required locations.
    - The 'kitchen' place always exists.
    - Sandwiches, once made, can be used for any child requiring that type
      (gluten-free or regular).
    - A tray at a location can hold multiple sandwiches and serve multiple
      children waiting at that location.

    Heuristic Initialization:
    - Extracts static information about which children are allergic or not.
    - Identifies all tray objects present in the initial state by parsing
      task.objects.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all children who are currently waiting and have not yet been served.
    2. Determine the allergy status (needs gluten-free or regular) for each waiting, unserved child.
    3. Count the total number of waiting, unserved children needing gluten-free sandwiches and those needing regular sandwiches.
    4. Count the number of gluten-free and regular sandwiches that are currently available (either in the kitchen or already on a tray).
    5. Count the number of gluten-free and regular sandwiches that are currently on trays.
    6. Count the number of 'notexist' sandwich objects available to be made.
    7. Count the total number of tray objects (from initialization).
    8. Identify the set of distinct locations where waiting, unserved children are located.
    9. Identify the set of distinct locations where trays are currently located.
    10. Calculate the number of sandwiches that still need to be *made*: This is the deficit between the total number of suitable sandwiches required by waiting children and the number of suitable sandwiches already available. This count is capped by the number of 'notexist' sandwich objects. If more are needed than can be made, the state is likely unsolvable (return infinity).
    11. Calculate the number of sandwiches that need to be *put on trays*: This is the deficit between the total number of suitable sandwiches required by waiting children and the number of suitable sandwiches already on trays.
    12. Calculate the number of *move_tray* actions needed: This is the number of distinct locations with waiting, unserved children that do not currently have a tray. If children are waiting but no trays exist, the state is unsolvable (return infinity).
    13. Calculate the number of *serve* actions needed: This is simply the total number of waiting, unserved children.
    14. The total heuristic value is the sum of the costs from steps 10, 11, 12, and 13.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and initial objects.
        """
        self.task = task # Store task for access to initial_state, objects, static, goals

        # Extract static facts: allergy status
        self.allergic_children = set()
        self.not_allergic_children = set()
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'allergic_gluten' and len(parts) == 2:
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                self.not_allergic_children.add(parts[1])

        # Identify all tray objects from task.objects
        self.all_trays = set()
        # Assuming task.objects is a dict {object_name: type}
        if hasattr(task, 'objects') and task.objects:
             for obj_name, obj_type in task.objects.items():
                 if obj_type == 'tray':
                     self.all_trays.add(obj_name)
        # Note: If task.objects is not available or doesn't list all trays,
        # this heuristic might be inaccurate regarding the total number of trays.
        # A robust solution would parse the problem file's :objects section.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # --- Parse State ---
        served_children = set()
        waiting_children_map = {} # child -> place
        at_kitchen_sandwich_set = set()
        ontray_map = {} # sandwich -> tray
        no_gluten_sandwich_set_in_state = set() # Sandwiches in the current state that are GF
        at_tray_map = {} # tray -> place
        notexist_sandwich_set = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            if predicate == 'served' and len(parts) == 2:
                served_children.add(parts[1])
            elif predicate == 'waiting' and len(parts) == 3:
                waiting_children_map[parts[1]] = parts[2]
            elif predicate == 'at_kitchen_sandwich' and len(parts) == 2:
                at_kitchen_sandwich_set.add(parts[1])
            elif predicate == 'ontray' and len(parts) == 3:
                ontray_map[parts[1]] = parts[2]
            elif predicate == 'no_gluten_sandwich' and len(parts) == 2:
                 no_gluten_sandwich_set_in_state.add(parts[1])
            elif predicate == 'at' and len(parts) == 3 and parts[1] in self.all_trays: # Ensure it's a tray location
                 at_tray_map[parts[1]] = parts[2]
            elif predicate == 'notexist' and len(parts) == 2:
                notexist_sandwich_set.add(parts[1])

        # --- Derive Counts and Sets ---
        waiting_unserved_children = {
            c for c in waiting_children_map if c not in served_children
        }
        waiting_unserved_locations = {
            waiting_children_map[c] for c in waiting_unserved_children
        }

        allergic_waiting_unserved = waiting_unserved_children.intersection(self.allergic_children)
        non_allergic_waiting_unserved = waiting_unserved_children.intersection(self.not_allergic_children)

        # Sandwiches that exist in the state (either kitchen or ontray)
        existing_sandwiches = at_kitchen_sandwich_set.union(set(ontray_map.keys()))

        # Available sandwiches are existing sandwiches
        available_gf_sandwiches = existing_sandwiches.intersection(no_gluten_sandwich_set_in_state)
        available_reg_sandwiches = existing_sandwiches - available_gf_sandwiches # Existing sandwiches that are not GF

        ontray_sandwiches = set(ontray_map.keys())
        ontray_gf_sandwiches = ontray_sandwiches.intersection(no_gluten_sandwich_set_in_state)
        ontray_reg_sandwiches = ontray_sandwiches - ontray_gf_sandwiches # Sandwiches on tray that are not GF

        trays_at_locations_set = set(at_tray_map.values())

        num_waiting_unserved = len(waiting_unserved_children)
        num_allergic_waiting_unserved = len(allergic_waiting_unserved)
        num_non_allergic_waiting_unserved = len(non_allergic_waiting_unserved)
        num_available_gf = len(available_gf_sandwiches)
        num_available_reg = len(available_reg_sandwiches)
        num_ontray_gf = len(ontray_gf_sandwiches)
        num_ontray_reg = len(ontray_reg_sandwiches)
        num_notexist = len(notexist_sandwich_set)
        num_trays = len(self.all_trays) # Total trays from initial state

        # --- Unsolvability Check ---
        if num_waiting_unserved == 0:
            return 0 # Goal state

        if num_trays == 0:
             # If there are waiting children but no trays, it's unsolvable.
             return math.inf

        # Check if enough sandwich objects can potentially exist (made or already existing)
        total_sandwiches_needed = num_allergic_waiting_unserved + num_non_allergic_waiting_unserved
        total_sandwiches_potential = len(existing_sandwiches) + num_notexist

        if total_sandwiches_needed > total_sandwiches_potential:
             # Not enough sandwich objects exist or can be made to serve all children
             return math.inf

        # Check if enough GF sandwiches can potentially exist
        # This is a simplification; it assumes any 'notexist' can become GF.
        # A more accurate check would consider available GF bread/content vs total bread/content.
        # Sticking to the simpler check based on notexist for efficiency.
        total_gf_sandwiches_potential = num_available_gf + num_notexist
        if num_allergic_waiting_unserved > total_gf_sandwiches_potential:
             return math.inf # Not enough GF sandwiches can ever exist


        # --- Calculate Costs ---

        # 1. Make Cost: Number of suitable sandwiches that need to be made.
        # This is the deficit between needed and available sandwiches, capped by notexist.
        make_gf_needed = max(0, num_allergic_waiting_unserved - num_available_gf)
        make_reg_needed = max(0, num_non_allergic_waiting_unserved - num_available_reg)

        # Total sandwiches we ideally need to make
        ideal_total_to_make = make_gf_needed + make_reg_needed

        # The actual number we can make is limited by notexist sandwiches
        make_cost = min(ideal_total_to_make, num_notexist)


        # 2. Put on Tray Cost: Number of suitable sandwiches that need to transition to 'ontray'.
        # This is the deficit between the total number of suitable sandwiches needed on trays
        # and the number currently on trays.
        # Total GF needed on trays = num_allergic_waiting_unserved
        # Total Reg needed on trays = num_non_allergic_waiting_unserved
        # Total GF currently on trays = num_ontray_gf
        # Total Reg currently on trays = num_ontray_reg
        put_on_tray_cost = max(0, num_allergic_waiting_unserved - num_ontray_gf) + \
                           max(0, num_non_allergic_waiting_unserved - num_ontray_reg)


        # 3. Move Tray Cost: Number of locations needing a tray that don't have one.
        locations_needing_tray = waiting_unserved_locations - trays_at_locations_set
        move_tray_cost = len(locations_needing_tray)

        # 4. Serve Cost: Number of children still waiting to be served.
        serve_cost = num_waiting_unserved

        # Total heuristic is the sum of these costs.
        total_cost = make_cost + put_on_tray_cost + move_tray_cost + serve_cost

        return total_cost
