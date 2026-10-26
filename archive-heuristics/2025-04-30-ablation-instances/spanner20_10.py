from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts
    by summing the estimated costs for tightening, picking up spanners, and walking.
    It assumes a man can carry multiple spanners.

    # Assumptions
    - There is a single man object, identifiable in the state facts (e.g., via 'at' or 'carrying').
    - The goal is to tighten a specific set of nuts.
    - Each usable spanner can tighten exactly one nut before becoming unusable.
    - A man can carry multiple spanners.
    - The man is always located at some location ('at' predicate is true for the man).

    # Heuristic Initialization
    - Identify the set of nuts that need to be tightened (goal nuts).
    - Static facts (like links) are not used in this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of loose nuts that are part of the goal (i.e., need tightening). Let this count be `num_loose_nuts`. If 0, heuristic is 0.
    2. Add `num_loose_nuts` to the heuristic cost (for the `tighten_nut` actions).
    3. Identify the man object and his current location by examining 'at' and 'carrying' facts.
    4. Identify all spanners and nuts mentioned in the state to distinguish the man object.
    5. Count the number of usable spanners the man is currently carrying. Let this be `num_carried_usable`.
    6. Calculate the number of additional usable spanners the man needs to pick up: `spanners_to_pickup = max(0, num_loose_nuts - num_carried_usable)`.
    7. Add `spanners_to_pickup` to the heuristic cost (for the `pickup_spanner` actions).
    8. Calculate walk costs:
       a. Walks to nuts: Count how many loose nuts are not at the man's current location. Add this count to the heuristic (assuming 1 walk action per nut location not currently at).
       b. Walks to spanners: Count how many usable spanners are on the ground at the man's current location. If `spanners_to_pickup > 0`:
          - If there's at least one usable spanner on the ground at the man's location, the man saves one walk to a spanner location for the first pickup. Add `spanners_to_pickup - 1` to the heuristic.
          - Otherwise (no usable spanners on the ground at the man's location), the man needs to walk to a spanner location for each pickup. Add `spanners_to_pickup` to the heuristic.
    9. The total heuristic value is the sum of costs from steps 2, 7, and 8.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts.
        """
        self.goals = task.goals

        # Identify all nuts that are goals (need to be tightened)
        self.goal_nuts = {
            args[0] for goal in self.goals if get_parts(goal)[0] == "tightened"
            for args in [get_parts(goal)[1:]]
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose nuts that are goals
        loose_nuts_in_state = {
            args[0] for fact in state if match(fact, "loose", "*")
            for args in [get_parts(fact)[1:]]
        }
        loose_goal_nuts = self.goal_nuts.intersection(loose_nuts_in_state)
        num_loose_nuts = len(loose_goal_nuts)

        # If all goal nuts are tightened, heuristic is 0
        if num_loose_nuts == 0:
            return 0

        h = 0

        # 2. Add cost for tighten actions
        h += num_loose_nuts

        # 4. Identify all spanners and nuts mentioned in state
        all_spanner_names = set()
        all_nut_names = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "carrying" and len(parts) == 3:
                all_spanner_names.add(parts[2])
            elif parts[0] == "usable" and len(parts) == 2:
                 all_spanner_names.add(parts[1])
            elif parts[0] == "loose" and len(parts) == 2:
                 all_nut_names.add(parts[1])
            elif parts[0] == "tightened" and len(parts) == 2:
                 all_nut_names.add(parts[1])

        # 3. Find man object and location
        man_object = None
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name = get_parts(fact)[1]
                # The man is the object at a location that is not a known spanner or nut
                if obj_name not in all_spanner_names and obj_name not in all_nut_names:
                    man_object = obj_name
                    man_location = get_parts(fact)[2]
                    break # Assuming one man

        # If man object or location couldn't be determined (unexpected state)
        if man_object is None or man_location is None:
             # This should not happen in valid states based on domain dynamics,
             # but as a fallback for robustness:
             # print("Error: Man object or location not found in state.")
             return float('inf') # Indicate an unhandleable state

        # Collect spanner and nut locations, and carried spanners
        spanner_locations = {} # {spanner: location}
        nut_locations = {}     # {nut: location}
        carried_spanners = set() # set of spanner names being carried

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj_name = parts[1]
                loc_name = parts[2]
                if obj_name in all_spanner_names:
                    spanner_locations[obj_name] = loc_name
                elif obj_name in all_nut_names:
                    nut_locations[obj_name] = loc_name
                # Man location already found

            elif parts[0] == "carrying" and len(parts) == 3 and parts[1] == man_object:
                carried_spanners.add(parts[2])

        # Identify usable spanners
        usable_spanners_in_state = {
            args[0] for fact in state if match(fact, "usable", "*")
            for args in [get_parts(fact)[1:]]
        }

        # 5. Usable spanners the man is carrying
        carried_usable_spanners = carried_spanners.intersection(usable_spanners_in_state)
        num_carried_usable = len(carried_usable_spanners)

        # Usable spanners on the ground
        usable_spanners_on_ground = {
            s for s in usable_spanners_in_state if s in spanner_locations
        }

        # 6. Calculate spanners to pickup
        spanners_to_pickup = max(0, num_loose_nuts - num_carried_usable)

        # 7. Add cost for pickup actions
        h += spanners_to_pickup

        # 8a. Walks to nuts
        nuts_away = sum(1 for nut in loose_goal_nuts if nut_locations.get(nut) != man_location)
        h += nuts_away

        # 8b. Walks to spanners
        if spanners_to_pickup > 0:
            # Check if any usable spanner on the ground is at the man's location
            usable_spanners_at_man_location = sum(
                1 for s in usable_spanners_on_ground if spanner_locations.get(s) == man_location
            )

            if usable_spanners_at_man_location > 0:
                # Man can pick up the first needed spanner without walking to it
                h += spanners_to_pickup - 1
            else:
                # Man needs to walk to a spanner location for each pickup
                h += spanners_to_pickup

        # Ensure heuristic is non-negative
        return max(0, h)
