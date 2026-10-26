from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions outside the class
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
    """

    def __init__(self, task):
        """
        Initialize the heuristic.

        Heuristic Initialization:
        - No specific static facts are pre-processed in this version,
          as the required information (nut locations, spanner locations, links)
          is either dynamic or derived from dynamic facts for this heuristic's calculation.
          The set of all nuts could be extracted from goals, but we find loose nuts dynamically.
        """
        # We could extract all nut objects from goals here if needed,
        # but finding loose nuts dynamically in __call__ is sufficient.
        # Example: self.all_nuts = {get_parts(g)[1] for g in task.goals if match(g, "tightened", "*")}
        pass

    def __call__(self, node):
        """
        Estimate the minimum number of actions required to reach a goal state.

        Summary:
        This heuristic estimates the total number of 'tighten_nut', 'pickup_spanner',
        and 'walk' actions needed to tighten all loose nuts. It sums up the required
        actions based on the current state, simplifying walk costs.

        Assumptions:
        - Nuts are static objects; their location does not change.
        - Spanners become unusable after one 'tighten_nut' action.
        - The man object is named 'bob'.
        - All locations are reachable from each other (connectivity is assumed for walk cost simplification).
        - The number of usable spanners available throughout the problem is sufficient
          to tighten all nuts in solvable instances.

        Step-By-Step Thinking for Computing Heuristic:
        1. Count the number of loose nuts (`N_loose`). This is the base cost for
           'tighten_nut' actions. If `N_loose` is 0, the goal is reached, return 0.
        2. Find the man's ('bob') current location.
        3. Determine if the man is currently carrying a usable spanner.
        4. Identify all nut objects present in the state (loose or tightened).
        5. Find all spanner objects present in the state (carried, usable, or at a location).
        6. Find usable spanners available on the ground (usable and not carried) and record their locations.
        7. Calculate the number of 'pickup_spanner' actions needed: This is the total
           number of spanners required (`N_loose`) minus 1 if the man is already
           carrying a usable spanner. Add this count to the heuristic.
        8. Identify the locations of all loose nuts (`NutLocations`).
        9. Estimate the 'walk' cost to visit all `NutLocations`: This is the number
           of distinct nut locations, minus 1 if the man is already at one of them.
           Add this cost to the heuristic. This estimates the travel between nut locations.
        10. Determine how many of the needed spanners (from step 7) must be picked up
            from locations that do *not* contain loose nuts. This requires counting
            usable spanners on the ground that are located at `NutLocations`.
            Subtract this count from the total spanners needed from the ground
            to find spanners needed from 'other' locations (`SpannersNeededFromOtherLocs`).
        11. Estimate the additional 'walk' cost to visit these 'other' spanner locations:
            This is simply `SpannersNeededFromOtherLocs`. Add this cost to the heuristic.
        12. The total heuristic value is the sum of costs from steps 1, 7, 9, and 11.
        """
        state = node.state

        # Find man object and his location
        man_obj = 'bob' # Assuming the man is named 'bob'
        man_loc = None
        for fact in state:
            if match(fact, "at", man_obj, "*"):
                man_loc = get_parts(fact)[2]
                break
        # In a valid state, the man should always be at a location.

        # Find loose nuts and their locations
        loose_nuts = set()
        all_nuts_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "loose" and len(parts) == 2:
                n = parts[1]
                loose_nuts.add(n)
                all_nuts_in_state.add(n)
            elif parts[0] == "tightened" and len(parts) == 2:
                 all_nuts_in_state.add(parts[1])


        N_loose = len(loose_nuts)
        if N_loose == 0:
            return 0

        # Find locations for loose nuts (assuming nuts are static)
        NutLocations = set()
        for n in loose_nuts:
            for fact in state:
                if match(fact, "at", n, "*"):
                    NutLocations.add(get_parts(fact)[2])
                    break # Assuming each nut is at only one location


        # Find all spanner objects in the state
        all_spanners_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "carrying" and len(parts) == 3 and parts[1] == man_obj:
                all_spanners_in_state.add(parts[2])
            elif parts[0] == "usable" and len(parts) == 2:
                all_spanners_in_state.add(parts[1])
            elif parts[0] == "at" and len(parts) == 3:
                 obj = parts[1]
                 # If the object is not the man and not a known nut, assume it's a spanner
                 if obj != man_obj and obj not in all_nuts_in_state:
                      all_spanners_in_state.add(obj)


        # Check if man is carrying usable spanner
        carried_spanner_usable = False
        carried_spanner_obj = None
        for fact in state:
            if match(fact, "carrying", man_obj, "*"):
                carried_spanner_obj = get_parts(fact)[2]
                if "(usable " + carried_spanner_obj + ")" in state:
                    carried_spanner_usable = True
                break # Assuming man carries at most one spanner


        # Find usable spanners on ground and their locations
        usable_spanners_on_ground = set()
        usable_spanner_locations_map = {} # Map spanner object to location

        for s in all_spanners_in_state:
            # Check if usable
            is_usable = "(usable " + s + ")" in state
            # Check if on ground (not carried)
            is_carried = (s == carried_spanner_obj)

            if is_usable and not is_carried:
                usable_spanners_on_ground.add(s)
                # Find its location
                for fact in state:
                    if match(fact, "at", s, "*"):
                        usable_spanner_locations_map[s] = get_parts(fact)[2]
                        break
                # In a valid state, a usable spanner on the ground should have a location.


        # Calculate heuristic components

        # Base cost: tighten_nut actions
        h = N_loose

        # Cost for pickup_spanner actions
        SpannersNeededFromGround = max(0, N_loose - (1 if carried_spanner_usable else 0))
        h += SpannersNeededFromGround

        # Cost for walk actions

        # Walk cost to visit nut locations
        WalkCostNuts = max(0, len(NutLocations) - (1 if man_loc in NutLocations else 0))
        h += WalkCostNuts

        # Walk cost to visit additional spanner locations
        # Count usable spanners on the ground that are located at a nut location
        SpannersAvailableAtNutLocs = sum(1 for s in usable_spanners_on_ground if usable_spanner_locations_map.get(s) in NutLocations)

        # Number of spanners needed from locations that are NOT nut locations
        SpannersNeededFromOtherLocs = max(0, SpannersNeededFromGround - SpannersAvailableAtNutLocs)

        # Add walk cost for these additional spanner locations
        h += SpannersNeededFromOtherLocs # Simplified: assumes one walk segment per needed spanner location not at a nut.

        return h
