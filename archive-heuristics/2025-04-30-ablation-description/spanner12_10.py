# from heuristics.heuristic_base import Heuristic # Assuming this base class exists

from fnmatch import fnmatch
import math # For infinity

# Helper functions (can be outside or inside class)
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
    # The number of parts must exactly match the number of pattern arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    goal nuts that are currently loose. It counts the necessary tighten actions,
    spanner acquisition actions (pickup), and estimated move actions.

    # Assumptions
    - The man can carry multiple spanners.
    - Tightening a nut consumes the usability of *one* carried spanner.
    - Each loose goal nut requires one usable spanner to be used.
    - Spanners can be acquired by picking them up if they are on the ground
      at the man's current location and are usable.
    - Moves are estimated based on the total number of location-dependent actions
      needed (spanner acquisitions and tightens), assuming a cost of 1 per move
      between relevant locations, with a potential saving of 1 move if starting
      at a location suitable for the first action.
    - The graph of locations is connected.
    - There is only one man object in the domain.

    # Heuristic Initialization
    - Extracts the set of nuts that are required to be tightened in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of goal nuts that are currently in a 'loose' state. Let this count be `k`.
    2. If `k` is 0, the heuristic is 0 (goal reached for these nuts).
    3. Check if the total number of usable spanners in the world is less than `k`. If so, the problem is unsolvable from this state, return infinity.
    4. Initialize heuristic value `h = k` (representing the `tighten_nut` action needed for each loose goal nut).
    5. Determine the man's current state: his location and the set of usable spanners he is currently carrying or that are on the ground at his location.
    6. Calculate the number of *additional* usable spanners the man needs to acquire (`spanners_to_acquire`). This is `k` minus the number of usable spanners he has immediate access to (carrying or at his location). Add `spanners_to_acquire` to `h` (representing `pickup_spanner` actions).
    7. Estimate the number of `walk` actions needed:
       - The man needs to perform `k` tighten actions (at nut locations) and `spanners_to_acquire` pickup actions (at usable spanner locations on the ground).
       - The total number of location-dependent actions is `k + spanners_to_acquire`.
       - A sequence of `N` location-dependent actions requires at least `N-1` moves if the starting location is suitable for the first action, and `N` moves otherwise (assuming connectivity).
       - Check if the man's current location is suitable for the first action (either a usable spanner location on the ground if a pickup is needed first, or a nut location if a tighten is needed first).
       - Add the estimated number of moves (`k + spanners_to_acquire` or `k + spanners_to_acquire - 1`) to `h`.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal nuts."""
        # Extract nuts that need to be tightened from the goal state.
        self.goal_nuts = {get_parts(goal)[1] for goal in task.goals if match(goal, "tightened", "*")}
        # Static facts are not used for the simple move estimate.

    def __call__(self, node):
        """Estimate the minimum cost to tighten all remaining loose goal nuts."""
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nuts = {n for n in self.goal_nuts if f'(loose {n})' in state}
        num_loose_goal_nuts = len(loose_goal_nuts)

        # 2. If no loose goal nuts, goal is reached for these nuts
        if num_loose_goal_nuts == 0:
            return 0

        # Find man, his location, and carried spanners
        man_name = None
        L_M = None
        carried_spanners = set()
        known_spanners = set()
        known_nuts = set(self.goal_nuts)

        # First pass to find man via 'carrying' and collect known objects
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "carrying":
                man_name = parts[1] # Assuming the first argument is the man
                carried_spanners.add(parts[2])
                known_spanners.add(parts[2])
            elif parts[0] == "usable":
                 if len(parts) > 1: known_spanners.add(parts[1])
            elif parts[0] == "tightened" or parts[0] == "loose":
                 if len(parts) > 1: known_nuts.add(parts[1])

        # Second pass to find man's location and potentially man if not found by carrying
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if man_name is not None and obj == man_name:
                    L_M = loc
                # Fallback: if man not found yet, assume this 'at' fact is for the man
                # if the object is not a known spanner or nut.
                # This assumes man is the only other 'locatable' type.
                elif man_name is None and obj not in known_spanners and obj not in known_nuts:
                     man_name = obj
                     L_M = loc

        # If man_name or L_M is still None, the state is likely invalid or malformed
        if man_name is None or L_M is None:
             # print(f"Warning: Could not find man ({man_name}) or his location ({L_M}) in state.")
             return math.inf # Cannot proceed, treat as unsolvable

        # 3. Solvability check: Need enough usable spanners in the world
        all_usable_spanners = {s for fact in state if match(fact, "usable", s)}
        if len(all_usable_spanners) < num_loose_goal_nuts:
             return math.inf # Problem is unsolvable from this state

        # 4. Initialize heuristic with cost for tighten actions
        h = num_loose_goal_nuts

        # 5. Determine immediately available usable spanners
        usable_carried_spanners = {s for s in carried_spanners if f'(usable {s})' in state}
        
        usable_spanners_on_ground = {s for s in known_spanners if f'(usable {s})' in state and self._get_location(state, s) is not None}
        usable_spanners_at_lm = {s for s in usable_spanners_on_ground if self._get_location(state, s) == L_M}

        immediately_available_usable_spanners = usable_carried_spanners | usable_spanners_at_lm

        # 6. Calculate spanner acquisition actions needed
        spanners_to_acquire = max(0, num_loose_goal_nuts - len(immediately_available_usable_spanners))
        h += spanners_to_acquire # Cost for pickup actions

        # 7. Estimate move actions needed
        # Locations of loose goal nuts
        nut_locations = {self._get_location(state, n) for n in loose_goal_nuts}
        nut_locations.discard(None) # Remove None if any nut location is missing (shouldn't happen)

        # Locations of usable spanners on the ground that *could* be acquired
        # We only care about locations of usable spanners on the ground, as carried ones don't need pickup moves.
        usable_spanner_locations_on_ground = {self._get_location(state, s) for s in usable_spanners_on_ground}
        usable_spanner_locations_on_ground.discard(None)

        # Total location-dependent actions requiring the man to be at a specific place
        # These are the 'tighten' actions (at nut locations) and 'pickup' actions (at spanner locations)
        num_location_dependent_actions = spanners_to_acquire + num_loose_goal_nuts

        moves = num_location_dependent_actions # Base estimate: 1 move per location-dependent action

        # Check if the man starts at a useful location for the *first* required action
        starts_at_useful_location = False
        if num_location_dependent_actions > 0: # Only relevant if actions are needed
            if spanners_to_acquire > 0:
                # First action is pickup, needs to be at a usable spanner location on the ground
                if L_M in usable_spanner_locations_on_ground:
                    starts_at_useful_location = True
            else: # spanners_to_acquire == 0, means he has enough usable spanners immediately available
                # First action is tighten, needs to be at a nut location
                if L_M in nut_locations:
                    starts_at_useful_location = True

        if starts_at_useful_location:
             # Save the first move if starting at a location where the first action can occur
             moves -= 1

        # Ensure moves is not negative
        moves = max(0, moves)

        h += moves

        # 8. Return total heuristic value
        return h

    def _get_location(self, state, obj_name):
        """Finds the location of an object in the state."""
        for fact in state:
            if match(fact, "at", obj_name, "*"):
                return get_parts(fact)[2]
        return None # Object is not 'at' any location (e.g., carried)
