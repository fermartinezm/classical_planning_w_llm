from fnmatch import fnmatch
from collections import deque
import sys # Used for float('inf')

# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not available for standalone testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            # Add any other necessary task attributes accessed by the heuristic
            # For spanner, we need initial_state to find initial locations
            self.initial_state = task.initial_state


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
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
    This heuristic estimates the number of actions (tighten, pickup, walk)
    required to reach a state where all nuts are tightened. It counts the
    remaining tighten actions, estimates necessary spanner pickups, and
    adds the minimum travel cost for the man to reach a relevant location
    (either a loose nut or a usable spanner on the ground).

    # Assumptions:
    - The man can carry multiple spanners simultaneously.
    - Each usable spanner can tighten exactly one nut before becoming unusable.
    - Links between locations are bidirectional.
    - The problem is solvable with the initially available usable spanners.

    # Heuristic Initialization
    - The heuristic precomputes the shortest path distances between all pairs
      of locations in the domain graph, derived from the static `link` facts
      and locations mentioned in the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all loose nuts and their locations. Count the total number of loose nuts (`N_loose`).
    3. If `N_loose` is 0, the state is a goal state, return 0.
    4. Identify all usable spanners and determine if they are carried by the man or on the ground.
    5. Count the number of usable spanners carried by the man (`N_carrying_usable`) and the total number of usable spanners available (`Total_usable`).
    6. Check for unsolvability based on spanner availability: If `N_loose > Total_usable`, return a large value (e.g., 1000000).
    7. Initialize the heuristic value `h` with `N_loose` (representing the `tighten_nut` actions needed).
    8. Calculate the number of additional spanners the man needs to pick up from the ground: `num_pickups_needed = max(0, N_loose - N_carrying_usable)`. Add `num_pickups_needed` to `h` (representing `pickup_spanner` actions).
    9. Identify the set of "target" locations the man needs to visit:
       - All locations with loose nuts.
       - All locations with usable spanners on the ground, but only if `num_pickups_needed > 0`.
    10. Calculate the minimum shortest path distance from the man's current location to any of the target locations. This estimates the cost of the first necessary movement.
    11. Add this minimum movement cost to `h`.
    12. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances between locations.
        """
        super().__init__(task)

        self.graph = {}
        all_locations = set()

        # Build graph from static link facts
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.graph.setdefault(l1, set()).add(l2)
                self.graph.setdefault(l2, set()).add(l1) # Links are bidirectional
                all_locations.add(l1)
                all_locations.add(l2)

        # Add locations from initial state facts that might not be linked
        # This ensures all locations in the problem are included in the distance map
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 loc = get_parts(fact)[2]
                 all_locations.add(loc)
                 self.graph.setdefault(loc, set()) # Ensure all locations are keys

        self.distances = {}
        # Compute all-pairs shortest paths using BFS
        for start_loc in all_locations:
            self.distances[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[start_loc][start_loc] = 0

            while q:
                curr_loc, curr_dist = q.popleft()

                # Get neighbors, handling locations with no links
                neighbors = self.graph.get(curr_loc, set())

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_loc][neighbor] = curr_dist + 1
                        q.append((neighbor, curr_dist + 1))

            # Mark unreachable locations with infinity
            for other_loc in all_locations:
                if other_loc not in self.distances[start_loc]:
                    self.distances[start_loc][other_loc] = float('inf') # Use infinity


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Extract relevant information from the state
        man_loc = None
        loose_nuts = set()
        nut_locations = {}
        usable_spanners = set()
        spanner_locations = {} # Location if on ground, None if carried
        carried_spanners = set()

        # Identify the man object (assuming there's only one and its name starts with 'man' or 'bob')
        # A more robust way would parse types from domain, but this is domain-dependent
        # and 'bob' is likely the man object name in instances.
        man_obj_name = None
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == "at" and len(parts) == 3:
                 obj, loc = parts[1:]
                 # Check if this object is the man. We assume the man is the only object of type 'man'.
                 # We don't have type info here, so rely on naming convention or check if it's the one object that is 'carrying'.
                 # Let's rely on 'carrying' predicate to identify the man object name.
                 # We will find the man's location based on the object name found via 'carrying'.
                 pass # Defer man_loc assignment until man_obj_name is found

        # Find the man object name and carried spanners first
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == "carrying" and len(parts) == 3:
                 man_obj_name = parts[1]
                 spanner_obj = parts[2]
                 carried_spanners.add(spanner_obj)

        # Now iterate again to get locations and other facts, using the identified man_obj_name
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "at":
                obj, loc = parts[1:]
                if obj == man_obj_name:
                     man_loc = loc
                elif obj.startswith('nut'): # Assuming 'nut' is the nut's name prefix
                     nut_locations[obj] = loc
                elif obj.startswith('spanner'): # Assuming 'spanner' is the spanner's name prefix
                     spanner_locations[obj] = loc
            elif predicate == "usable":
                 # fact is (usable ?s)
                 spanner_obj = parts[1]
                 usable_spanners.add(spanner_obj)
            elif predicate == "loose":
                 # fact is (loose ?n)
                 nut_obj = parts[1]
                 loose_nuts.add(nut_obj)
            # We don't need 'tightened' facts directly, as 'loose' implies not tightened

        # Ensure man_loc was found (should always be the case in valid states)
        if man_loc is None:
             # This indicates an unexpected state structure, maybe return a high cost
             # This could happen if the man object name couldn't be determined (e.g., no 'carrying' fact initially)
             # A fallback could be to find the single object of type 'man' if type info was available.
             # For this domain, assuming 'bob' or the object found via 'carrying' is sufficient.
             # If man_obj_name wasn't found via 'carrying', try finding the single object that is 'at' a location and isn't a nut or spanner.
             if man_obj_name is None:
                 for fact in state:
                     parts = get_parts(fact)
                     if parts and parts[0] == "at" and len(parts) == 3:
                         obj, loc = parts[1:]
                         if not obj.startswith('nut') and not obj.startswith('spanner'):
                             man_obj_name = obj
                             man_loc = loc
                             break # Found the man

             if man_loc is None: # Still couldn't find the man
                 return float('inf') # Or a large number indicating error/unsolvable


        # 2. Count loose nuts
        N_loose = len(loose_nuts)

        # 3. Goal check
        if N_loose == 0:
            return 0

        # 4. Count usable spanners
        N_carrying_usable = len(usable_spanners.intersection(carried_spanners))
        # Note: A spanner is usable *and* carried.
        # The usable_spanners set contains all usable spanners in the world.
        # carried_spanners set contains all spanners carried by the man (usable or not).
        # N_carrying_usable is the count of spanners that are *both* usable *and* carried.

        N_ground_usable = len(usable_spanners) - N_carrying_usable
        Total_usable = N_carrying_usable + N_ground_usable

        # 6. Solvability check based on spanners
        if N_loose > Total_usable:
            # Not enough usable spanners exist in the world to tighten all loose nuts
            return 1000000 # Large finite number indicating likely unsolvable

        # 7. Base cost: tighten actions
        h = N_loose

        # 8. Add cost for pickup actions needed
        # The man needs N_loose usable spanners in total. He starts with N_carrying_usable.
        # He needs to pick up max(0, N_loose - N_carrying_usable) more usable spanners from the ground.
        num_pickups_needed = max(0, N_loose - N_carrying_usable)
        h += num_pickups_needed

        # 9. Identify target locations for movement
        loose_nut_locations = {nut_locations[n] for n in loose_nuts if n in nut_locations} # Ensure nut location is known
        usable_ground_spanner_locations = {
            spanner_locations[s]
            for s in usable_spanners.difference(carried_spanners)
            if s in spanner_locations and spanner_locations[s] is not None # Ensure spanner location is known and on ground
        }

        target_locations = set()
        target_locations.update(loose_nut_locations)
        if num_pickups_needed > 0:
            target_locations.update(usable_ground_spanner_locations)

        # 10. Calculate minimum movement cost to a target location
        movement_cost = 0
        if target_locations: # Only add movement cost if there are targets
            min_dist = float('inf')
            # Ensure man_loc is a valid key in distances map
            if man_loc in self.distances:
                for target_loc in target_locations:
                     # Ensure the target location is reachable from man_loc
                     if target_loc in self.distances[man_loc]:
                        min_dist = min(min_dist, self.distances[man_loc][target_loc])
                     # else: target_loc is unreachable from man_loc, min_dist remains inf

            if min_dist != float('inf'):
                 movement_cost = min_dist
            else:
                 # If closest target is unreachable, problem is likely unsolvable
                 return 1000000

        # 11. Add movement cost
        h += movement_cost

        # 12. Return total heuristic value
        return h
