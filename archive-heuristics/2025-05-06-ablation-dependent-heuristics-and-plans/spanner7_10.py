from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It considers the cost of tightening each nut, picking up spanners, and the travel required.

    # Assumptions:
    - Each spanner can be used to tighten exactly one nut.
    - The man can carry at most one spanner at a time (implied by action effects).
    - Travel cost between locations is the shortest path distance (number of walk actions).
    - The heuristic assumes there are enough usable spanners available in the problem instance to tighten all goal nuts.
    - Object naming conventions: man starts with 'b' or 'm', spanner with 's', nut with 'n'.

    # Heuristic Initialization
    - Extracts the set of nuts that need to be tightened from the goal conditions.
    - Builds a graph of locations based on the `link` static facts.
    - Computes all-pairs shortest paths between all locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the set of loose nuts that are goal conditions (`N_loose_goal`). This is done by checking which nuts are `loose` in the current state and `tightened` in the goal.
    2. If `N_loose_goal` is empty, the heuristic is 0 (goal state reached).
    3. Identify the man's current location (`man_loc`) and determine if the man is currently carrying a usable spanner (`man_carrying_usable`).
    4. Identify the locations of all usable spanners that are not being carried (`usable_spanner_locs`).
    5. Identify the locations of all loose nuts that are goals (`nut_locs`).
    6. Calculate the base cost, which includes the minimum number of `tighten_nut` and `pickup_spanner` actions required:
       - Add 1 for each nut in `N_loose_goal` (for the `tighten_nut` action).
       - Add 1 for each spanner pickup needed. The number of pickups needed is the total number of nuts to tighten minus the number of usable spanners the man is currently carrying (at most 1), ensuring the result is not negative: `max(0, |N_loose_goal| - man_carrying_usable)`.
    7. Calculate the travel cost:
       - The man needs to travel from his current location (`man_loc`) to the first required item. If he is not carrying a usable spanner and needs one (`|N_loose_goal| > 0`), the first required item is the closest usable spanner. Otherwise, it's the closest loose goal nut. Add the shortest path distance for this first leg of travel. If the required item is unreachable, the heuristic is infinity.
       - For the remaining `|N_loose_goal| - 1` nuts, the man needs to perform a cycle of getting a spanner (if needed) and traveling to the next nut. This involves at least two travel segments (from previous nut location to spanner location, and from spanner location to next nut location). Add a fixed cost of 2 travel actions (minimum 1 walk per segment) for each of these remaining nuts as a simplified estimate of the travel within each subsequent cycle.
    8. The total heuristic value is the sum of the base cost and the travel cost. If any required location or spanner is unreachable, the heuristic returns infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal nuts (nuts that must be tightened)
        self.goal_nuts = {
            args[0] for goal in self.goals if get_parts(goal)[0] == "tightened"
        }

        # Build location graph and compute shortest paths
        self.locations = set()
        self.links = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.links.add((l1, l2))
                self.links.add((l2, l1)) # Links are bidirectional

        self.distances = {}
        for start_loc in self.locations:
            self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Computes shortest path distances from start_loc to all other locations."""
        q = deque([(start_loc, 0)])
        visited = {start_loc: 0}
        self.distances[(start_loc, start_loc)] = 0

        while q:
            current_loc, dist = q.popleft()

            for l1, l2 in self.links:
                if l1 == current_loc and l2 not in visited:
                    visited[l2] = dist + 1
                    self.distances[(start_loc, l2)] = dist + 1
                    q.append((l2, dist + 1))

        # Mark unreachable locations with infinity
        for loc in self.locations:
            if (start_loc, loc) not in self.distances:
                 self.distances[(start_loc, loc)] = float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify loose nuts that are goals
        loose_goal_nuts = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "loose", "*") and get_parts(fact)[1] in self.goal_nuts
        }

        K = len(loose_goal_nuts)

        # If all goal nuts are tightened, heuristic is 0
        if K == 0:
            return 0

        # Identify man's location and if carrying a usable spanner
        man_name = None
        man_loc = None
        man_carrying_usable = 0
        carried_spanner = None

        # Find the man object name (assuming only one man and he is locatable or carrying)
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'carrying' and len(parts) == 3:
                 man_name = parts[1]
                 break
        if man_name is None:
             # Fallback: Assume object starting with 'b' or 'm' is the man
             for fact in state:
                 parts = get_parts(fact)
                 if parts[0] == 'at' and len(parts) == 3:
                     obj, loc = parts[1], parts[2]
                     if obj.startswith('b') or obj.startswith('m'):
                          man_name = obj
                          break

        if man_name is None:
             # Man object not found in state facts
             return float('inf')

        # Find man's location
        for fact in state:
            if match(fact, "at", man_name, "*"):
                man_loc = get_parts(fact)[2]
                break

        if man_loc is None:
             # Man is not at any location? Unsolvable.
             return float('inf')

        # Check if man is carrying a usable spanner
        for fact in state:
            if match(fact, "carrying", man_name, "*"):
                carried_spanner = get_parts(fact)[2]
                break
        if carried_spanner and f"(usable {carried_spanner})" in state:
             man_carrying_usable = 1

        # Identify usable spanners not carried and their locations
        usable_spanner_locs = {} # Map spanner name to location
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Assuming spanners start with 's'
                if obj.startswith('s') and obj != carried_spanner:
                    if f"(usable {obj})" in state:
                        usable_spanner_locs[obj] = loc

        # Identify loose goal nut locations
        nut_locs = {} # Map nut name to location
        for nut in loose_goal_nuts:
             found_loc = False
             for fact in state:
                 if match(fact, "at", nut, "*"):
                     nut_locs[nut] = get_parts(fact)[2]
                     found_loc = True
                     break
             if not found_loc:
                 # Nut exists but is not at any location? Unsolvable.
                 return float('inf')

        # Calculate base cost (tighten and pickup actions)
        cost = K # Tighten actions
        pickups_needed = max(0, K - man_carrying_usable)
        cost += pickups_needed # Pickup actions

        # Calculate travel cost
        travel_cost = 0
        if K > 0:
            # Cost to reach the first required item
            first_target_loc = None
            if man_carrying_usable:
                # Go to closest nut
                if nut_locs:
                    closest_nut_loc = min(nut_locs.values(), key=lambda loc: self.distances.get((man_loc, loc), float('inf')))
                    if self.distances.get((man_loc, closest_nut_loc), float('inf')) == float('inf'):
                         return float('inf') # Unreachable
                    first_target_loc = closest_nut_loc
                else: # K > 0 but nut_locs is empty? Should not happen if loose_goal_nuts > 0
                    return float('inf')
            else:
                # Need spanner first. Go to closest spanner.
                if usable_spanner_locs:
                    closest_spanner_loc = min(usable_spanner_locs.values(), key=lambda loc: self.distances.get((man_loc, loc), float('inf')))
                    if self.distances.get((man_loc, closest_spanner_loc), float('inf')) == float('inf'):
                         return float('inf') # Unreachable
                    first_target_loc = closest_spanner_loc
                elif nut_locs: # Need spanner but none available
                     return float('inf') # Unsolvable
                # If no usable spanners and no nuts need tightening (K=0), handled above.
                # If no usable spanners but nuts need tightening (K>0), handled above.
                # If usable spanners available, first target is the closest one.

            # Add distance to the first target
            if first_target_loc:
                 travel_cost += self.distances.get((man_loc, first_target_loc), float('inf'))
                 if travel_cost == float('inf'):
                      return float('inf') # Unreachable

            # Cost for remaining nuts (K-1 of them)
            # Each remaining nut requires getting a spanner and going to the nut.
            # This involves travel from previous nut loc to spanner, and spanner to current nut loc.
            # Minimum 2 travel actions per remaining nut (1 walk to spanner, 1 walk to nut).
            travel_cost += (K - 1) * 2

        cost += travel_cost

        return cost
