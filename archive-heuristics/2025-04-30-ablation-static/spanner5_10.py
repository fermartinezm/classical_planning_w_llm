# Need to import necessary modules
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # Uncomment if running in planner environment

# Define dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Number of parts must exactly match the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It counts the required tighten and pickup actions and adds the estimated
    travel cost for the man to reach the first location where a necessary
    action (picking up a spanner or tightening a nut) can occur.

    # Assumptions
    - Each loose nut requires one tighten action.
    - Each tighten action consumes one usable spanner.
    - Each spanner used must be picked up by the man.
    - The man is the only agent performing actions.
    - The location graph is static and bidirectional (although PDDL only defines one-way links, we assume travel is possible in reverse for distance calculation).
    - The man object is named 'bob'. (This assumption is based on example instances; a more general heuristic would identify the man object from the task definition).

    # Heuristic Initialization
    - Extracts all locations and 'link' facts from the task's static information and initial state.
    - Computes all-pairs shortest paths between locations using BFS.
    - Stores the goal conditions (which nuts need to be tightened).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Check for Goal State: If all goal conditions (all target nuts tightened) are met, the heuristic is 0.
    2.  Identify State Information:
        -   Find the man's current location.
        -   Check if the man is currently carrying at least one usable spanner.
        -   Identify all loose nuts and their current locations.
        -   Identify all usable spanners that are not being carried and their current locations.
    3.  Check Solvability: Count the number of loose nuts (`num_loose`) and the total number of available usable spanners (`num_usable`, including any usable one carried). If `num_loose > num_usable`, the problem is unsolvable from this state with the available spanners, return infinity.
    4.  Calculate Non-Travel Actions:
        -   Each loose nut requires one `tighten_nut` action (cost 1). Total: `num_loose`.
        -   Each `tighten_nut` action requires a spanner to be picked up. If the man is already carrying a usable spanner, he saves one pickup action for the first nut. Total pickups: `num_loose - (1 if man is carrying usable else 0)`.
        -   Total non-travel cost = `num_loose` (tighten) + `num_loose - (1 if man is carrying usable else 0)` (pickup).
    5.  Estimate Travel Cost to First Task: The man needs to move from his current location to the first place where he can perform a necessary action.
        -   If the man is carrying a usable spanner, the first necessary action is tightening a nut. The minimum travel is the shortest distance from his current location to any location with a loose nut.
        -   If the man is not carrying a usable spanner, the first necessary action is picking one up. The minimum travel is the shortest distance from his current location to any location with a usable spanner that is not being carried.
        -   Handle cases where the required locations are unreachable (distance is infinity). If the first task location is unreachable, the problem is unsolvable, return infinity.
    6.  Sum Costs: The total heuristic value is the sum of the total non-travel actions and the estimated travel cost to the first task.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and computing
        all-pairs shortest paths between locations.
        """
        self.goals = task.goals  # Goal conditions (e.g., set of '(tightened nutX)' facts).

        # Extract all locations and build the adjacency list for the graph.
        locations = set()
        # Locations can be objects of type location, appearing in 'at' or 'link' facts
        # Check initial state and static facts
        for fact in task.initial_state | task.static:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 # (at ?m - locatable ?l - location)
                 # The second argument is the location
                 locations.add(parts[2])
             elif parts[0] == 'link' and len(parts) == 3:
                 # (link ?l1 - location ?l2 - location)
                 locations.add(parts[1])
                 locations.add(parts[2])

        self.locations = list(locations)
        self.adj = {loc: set() for loc in self.locations}

        # Build adjacency list from link facts (assuming bidirectional links)
        for fact in task.initial_state | task.static:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                if l1 in self.adj and l2 in self.adj: # Ensure locations were collected
                    self.adj[l1].add(l2)
                    self.adj[l2].add(l1) # Assume bidirectional

        # Compute all-pairs shortest paths using BFS from each location.
        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = {loc: float('inf') for loc in self.locations}
            self.dist[start_node][start_node] = 0
            queue = deque([start_node])

            while queue:
                u = queue.popleft()
                if u in self.adj: # Ensure u is a valid location with neighbors
                    for v in self.adj[u]:
                        if self.dist[start_node][v] == float('inf'):
                            self.dist[start_node][v] = self.dist[start_node][u] + 1
                            queue.append(v)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check for Goal State
        if self.goals <= state:
            return 0

        # 2. Identify State Information
        man_location = None
        carried_spanners = [] # List of spanners carried by the man
        man_carrying_usable = False

        man_name = 'bob' # Assuming man object is named 'bob'

        loose_nuts = {} # {nut_name: location}
        usable_spanners_at_locs = {} # {spanner_name: location}

        # First pass to find man's location and carried spanners
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1] == man_name:
                 man_location = parts[2]
            elif parts[0] == 'carrying' and len(parts) == 3 and parts[1] == man_name:
                 carried_spanners.append(parts[2])

        # Check if any carried spanner is usable
        carried_spanner_names = set(carried_spanners)
        for s in carried_spanner_names:
             if f'(usable {s})' in state:
                 man_carrying_usable = True
                 break # Found a usable one, no need to check others

        # Second pass to find loose nuts and usable spanners at locations
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1:]
                 if obj.startswith('nut') and f'(loose {obj})' in state:
                     loose_nuts[obj] = loc
                 elif obj.startswith('spanner') and obj not in carried_spanner_names:
                     if f'(usable {obj})' in state:
                          usable_spanners_at_locs[obj] = loc


        # 3. Check Solvability
        num_loose = len(loose_nuts)
        num_usable = (1 if man_carrying_usable else 0) + len(usable_spanners_at_locs)

        if num_loose > num_usable:
            return float('inf') # Unsolvable with available usable spanners

        # num_loose == 0 case is handled at the start

        # 4. Calculate Non-Travel Actions
        # Each loose nut needs a tighten action (cost 1)
        # Each loose nut needs a spanner picked up (cost 1), unless the first one is carried
        non_travel_cost = num_loose # tighten actions
        # We need num_loose spanners. If we are already carrying one usable, we need num_loose - 1 pickups.
        # If we are carrying spanners but NONE are usable, we still need num_loose pickups.
        # The check `man_carrying_usable` correctly handles this.
        non_travel_cost += num_loose - (1 if man_carrying_usable else 0) # pickup actions

        # 5. Estimate Travel Cost to First Task
        travel_cost_to_first_item = float('inf')

        # Ensure man_location is valid before looking up distances
        if man_location in self.dist:
            if man_carrying_usable:
                # Need to go to the nearest loose nut location
                min_dist_to_any_nut = float('inf')
                for nut, loc in loose_nuts.items():
                    if loc in self.dist[man_location]:
                        min_dist_to_any_nut = min(min_dist_to_any_nut, self.dist[man_location][loc])
                travel_cost_to_first_item = min_dist_to_any_nut
            else:
                # Need to go to the nearest usable spanner location
                min_dist_to_any_spanner = float('inf')
                for spanner, loc in usable_spanners_at_locs.items():
                     if loc in self.dist[man_location]:
                        min_dist_to_any_spanner = min(min_dist_to_any_spanner, self.dist[man_location][loc])
                travel_cost_to_first_item = min_dist_to_any_spanner
        # else: man_location is not in self.dist, implies unreachable locations, travel_cost_to_first_item remains inf

        # If the first task location is unreachable, the problem is unsolvable
        if travel_cost_to_first_item == float('inf'):
             return float('inf')

        # 6. Sum Costs
        total_cost = non_travel_cost + travel_cost_to_first_item

        return total_cost
