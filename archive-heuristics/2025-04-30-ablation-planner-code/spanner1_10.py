from fnmatch import fnmatch
from collections import deque
import math # Import math for infinity

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Utility functions
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
    # Ensure we don't go out of bounds if pattern is longer than fact parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the total cost to tighten all required nuts. It sums the estimated cost for each untightened nut independently. The estimated cost for a single nut includes the cost of the tighten action, the travel cost for the man to reach the nut's location, and the cost for the man to acquire a usable spanner and travel with it to the nut's location (if not already carrying one).

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Nuts and spanners are at fixed locations.
    - Each usable spanner can tighten exactly one nut.
    - There are enough usable spanners available (either carried or at locations) to tighten all goal nuts in solvable instances.
    - Travel cost between linked locations is 1. Shortest path distances are used for travel estimates.

    # Heuristic Initialization
    - Extract all locations and build a graph based on `link` predicates.
    - Compute all-pairs shortest paths between locations using BFS. Store these distances.
    - Identify the set of nuts that need to be tightened based on the goal state.
    - Identify the man object name.
    - Identify all spanner object names and nut object names for type checking.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of nuts that are in the goal state as `tightened` but are currently `loose` in the current state. These are the untightened goal nuts.
    2. If there are no untightened goal nuts, the heuristic is 0 (goal state reached).
    3. If there are untightened goal nuts, initialize the total heuristic cost `h` to 0.
    4. Determine the man's current location.
    5. Check if the man is currently carrying a usable spanner.
    6. Identify the locations of all currently available usable spanners that are not being carried by the man.
    7. For each untightened goal nut `N` at its location `L_N`:
        a. Add 1 to the cost for the `tighten_nut` action itself.
        b. Calculate the cost to get the man to `L_N` *with* a usable spanner:
            - If the man is currently carrying a usable spanner: The cost is the shortest path distance from the man's current location to `L_N`.
            - If the man is NOT currently carrying a usable spanner: The man must first travel to a location `L_S` of an available usable spanner `S`, pick it up (cost 1), and then travel from `L_S` to `L_N`. Find the minimum cost for this sequence over all available usable spanners: `min_{S, L_S} (dist(man_location, L_S) + 1 + dist(L_S, L_N))`. If no usable spanners are available at locations, this implies an issue (e.g., unsolvable state or all spanners are carried but not usable, which shouldn't happen based on domain effects), return a large value like infinity.
        c. Add this calculated travel-and-spanner-acquisition cost to the cost for nut `N`.
        d. Add the total cost for nut `N` to the overall heuristic `h`.
    8. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing shortest path distances
        and identifying goal nuts and the man object.
        """
        # Assume task provides goals, static facts, and objects (with name and type)
        self.goals = task.goals
        self.static = task.static
        self.objects = task.objects

        # Identify objects by type
        self.locations = {obj.name for obj in task.objects if obj.type == 'location'}
        self.spanners = {obj.name for obj in task.objects if obj.type == 'spanner'}
        self.nuts = {obj.name for obj in task.objects if obj.type == 'nut'}

        # Identify the man object name (assuming there is exactly one man)
        self.man_name = None
        for obj in task.objects:
            if obj.type == 'man':
                self.man_name = obj.name
                break

        # Build adjacency list for the location graph from 'link' predicates
        self.adj = {loc: set() for loc in self.locations}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                if l1 in self.locations and l2 in self.locations:
                    self.adj[l1].add(l2)
                    self.adj[l2].add(l1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.dist = {loc: {} for loc in self.locations}
        for start_node in self.locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.dist[start_node][start_node] = 0

            while q:
                current_loc, current_d = q.popleft()

                for neighbor in self.adj.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist[start_node][neighbor] = current_d + 1
                        q.append((neighbor, current_d + 1))

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut_name = get_parts(goal)[1]
                if nut_name in self.nuts: # Ensure it's actually a nut object
                    self.goal_nuts.add(nut_name)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # Find current state of nuts (loose and their locations)
        current_loose_nuts = set()
        nut_locations = {}

        for fact in state:
            if match(fact, "loose", "*"):
                nut_name = get_parts(fact)[1]
                if nut_name in self.nuts: # Ensure it's a nut
                    current_loose_nuts.add(nut_name)
            if match(fact, "at", "*", "*"):
                 obj_name, loc_name = get_parts(fact)[1:]
                 if obj_name in self.nuts and loc_name in self.locations: # Ensure it's a nut at a location
                     nut_locations[obj_name] = loc_name


        # Filter for nuts that are both loose (in state) and in goal
        untightened_goal_nuts_at_loc = {
            nut: loc for nut, loc in nut_locations.items()
            if nut in current_loose_nuts and nut in self.goal_nuts
        }

        # If all goal nuts are tightened, heuristic is 0
        if not untightened_goal_nuts_at_loc:
            return 0

        # Find man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_location = get_parts(fact)[2]
                break

        if man_location is None or man_location not in self.locations:
             # Should not happen in valid states, but handle defensively
             return math.inf # Man's location unknown or invalid

        # Check if man is carrying a usable spanner
        man_carrying_usable_spanner = False
        carried_spanner_name = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner_name = get_parts(fact)[2]
                if carried_spanner_name in self.spanners: # Ensure it's a spanner
                    break # Found the carried spanner
                else:
                    carried_spanner_name = None # Not carrying a spanner object we know about

        if carried_spanner_name and f"(usable {carried_spanner_name})" in state:
             man_carrying_usable_spanner = True

        # Find locations of usable spanners not carried by the man
        usable_spanners_at_loc = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name, loc_name = get_parts(fact)[1:]
                # Check if it's a spanner at a location
                if obj_name in self.spanners and loc_name in self.locations:
                    # Check if it's usable and not the one being carried
                    if obj_name != carried_spanner_name and f"(usable {obj_name})" in state:
                         usable_spanners_at_loc[obj_name] = loc_name

        total_cost = 0

        # Calculate cost for each untightened goal nut independently
        for nut_name, nut_location in untightened_goal_nuts_at_loc.items():
            # Cost for the tighten action
            cost_this_nut = 1

            # Cost to get man to nut_location with a usable spanner
            if man_carrying_usable_spanner:
                # Man is ready to go directly to the nut location
                travel_cost = self.dist.get(man_location, {}).get(nut_location, math.inf)
                if travel_cost == math.inf: return math.inf # Cannot reach nut
                cost_this_nut += travel_cost
            else:
                # Man needs to get a spanner first
                min_spanner_path_cost = math.inf
                if usable_spanners_at_loc:
                    for spanner_name, spanner_location in usable_spanners_at_loc.items():
                        # Cost = travel to spanner + pickup + travel to nut
                        travel_to_spanner = self.dist.get(man_location, {}).get(spanner_location, math.inf)
                        travel_spanner_to_nut = self.dist.get(spanner_location, {}).get(nut_location, math.inf)

                        if travel_to_spanner != math.inf and travel_spanner_to_nut != math.inf:
                             path_cost = travel_to_spanner + 1 + travel_spanner_to_nut
                             min_spanner_path_cost = min(min_spanner_path_cost, path_cost)

                if min_spanner_path_cost == math.inf:
                    # No path to any usable spanner or from spanner to nut
                    # This state is likely unsolvable (e.g., no usable spanners left)
                    return math.inf

                cost_this_nut += min_spanner_path_cost

            total_cost += cost_this_nut

        return total_cost
