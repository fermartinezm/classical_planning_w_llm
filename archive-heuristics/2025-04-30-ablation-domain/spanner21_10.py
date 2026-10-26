from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions outside the class
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    # Check if lengths match and all parts match pattern args
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_all_locations(all_facts_set):
    """Extract all unique location names from the set of all possible facts."""
    locations = set()
    for fact_str in all_facts_set:
        parts = get_parts(fact_str)
        if len(parts) > 1:
            predicate = parts[0]
            # Check predicates that take locations as arguments in the spanner domain
            if predicate == "at" and len(parts) == 3:
                # (at ?obj ?loc) - ?loc is the 3rd part
                locations.add(parts[2])
            elif predicate == "link" and len(parts) == 3:
                # (link ?loc1 ?loc2) - ?loc1 and ?loc2 are 2nd and 3rd parts
                locations.add(parts[1])
                locations.add(parts[2])
    return locations

def build_location_graph(all_locations, static_facts):
    """Build an adjacency list representation of the location graph."""
    graph = {loc: [] for loc in all_locations}

    # Add edges from link facts
    for fact in static_facts:
        if match(fact, "link", "*", "*"):
            _, loc1, loc2 = get_parts(fact)
            # Ensure locations are in our known set
            if loc1 in graph and loc2 in graph:
                graph[loc1].append(loc2)
                graph[loc2].append(loc1)
            # else: Warning: link refers to unknown location? (Should not happen with correct PDDL and get_all_locations)

    return graph

def bfs_distances(graph, start_node):
    """Compute shortest path distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node in graph:
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            curr = queue.popleft()

            # Check if curr is still valid in graph keys (should be if initialized correctly)
            if curr not in graph: continue

            for neighbor in graph[curr]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[curr] + 1
                    queue.append(neighbor)
    # If start_node is not in graph, all distances remain inf. This is correct.
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It sums the number of loose nuts (for tighten actions), the cost to acquire
    a usable spanner if Bob isn't carrying one, and the maximum distance Bob
    needs to travel from his current location to reach any location with a loose nut.

    # Assumptions
    - Bob is the only agent.
    - Only usable spanners can tighten nuts.
    - Links between locations are bidirectional.
    - The problem is solvable (i.e., usable spanners exist if Bob isn't carrying one,
      and all necessary locations are connected and reachable from Bob's initial location).
    - All locations mentioned in the problem are captured by 'at' or 'link' predicates in task.facts.

    # Heuristic Initialization
    - Extracts the set of nuts that need to be tightened from the goal conditions.
    - Identifies all unique location names used in the problem.
    - Builds the location graph based on the static `link` facts, including all identified locations.
    - Stores the task object for efficient goal checking.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the goal is already reached using `self.task.goal_reached(state)`. If true, return 0.
    2. Parse the current state to find:
       - Bob's current location.
       - Which spanners are usable.
       - Which nuts are loose.
       - The location of every object (`at` predicate).
       - If Bob is carrying any spanner (`carrying` predicate).
    3. Determine if Bob is currently carrying a usable spanner.
    4. Identify the set of locations containing usable spanners that are on the ground (not carried by Bob).
    5. Identify the set of loose nut objects.
    6. If there are no loose nuts, the heuristic is 0 (already covered by goal check).
    7. Compute the number of loose nuts (`num_loose_nuts`). This is the base cost for the `tighten` actions.
    8. Compute shortest path distances from Bob's current location to all other locations using BFS on the location graph. If Bob's location is not in the graph or unreachable from itself (should not happen in valid problems), return a large number (1000) indicating an issue.
    9. Calculate the spanner acquisition cost (`spanner_acquisition_cost`):
       - If Bob is carrying at least one usable spanner, this cost is 0.
       - If Bob is not carrying a usable spanner, find the minimum distance from Bob's current location to any location containing a usable spanner on the ground. If no such spanner is reachable, return a large number (1000) indicating unsolvability from this state. Otherwise, the cost is this minimum distance plus 1 (for the `pickup` action).
    10. Calculate the movement cost to reach nuts (`max_dist_to_nut`):
        - For each loose nut, find its location. Find the maximum distance from Bob's current location to the location of any loose nut. If any loose nut is at an unknown or unreachable location, return a large number (1000) indicating unsolvability from this state.
    11. The total heuristic value is the sum: `num_loose_nuts + spanner_acquisition_cost + max_dist_to_nut`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal nuts and building the location graph."""
        # Extract the set of nuts that must be tightened in the goal state.
        # This is used mainly for identifying nut objects, the goal check uses task.goal_reached
        self.goal_nuts = {get_parts(goal)[1] for goal in task.goals if match(goal, "tightened", "*")}

        # Identify all unique location names used in the problem instance
        self.all_locations = get_all_locations(task.facts)

        # Build the location graph from static link facts, including all identified locations
        self.location_graph = build_location_graph(self.all_locations, task.static)

        # Store the task for efficient goal checking
        self.task = task


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if the goal is already reached.
        if self.task.goal_reached(state):
             return 0

        # 2. Parse the current state
        bob_location = None
        bob_carrying_spanner_obj = None # The specific spanner object Bob is carrying
        is_usable = set() # Set of usable spanner objects
        is_loose = set() # Set of loose nut objects
        obj_location = {} # Map object name to its location string

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                obj_location[obj] = loc
                if obj == "bob":
                    bob_location = loc
            elif parts[0] == "carrying" and len(parts) == 3:
                man, spanner = parts[1], parts[2]
                if man == "bob":
                    bob_carrying_spanner_obj = spanner
            elif parts[0] == "usable" and len(parts) == 2:
                spanner = parts[1]
                is_usable.add(spanner)
            elif parts[0] == "loose" and len(parts) == 2:
                nut = parts[1]
                is_loose.add(nut)

        # If Bob's location wasn't found, the state is malformed or Bob isn't at a location.
        # This shouldn't happen in a valid state, but return large cost for safety.
        if bob_location is None:
             return 1000

        # 3. Determine if Bob is currently carrying a usable spanner
        bob_carrying_usable_spanner = (bob_carrying_spanner_obj is not None and bob_carrying_spanner_obj in is_usable)

        # 4. Identify locations of usable spanners on the ground
        usable_spanners_on_ground_locs = set()
        for spanner in is_usable:
            if spanner != bob_carrying_spanner_obj: # If it's usable and not carried by Bob
                 loc = obj_location.get(spanner)
                 if loc is not None: # Ensure it has a location
                     usable_spanners_on_ground_locs.add(loc)

        # 5. Identify the set of loose nut objects (already done in parsing)
        # 6. If no loose nuts, goal is reached (already checked)
        if not is_loose:
            return 0

        # 7. Calculate the base cost (tighten actions)
        num_loose_nuts = len(is_loose)
        total_cost = num_loose_nuts

        # 8. Compute distances from Bob's current location
        distances_from_bob = bfs_distances(self.location_graph, bob_location)

        # Check if Bob's location is reachable within the graph (should be dist 0)
        if distances_from_bob.get(bob_location, float('inf')) == float('inf'):
             # Bob is at a location not in the graph or unreachable from itself.
             # This indicates a problem with the graph or state.
             return 1000

        # 9. Calculate the spanner acquisition cost
        spanner_acquisition_cost = 0
        if not bob_carrying_usable_spanner:
             min_dist_to_spanner = float('inf')
             for loc in usable_spanners_on_ground_locs:
                 if loc in distances_from_bob and distances_from_bob[loc] != float('inf'):
                      min_dist_to_spanner = min(min_dist_to_spanner, distances_from_bob[loc])

             if min_dist_to_spanner == float('inf'):
                 # No reachable usable spanner on the ground and Bob isn't carrying one.
                 # Likely unsolvable if loose nuts exist.
                 return 1000
             else:
                 spanner_acquisition_cost = min_dist_to_spanner + 1 # +1 for pickup

        total_cost += spanner_acquisition_cost

        # 10. Calculate the movement cost to reach nuts
        max_dist_to_nut = 0
        for nut in is_loose:
            nut_loc = obj_location.get(nut) # Get location of this specific loose nut
            # Ensure nut has a location and it's reachable from Bob
            if nut_loc is None or nut_loc not in distances_from_bob or distances_from_bob[nut_loc] == float('inf'):
                 # A loose nut exists but its location is unknown or unreachable. State is unsolvable.
                 return 1000
            max_dist_to_nut = max(max_dist_to_nut, distances_from_bob[nut_loc])

        total_cost += max_dist_to_nut

        # 11. Return the total estimated cost.
        return total_cost
