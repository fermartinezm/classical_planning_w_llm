import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_node, graph):
    """
    Perform Breadth-First Search to find shortest distances from a start node
    in an unweighted graph.

    Args:
        start_node: The node to start the search from.
        graph: An adjacency dictionary representing the graph {node: [neighbors]}.

    Returns:
        A dictionary mapping each reachable node to its distance from the start_node.
    """
    distances = {start_node: 0}
    queue = collections.deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Handle nodes that might not have outgoing links
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    goal nuts. It considers the cost of tightening each loose goal nut
    independently and sums these costs. The cost for a single nut includes
    the tighten action, a pickup action (if a spanner isn't already carried),
    and the minimum walking distance for the man to get to a usable spanner
    and then to the nut's location.

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Each spanner can only be used once (becomes unusable after one tighten action).
    - The man can carry at most one spanner at a time.
    - The graph of locations connected by 'link' predicates is static.
    - All locations are reachable from each other if a path exists.
    - The problem is solvable (enough usable spanners exist). If not, the heuristic returns infinity.

    # Heuristic Initialization
    - Extracts all locations and 'link' relationships from static facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the set of nuts that need to be tightened based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all nuts that are currently 'loose' and are part of the goal state.
    2. If there are no loose goal nuts, the heuristic is 0.
    3. Find the man's current location.
    4. Find all usable spanners and their current locations.
    5. Check if the man is currently carrying a usable spanner.
    6. Count the number of loose goal nuts (`k`). Count the number of available usable spanners (`m_ground` on ground + `m_carried` carried). If `k > m_ground + m_carried`, the problem is unsolvable from this state, return infinity.
    7. Initialize the total heuristic cost to 0.
    8. For each loose goal nut `N` at location `L_N`:
        a. Add 1 to the cost for the `tighten_nut` action.
        b. Calculate the minimum cost for the man to get to `L_N` while carrying a usable spanner.
           i. If the man is currently carrying a usable spanner: The cost is the distance from the man's current location (`L_M`) to `L_N`. Add `dist(L_M, L_N)` to the cost for this nut.
           ii. If the man is NOT currently carrying a usable spanner: He needs to go to a usable spanner, pick it up, and then go to `L_N`. Find the minimum cost over all available usable spanners `S` at location `L_S`. The cost for a specific spanner `S` is `dist(L_M, L_S) + 1 (pickup) + dist(L_S, L_N)`. Find the minimum of this value over all usable spanners. Add this minimum cost to the cost for this nut. If no usable spanners are available, this nut cannot be tightened, and the total heuristic should be infinity (this case is handled by the check in step 6).
    9. Sum the costs calculated for each loose goal nut to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and identifying goal nuts.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify all locations and build the graph
        locations = set()
        adjacency_list = collections.defaultdict(list)

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                locations.add(loc1)
                locations.add(loc2)
                adjacency_list[loc1].append(loc2)
                adjacency_list[loc2].append(loc1) # Links are bidirectional

        self.locations = list(locations) # Store locations list
        self.adjacency_list = adjacency_list

        # 2. Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = bfs(start_loc, self.adjacency_list)

        # 3. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut = args[0]
                self.goal_nuts.add(nut)

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returns infinity if no path."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # This should ideally not happen in solvable problems if all locations are linked
             # within the graph, but return a large number just in case.
             return float('inf')
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose goal nuts in the current state
        loose_goal_nuts_in_state = {} # {nut: location}
        current_locations = {} # {object: location}
        usable_spanners_in_state = {} # {spanner: location}
        man_location = None
        man_carrying_spanner = None # spanner object if carrying, else None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
                # Identify man's location
                if match(fact, "at", "*", "*") and parts[1] in [p[1] for p in [get_parts(f) for f in task.initial_state] if p[0] == 'at' and len(p) > 1 and p[1] != 'spanner' and p[1] != 'nut']: # Assuming the first object of type man in initial state is the man
                     man_location = loc
            elif parts[0] == 'loose' and parts[1] in self.goal_nuts:
                 nut = parts[1]
                 # Need to find the location of this nut in the current state
                 if nut in current_locations:
                     loose_goal_nuts_in_state[nut] = current_locations[nut]
            elif parts[0] == 'usable':
                 spanner = parts[1]
                 # Need to find the location of this spanner in the current state
                 if spanner in current_locations:
                     usable_spanners_in_state[spanner] = current_locations[spanner]
            elif parts[0] == 'carrying':
                 carrier, spanner = parts[1], parts[2]
                 # Assuming the carrier is the man
                 if match(fact, "carrying", "*", "*") and carrier == [p[1] for p in [get_parts(f) for f in task.initial_state] if p[0] == 'at' and len(p) > 1 and p[1] != 'spanner' and p[1] != 'nut'][0]:
                     # Check if the carried spanner is usable in this state
                     if "(usable {})".format(spanner) in state:
                         man_carrying_spanner = spanner
                         # The location of the carried spanner is the man's location
                         usable_spanners_in_state[spanner] = man_location


        num_loose_goal_nuts = len(loose_goal_nuts_in_state)

        # 2. If no loose goal nuts, heuristic is 0.
        if num_loose_goal_nuts == 0:
            return 0

        # 6. Check solvability based on spanners
        num_usable_spanners = len(usable_spanners_in_state)
        if num_loose_goal_nuts > num_usable_spanners:
             # Not enough usable spanners to tighten all goal nuts
             return float('inf') # Problem is unsolvable from this state

        # 7. Initialize total heuristic cost
        total_cost = 0

        # 8. Calculate cost for each loose goal nut
        for nut, nut_location in loose_goal_nuts_in_state.items():
            # a. Add 1 for the tighten_nut action
            cost_this_nut = 1

            # b. Calculate cost to get man to nut_location carrying a usable spanner
            cost_to_get_man_with_spanner_to_nut = float('inf')

            if man_carrying_spanner is not None:
                # Man is already carrying a usable spanner, just needs to walk to the nut
                cost_to_get_man_with_spanner_to_nut = self.get_distance(man_location, nut_location)
            else:
                # Man is not carrying a spanner, needs to fetch one
                min_fetch_and_travel_cost = float('inf')
                for spanner, spanner_location in usable_spanners_in_state.items():
                    # Cost to go from man's location to spanner, pickup, then go to nut
                    # Ensure spanner is not the one the man might be carrying (already handled by man_carrying_spanner check)
                    # Ensure spanner is not already used for another nut calculation in this sum (additive heuristic ignores this)
                    # Ensure locations are valid and reachable
                    if man_location in self.distances and spanner_location in self.distances[man_location] and nut_location in self.distances[spanner_location]:
                         fetch_and_travel_cost = self.get_distance(man_location, spanner_location) + 1 + self.get_distance(spanner_location, nut_location)
                         min_fetch_and_travel_cost = min(min_fetch_and_travel_cost, fetch_and_travel_cost)

                if min_fetch_and_travel_cost == float('inf'):
                    # Should not happen if num_loose_goal_nuts <= num_usable_spanners,
                    # unless some locations are unreachable.
                    return float('inf') # Cannot get a spanner to this nut
                cost_to_get_man_with_spanner_to_nut = min_fetch_and_travel_cost

            # Add the cost to get man with spanner to the nut location
            if cost_to_get_man_with_spanner_to_nut == float('inf'):
                 return float('inf') # Cannot reach this nut location with a spanner

            cost_this_nut += cost_to_get_man_with_spanner_to_nut
            total_cost += cost_this_nut

        # 9. Return total heuristic
        return total_cost

# Example usage (assuming 'task' object is available)
# heuristic = spannerHeuristic(task)
# h_value = heuristic(current_node)
