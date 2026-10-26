import collections
import math
from fnmatch import fnmatch

# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing
# Remove this section when integrating into a planner environment
class Heuristic:
    def __init__(self, task):
        self.task = task
    def __call__(self, node):
        raise NotImplementedError

# End of dummy class


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
    # Check if the number of parts is at least the number of args
    if len(parts) < len(args):
         return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all goal nuts by summing the
    estimated minimum cost for each individual loose goal nut. The cost for
    a single nut includes the tighten action itself, plus the estimated cost
    for the man to acquire a usable spanner and travel to the nut's location.

    # Assumptions
    - There is only one man ('bob').
    - Each usable spanner can tighten exactly one nut.
    - The man can carry at most one spanner at a time.
    - The location graph defined by 'link' predicates is static.
    - Travel cost between linked locations is 1.

    # Heuristic Initialization
    - Build the location graph from 'link' static facts.
    - Compute all-pairs shortest path distances between all locations using BFS.
    - Identify the set of goal nuts from the task definition.
    - Map goal nuts to their locations (assumed static from initial state).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all nuts that are goals and are currently loose in the state.
    2. If there are no loose goal nuts, the state is a goal state, and the heuristic is 0.
    3. Find the current location of the man ('bob').
    4. Determine if the man is currently carrying a usable spanner.
    5. Identify all usable spanners that are currently at a location (not carried by the man) and their locations.
    6. Initialize the total heuristic cost to 0.
    7. For each loose goal nut N at location L_N:
       a. Add 1 to the total cost (for the 'tighten_nut' action).
       b. Calculate the minimum cost for the man to get a usable spanner and reach L_N:
          i. If the man is already carrying a usable spanner: The cost is the shortest path distance from the man's current location to L_N.
          ii. If the man is not carrying a usable spanner: Find the minimum cost among all available usable spanners. For each available usable spanner S_a at location L_Sa, the cost to get it to L_N via the man is:
             - Shortest path distance from the man's current location to L_Sa (walk to spanner).
             - Plus 1 (for the 'pickup_spanner' action).
             - Plus shortest path distance from L_Sa to L_N (walk from spanner location to nut location).
             Take the minimum of these costs over all available usable spanners. If no usable spanners are available or reachable, this cost is infinite.
       c. Add the calculated minimum spanner-acquisition-and-travel cost to the total heuristic cost.
       d. If the cost calculated in step 7b is infinite for any nut, the total heuristic is infinite (unsolvable state).
    8. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the location graph and computing distances."""
        super().__init__(task)

        self.locations = set()
        self.graph = collections.defaultdict(list)

        # Collect all locations mentioned in initial state and static facts
        all_facts = list(task.initial_state) + list(task.static)
        for fact in all_facts:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 # The third part is the location
                 self.locations.add(parts[2])
             if parts[0] == 'link' and len(parts) == 3:
                 # The second and third parts are locations
                 self.locations.add(parts[1])
                 self.locations.add(parts[2])
                 # Add directed link
                 self.graph[parts[1]].append(parts[2])


        # Compute all-pairs shortest paths
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node)

        # Identify goal nuts and their locations (assuming nut locations are static)
        self.goal_nuts = set()
        self.goal_nut_locations = {}
        initial_nut_locations = {}

        # Find initial locations of all nuts
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assuming objects starting with 'nut' are nuts based on domain example
                if obj.startswith('nut'):
                    initial_nut_locations[obj] = loc

        # Find goal nuts from task goals
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened' and len(parts) == 2:
                nut = parts[1]
                self.goal_nuts.add(nut)
                # Get the location of the goal nut from the initial state
                if nut in initial_nut_locations:
                    self.goal_nut_locations[nut] = initial_nut_locations[nut]
                else:
                    # This indicates a problem with the instance definition
                    print(f"Error: Goal nut {nut} not found in initial state locations.")
                    # We cannot proceed meaningfully if a goal nut has no location
                    raise ValueError(f"Goal nut {nut} has no initial location.")


    def _bfs(self, start_node):
        """Perform BFS from a start node to find distances to all reachable nodes."""
        distances = {loc: float('inf') for loc in self.locations}
        if start_node not in self.locations:
             # Start node is not a known location, cannot compute distances
             return distances

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Ensure current_node is in graph keys before accessing neighbors
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, l1, l2):
        """Get the shortest distance between two locations."""
        if l1 not in self.locations or l2 not in self.locations:
             # One or both locations are not in the known graph
             return float('inf')
        if l1 == l2:
            return 0
        # Distances dictionary is distances[start_node][end_node]
        return self.distances[l1].get(l2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose goal nuts in the current state
        loose_goal_nuts_in_state = set()
        current_nut_locations = {}
        current_spanner_locations = {}
        current_usable_spanners = set()
        current_carried_spanner = None
        bob_location = None

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                # Assuming 'bob' is the man based on example
                if obj == 'bob':
                    bob_location = loc
                elif obj.startswith('nut'):
                    current_nut_locations[obj] = loc
                elif obj.startswith('spanner'):
                    current_spanner_locations[obj] = loc
            elif parts[0] == 'loose':
                nut = parts[1]
                if nut in self.goal_nuts:
                    loose_goal_nuts_in_state.add(nut)
            elif parts[0] == 'usable':
                spanner = parts[1]
                current_usable_spanners.add(spanner)
            elif parts[0] == 'carrying':
                 man, spanner = parts[1], parts[2]
                 if man == 'bob': # Assuming 'bob' is the man
                     current_carried_spanner = spanner

        # If bob's location is unknown, the state is likely invalid or unsolvable
        if bob_location is None:
             return float('inf')


        # 2. If no loose goal nuts, it's a goal state
        if not loose_goal_nuts_in_state:
            return 0

        # 5. Identify available usable spanners (at locations, not carried)
        available_usable_spanners_at_locs = set() # Stores (spanner, location)
        for spanner in current_usable_spanners:
            # Check if the usable spanner is NOT the one bob is carrying
            if spanner != current_carried_spanner:
                 # Check if the usable spanner is at a known location in the state
                 if spanner in current_spanner_locations:
                      available_usable_spanners_at_locs.add((spanner, current_spanner_locations[spanner]))
                 # Note: If a usable spanner is not at a location and not carried, it's unreachable.

        # 6. Initialize total heuristic cost
        total_cost = 0

        # 7. For each loose goal nut
        for nut in loose_goal_nuts_in_state:
            # Nut location is assumed static from initial state
            L_N = self.goal_nut_locations.get(nut)
            # We already checked in __init__ that goal nuts have initial locations,
            # and nut locations don't change in this domain.

            # a. Add cost for tighten action
            cost_for_this_nut = 1

            # b. Calculate cost to get bob + spanner to L_N
            bob_has_usable_spanner = (current_carried_spanner is not None) and (current_carried_spanner in current_usable_spanners)

            travel_pickup_cost = float('inf')

            if bob_has_usable_spanner:
                # Bob has a usable spanner, just need to walk to the nut
                travel_cost = self.get_distance(bob_location, L_N)
                if travel_cost != float('inf'):
                    travel_pickup_cost = travel_cost
            else:
                # Bob needs to get a spanner first
                min_travel_pickup_cost_for_nut = float('inf')
                for (S_a, L_Sa) in available_usable_spanners_at_locs:
                    dist_bob_to_spanner = self.get_distance(bob_location, L_Sa)
                    dist_spanner_to_nut = self.get_distance(L_Sa, L_N)

                    if dist_bob_to_spanner != float('inf') and dist_spanner_to_nut != float('inf'):
                        # Cost = walk to spanner + pickup + walk to nut
                        current_travel_pickup_cost = dist_bob_to_spanner + 1 + dist_spanner_to_nut
                        min_travel_pickup_cost_for_nut = min(min_travel_pickup_cost_for_nut, current_travel_pickup_cost)

                travel_pickup_cost = min_travel_pickup_cost_for_nut

            # c. Add travel/pickup cost to nut cost
            if travel_pickup_cost == float('inf'):
                 # If it's impossible to get a spanner to this nut, the state is unsolvable
                 return float('inf')

            cost_for_this_nut += travel_pickup_cost

            # Add cost for this nut to total
            total_cost += cost_for_this_nut

        # 8. Return total heuristic cost
        return total_cost
