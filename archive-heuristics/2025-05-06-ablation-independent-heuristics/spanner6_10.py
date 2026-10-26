# Import necessary modules
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """Parses a PDDL fact string like '(predicate arg1 arg2)' into a list ['predicate', 'arg1', 'arg2']."""
    # Remove surrounding brackets and split by space
    parts = fact_str[1:-1].split()
    return parts

# Helper function for BFS to find shortest paths
def bfs(graph, start_node):
    """Performs BFS starting from start_node on the graph to find shortest distances."""
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        if current_node in graph: # Handle nodes with no links
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal (tighten all required nuts) by summing
    the estimated number of tighten actions, pickup actions, and walking actions.
    It calculates the number of loose goal nuts (tighten actions), the number
    of additional usable spanners needed (pickup actions), and estimates the
    walking cost using a Nearest Neighbor approach to visit nut locations and
    selected spanner pickup locations.

    Assumptions:
    - Nuts are static at their initial locations.
    - Links between locations are bidirectional.
    - There is only one man object.
    - Spanners are objects that can be 'usable', 'at' locations, or 'carrying'.
    - The 'usable' predicate is only removed by the 'tighten_nut' action.
    - The man can carry multiple spanners.
    - All locations mentioned in initial state, goals, or static links are part of the graph.

    Heuristic Initialization:
    1.  Parses static facts to build a graph of locations based on 'link' predicates.
    2.  Identifies all unique locations in the domain from static links, initial state, and goals.
    3.  Computes all-pairs shortest paths between all identified locations using BFS. Stores distances in `self.location_distances`.
    4.  Identifies the man object by looking for the first argument in 'carrying' facts or the single non-nut object in 'at' facts in the initial state, falling back to 'bob' if necessary. Stores in `self.man_obj`.
    5.  Identifies all spanner objects present in the initial state based on 'at', 'usable', or 'carrying' facts, excluding the man and goal nuts. Stores in `self.initial_spanners`.
    6.  Identifies the static location for each goal nut from the initial state. Stores in `self.nut_location`.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to find:
        -   The man's current location (`man_loc`).
        -   The set of spanners the man is currently carrying (`carried_spanners`).
        -   The set of names of spanners that are currently usable (`usable_spanner_names_in_state`).
        -   The set of goal nuts that are currently loose (`loose_goal_nuts`).
        -   A list of all spanners currently at locations (`spanners_at_locs_list`).
    2.  If `man_loc` is not found, return `float('inf')`.
    3.  Calculate `k`, the number of loose goal nuts (`k = len(loose_goal_nuts)`).
    4.  If `k == 0`, the goal is reached, return 0.
    5.  Filter `spanners_at_locs_list` to get only usable spanners at locations (`usable_spanners_at_locs_list`).
    6.  Count usable spanners the man is currently carrying (`carried_usable_now`).
    7.  Calculate `k_pickup`, the number of additional usable spanners the man needs to pick up (`k_pickup = max(0, k - carried_usable_now)`).
    8.  If `k_pickup` is greater than the total number of usable spanners available at locations, return `float('inf')` (dead end).
    9.  Select the `k_pickup` usable spanners at locations that are closest to `man_loc`.
    10. Identify the set of locations to visit: locations of loose goal nuts (`nut_locs`) and locations of selected spanners (`pickup_spanner_locs`). Combine into `required_locations`.
    11. Calculate the estimated walking cost from `man_loc` to visit all `required_locations` using the Nearest Neighbor TSP heuristic (`walk_cost`).
    12. If `walk_cost` is `float('inf')`, return `float('inf')` (required locations are unreachable).
    13. The heuristic value is `k + k_pickup + walk_cost`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static = task.static

        # --- Precomputation ---

        self.location_graph = {}
        locations = set()

        # Parse static links and build graph
        for fact_str in self.static:
            if fact_str.startswith('(link '):
                _, l1, l2 = parse_fact(fact_str)
                locations.add(l1)
                locations.add(l2)
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1) # Links are bidirectional

        # Add locations from initial state and goals to ensure they are in the graph keys
        for state_set in [self.initial_state, self.goals]:
             for fact_str in state_set:
                  if fact_str.startswith('(at '):
                      _, obj, loc = parse_fact(fact_str)
                      locations.add(loc)
                      self.location_graph.setdefault(loc, []) # Ensure all locations are keys

        # Compute all-pairs shortest paths
        self.location_distances = {}
        all_locations = list(self.location_graph.keys())
        for start_loc in all_locations:
            self.location_distances[start_loc] = bfs(self.location_graph, start_loc)

        # Identify man object (assuming only one man)
        self.man_obj = None
        goal_nuts = {parse_fact(g)[1] for g in self.goals if g.startswith('(tightened ')}
        potential_men = set()
        for fact_str in self.initial_state:
            parts = parse_fact(fact_str)
            if parts[0] == 'carrying':
                self.man_obj = parts[1] # Found man via carrying fact
                break
            elif parts[0] == 'at' and parts[1] not in goal_nuts:
                 potential_men.add(parts[1])

        if self.man_obj is None and len(potential_men) == 1:
             self.man_obj = potential_men.pop() # Assume the single non-nut object at a location is the man
        elif self.man_obj is None:
             # Fallback: Assume 'bob' based on examples if no other man identified
             self.man_obj = 'bob' # Less robust, but necessary if identification fails

        # Identify all spanner objects present initially
        self.initial_spanners = set()
        for fact_str in self.initial_state:
            parts = parse_fact(fact_str)
            if parts[0] == 'at' and parts[1] != self.man_obj and parts[1] not in goal_nuts:
                 self.initial_spanners.add(parts[1])
            elif parts[0] == 'usable':
                 self.initial_spanners.add(parts[1])
            elif parts[0] == 'carrying' and parts[1] == self.man_obj:
                 self.initial_spanners.add(parts[2])


        # Identify nut objects and their static locations
        self.nut_location = {}
        for nut in goal_nuts:
            # Find the initial location of this nut
            found_loc = None
            for fact_str in self.initial_state:
                if fact_str.startswith('(at '):
                    _, obj, loc = parse_fact(fact_str)
                    if obj == nut:
                        found_loc = loc
                        break
            if found_loc:
                self.nut_location[nut] = found_loc
            # else: # Goal nut has no initial location - indicates a potentially malformed problem
            #     pass # Heuristic will likely return inf later if nut location is needed


    def calculate_walk_cost(self, start_loc, required_locs):
        """Calculates estimated walk cost using Nearest Neighbor TSP heuristic."""
        if not required_locs:
            return 0

        current_loc = start_loc
        remaining_locs = set(required_locs)
        walk_cost = 0

        while remaining_locs:
            nearest_loc = None
            min_dist = float('inf')

            # Find the nearest required location from the current location
            for loc in remaining_locs:
                # Use precomputed distances, handle unreachable locations
                dist = self.location_distances.get(current_loc, {}).get(loc, float('inf'))
                if dist == float('inf'):
                    # If any required location is unreachable from the current path, the whole set is unreachable
                    return float('inf')
                if dist < min_dist:
                    min_dist = dist
                    nearest_loc = loc

            # If nearest_loc is still None, it means remaining_locs was not empty but no reachable location was found.
            # This should be caught by the dist == float('inf') check inside the loop, but as a safeguard:
            if nearest_loc is None:
                 return float('inf')


            walk_cost += min_dist
            current_loc = nearest_loc
            remaining_locs.remove(nearest_loc)

        return walk_cost


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        man_loc = None
        carried_spanners = set()
        loose_goal_nuts = set()
        usable_spanner_names_in_state = set()
        spanners_at_locs_list = [] # List of (spanner_name, location) for spanners currently at locations

        # Parse state facts to find key information
        for fact_str in state:
            parts = parse_fact(fact_str)
            predicate = parts[0]

            if predicate == 'at':
                obj, loc = parts[1], parts[2]
                if obj == self.man_obj:
                    man_loc = loc
                # Collect all objects at locations initially identified as spanners
                if obj in self.initial_spanners:
                     spanners_at_locs_list.append((obj, loc))

            elif predicate == 'carrying':
                # Assume the second part is the object being carried by the man
                carried_spanners.add(parts[2])

            elif predicate == 'loose':
                nut = parts[1]
                # Check if this loose nut is one of the goal nuts
                if f'(tightened {nut})' in self.goals:
                     loose_goal_nuts.add(nut)

            elif predicate == 'usable':
                 usable_spanner_names_in_state.add(parts[1])

        # If man_loc is not found, the state is malformed or represents an impossible situation
        # Returning infinity is appropriate for a greedy search heuristic in such cases.
        if man_loc is None:
             return float('inf')

        k = len(loose_goal_nuts)

        # If all goal nuts are tightened, the goal is reached
        if k == 0:
            return 0

        # Filter spanners at locations to find only usable ones
        usable_spanners_at_locs_list = [(s, l) for s, l in spanners_at_locs_list if s in usable_spanner_names_in_state]

        # Count usable spanners the man is currently carrying
        carried_usable_now = len([s for s in carried_spanners if s in usable_spanner_names_in_state])

        # Calculate how many more usable spanners are needed
        k_pickup = max(0, k - carried_usable_now)

        # Check if enough usable spanners exist in the world (at locations)
        if k_pickup > len(usable_spanners_at_locs_list):
            # Not enough usable spanners available to tighten all remaining nuts
            return float('inf')

        # Select k_pickup spanners to pick up. Greedily choose those closest to the man's current location.
        # Sort available spanners by distance from man_loc
        # Handle case where man_loc might not be in location_distances (e.g., malformed state)
        man_distances = self.location_distances.get(man_loc, {})
        usable_spanners_at_locs_list.sort(key=lambda item: man_distances.get(item[1], float('inf')))

        selected_spanners_info = usable_spanners_at_locs_list[:k_pickup]
        pickup_spanner_locs = {l for (s, l) in selected_spanners_info}

        # Identify the locations of the loose goal nuts
        # Ensure nut location was found in init, otherwise skip this nut (problematic state)
        nut_locs = {self.nut_location[n] for n in loose_goal_nuts if n in self.nut_location}

        # Combine nut locations and selected spanner pickup locations
        required_locations = nut_locs | pickup_spanner_locs

        # Calculate the estimated walking cost to visit all required locations
        walk_cost = self.calculate_walk_cost(man_loc, required_locations)

        # If walking cost is infinity, it means some required location is unreachable
        if walk_cost == float('inf'):
             return float('inf')

        # The heuristic value is the sum of estimated actions:
        # k tighten actions + k_pickup pickup actions + estimated walk actions
        h_value = k + k_pickup + walk_cost

        return h_value
