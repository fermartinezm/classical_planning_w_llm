import collections
import sys

# Define a large number for unsolvable states
UNSOLVABLE_HEURISTIC = 1000000

# Helper functions
def parse_fact(fact_str):
    """Parses a PDDL fact string into a list of strings."""
    # Remove parentheses and split by space
    parts = fact_str.strip("()").split()
    return parts

def bfs(start_node, graph):
    """Computes shortest distances from start_node to all other nodes in graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph (e.g., isolated location)
         # Distances to other nodes remain inf.
         return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Check if current_node is in graph keys before iterating
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_distances(nodes, graph):
    """Computes shortest distances between all pairs of nodes."""
    all_distances = {}
    for start_node in nodes:
        all_distances[start_node] = bfs(start_node, graph)
    return all_distances

# Main heuristic class
class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Estimates the cost to reach a goal state by considering the number
    of remaining nuts to tighten, the travel cost for the man to reach
    a nut location, and the cost to acquire a usable spanner if needed.
    """

    def __init__(self, task):
        """
        Initializes the spanner heuristic.

        Precomputes static information like location graph, distances,
        goal nuts, their locations, and the man's name.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        # Heuristic Initialization:
        # The constructor performs precomputation that is constant for a given task
        # but varies between different tasks (problem instances).
        # This includes parsing the goal to identify target nuts, building
        # the graph of locations based on 'link' facts, computing shortest
        # path distances between all pairs of locations, identifying the man
        # object, and finding the static locations of the goal nuts.

        self.task = task
        self.goal_nuts = set()
        self.locations = set()
        self.location_graph = collections.defaultdict(list)
        self.distances = {}
        self.nut_locations = {}
        self.man_name = None # Assuming there's only one man

        # 1. Parse goal nuts from task.goals
        # task.goals is a frozenset of goal facts, e.g., frozenset({'(tightened nut1)', '(tightened nut2)'})
        for goal_fact in task.goals:
            parts = parse_fact(goal_fact)
            if parts and parts[0] == 'tightened' and len(parts) > 1:
                self.goal_nuts.add(parts[1])

        # Collect all facts from initial state and static information
        all_facts = set(task.initial_state) | set(task.static)

        # 2. Build location graph and find all locations
        # Locations are objects involved in 'link' facts or as the second argument in 'at' facts.
        for fact_str in all_facts:
            parts = parse_fact(fact_str)
            if not parts: continue # Skip empty facts if any

            if parts[0] == 'link':
                if len(parts) == 3:
                    loc1, loc2 = parts[1], parts[2]
                    self.location_graph[loc1].append(loc2)
                    self.location_graph[loc2].append(loc1)
                    self.locations.add(loc1)
                    self.locations.add(loc2)
            elif parts[0] == 'at':
                 if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    self.locations.add(loc)

        # 4. Find man's name
        # Assuming task.objects is available and contains type information like "obj_name - type_name"
        # If task.objects is not available or doesn't have types, this part is fragile.
        # Let's try to parse task.objects first.
        man_objects = []
        if hasattr(task, 'objects') and task.objects:
             # task.objects is often a list of strings like "obj_name - type_name"
             # Need to handle potential parsing errors or different formats
             try:
                 # Split by ' - ' and take the first part (name) if ' - man' is present
                 man_objects = [obj_str.split(' - ')[0] for obj_str in task.objects if ' - man' in obj_str]
             except Exception as e:
                 # Handle cases where task.objects format is unexpected
                 # print(f"Warning: Could not parse task.objects for man name: {e}") # Keep print for debugging if needed
                 man_objects = [] # Fallback to initial state check

        if man_objects:
            self.man_name = man_objects[0] # Assuming there's only one man
        else:
            # Fallback: Try to infer from initial state 'at' facts
            # Find the first object that is 'at' a location in the initial state.
            for fact_str in task.initial_state:
                parts = parse_fact(fact_str)
                if parts and parts[0] == 'at' and len(parts) == 3:
                    # This is a weak assumption, but common in simple domains
                    self.man_name = parts[1]
                    break # Assume the first object at a location in initial state is the man

        if self.man_name is None:
            # This indicates a problem with the task object or assumptions
            # print("Error: Could not determine man's name. Heuristic may fail.") # Keep print for debugging if needed
            # Heuristic might not work correctly without man's name.
            # Could raise an error or return a large value in __call__.
            pass # Let's proceed, hoping man_name is found in typical spanner problems.


        # 3. Compute all-pairs shortest paths
        # Ensure all locations found are included in the graph nodes for BFS
        all_nodes_for_bfs = list(self.locations)
        self.distances = compute_all_pairs_distances(all_nodes_for_bfs, self.location_graph)

        # 5. Find and store locations of goal nuts (assuming nuts are static)
        for nut in self.goal_nuts:
            found_loc = False
            # Check initial state first
            for fact_str in task.initial_state:
                parts = parse_fact(fact_str)
                if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] == nut:
                    self.nut_locations[nut] = parts[2]
                    found_loc = True
                    break
            # If not in initial state, check static facts
            if not found_loc:
                 for fact_str in task.static:
                    parts = parse_fact(fact_str)
                    if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] == nut:
                        self.nut_locations[nut] = parts[2]
                        found_loc = True
                        break
            # If nut location not found, this nut cannot be tightened.
            # This might indicate an unsolvable problem or a nut not relevant to the goal.
            # For goal nuts, their location *must* be defined.
            if not found_loc:
                 # print(f"Error: Location for goal nut {nut} not found in initial or static state.") # Keep print for debugging if needed
                 # This problem instance is likely malformed or unsolvable.
                 # The heuristic will likely return a large value later if this nut is loose.
                 # We could remove this nut from goal_nuts, but that changes the goal.
                 # Let's keep it and let the distance calculation handle unreachable locations (inf).
                 pass


    def __call__(self, state):
        """
        Computes the domain-dependent heuristic for the spanner domain.

        Args:
            state: A frozenset of facts representing the current state.

        Returns:
            An integer or float representing the estimated cost to reach the goal.
            Returns a large number if the state is estimated to be unsolvable.

        Step-By-Step Thinking for Computing Heuristic:
        1. Identify the set of goal nuts that are currently in a 'loose' state.
           In this domain, a nut is either 'loose' or 'tightened'. If a goal nut
           is not 'tightened' in the current state, it is considered 'loose'.
        2. If this set of loose goal nuts is empty, all goal nuts are tightened,
           and the heuristic value is 0 (goal state).
        3. Determine the man's current location from the state facts.
        4. Get the precomputed locations for all goal nuts. Filter these to get locations
           of the currently loose goal nuts.
        5. Calculate the shortest distance from the man's current location to each
           location containing a loose goal nut, and find the minimum of these distances.
           This identifies the nearest location where a tightening action is needed.
           If the man's location or any loose goal nut location is not in the
           precomputed distance map (e.g., disconnected graph), the state is
           unsolvable, return a large value.
        6. Check if the man is currently carrying any spanner that is also marked as 'usable'
           in the current state.
        7. Calculate the estimated cost to get the man to the nearest loose goal nut
           location *while possessing a usable spanner*.
           - If the man is already carrying a usable spanner: The cost is simply the
             shortest distance from his current location to the nearest loose goal nut location.
             If this location is unreachable, return a large value.
           - If the man is not carrying a usable spanner: He needs to acquire one.
             Find the nearest location in the state that contains a spanner marked as 'usable'.
             The cost to acquire the spanner is the distance from his current location
             to that spanner's location plus 1 (for the pickup action).
             After picking up the spanner, the man is at the spanner's location. The cost
             to then reach the nearest loose goal nut location is the distance from the
             spanner's location to the nearest loose goal nut location.
             The total cost in this case is (distance to nearest usable spanner + 1) + (distance from spanner location to nearest loose goal nut location).
             If no usable spanner exists anywhere in the state (neither carried nor at a location),
             or if the necessary locations are unreachable, the problem is likely
             unsolvable from this state, and a large heuristic value is returned.
        8. The final heuristic value is the sum of the number of loose goal nuts
           (representing the minimum number of tightening actions required) and the
           estimated cost calculated in step 7 (representing the travel and pickup
           cost to enable the *first* tightening action). This is an additive heuristic
           that estimates the cost of the remaining tightening actions plus the setup
           cost for the next immediate action.
        """

        loose_goal_nuts = set()
        usable_spanners_in_state = set()
        carrying_spanners = set()
        man_loc = None
        spanner_locations_in_state = {} # {spanner_name: location}
        tightened_nuts_in_state = set()

        # Parse state to find relevant facts
        for fact_str in state:
            parts = parse_fact(fact_str)
            if not parts: continue

            pred = parts[0]
            if pred == 'tightened' and len(parts) == 2:
                 tightened_nuts_in_state.add(parts[1])
            elif pred == 'usable' and len(parts) == 2:
                usable_spanners_in_state.add(parts[1])
            elif pred == 'carrying' and len(parts) == 3 and parts[1] == self.man_name:
                carrying_spanners.add(parts[2])
            elif pred == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 if obj == self.man_name:
                    man_loc = loc
                 # Collect locations of all objects that are not the man and could potentially be spanners
                 # We'll filter by 'usable' later. Assuming any non-man object at a location is a potential spanner.
                 # Nuts are static, their location is precomputed, no need to track their location in state.
                 elif obj != self.man_name and obj not in self.goal_nuts:
                     spanner_locations_in_state[obj] = loc

        # 1. Identify loose goal nuts: goal nuts that are not tightened
        loose_goal_nuts = {nut for nut in self.goal_nuts if nut not in tightened_nuts_in_state}

        # 2. If no loose goal nuts, goal reached
        if len(loose_goal_nuts) == 0:
            return 0

        # Ensure man_loc was found
        if man_loc is None:
             # Man's location not found in state, indicates a problem
             # print("Error: Man's location not found in state.") # Keep print for debugging if needed
             return UNSOLVABLE_HEURISTIC # Treat as unsolvable

        # 4. Locations of loose goal nuts (precomputed in self.nut_locations)
        # Ensure all loose goal nuts have a precomputed location and it's in the distance map
        loose_goal_nut_locs = set()
        for nut in loose_goal_nuts:
             if nut in self.nut_locations:
                  nut_loc = self.nut_locations[nut]
                  if nut_loc in self.locations: # Ensure the location is part of the graph
                     loose_goal_nut_locs.add(nut_loc)
                  else:
                      # Nut location is not in the graph of known locations
                      # print(f"Error: Location {nut_loc} for goal nut {nut} not in location graph.") # Keep print for debugging if needed
                      return UNSOLVABLE_HEURISTIC # Treat as unsolvable
             else:
                  # Location for a loose goal nut was not precomputed, problem with init/static parsing
                  # print(f"Error: Location for loose goal nut {nut} not precomputed.") # Keep print for debugging if needed
                  return UNSOLVABLE_HEURISTIC # Treat as unsolvable

        # If there are loose goal nuts but none have a valid location, it's unsolvable
        if len(loose_goal_nut_locs) == 0:
             return UNSOLVABLE_HEURISTIC


        # 5. Find nearest loose goal nut location from man's current location
        min_dist_to_nut = float('inf')
        nearest_loose_goal_nut_loc = None

        if man_loc in self.distances: # Ensure man_loc is a known location in the graph
            for nut_loc in loose_goal_nut_locs:
                if nut_loc in self.distances.get(man_loc, {}): # Ensure nut_loc is reachable from man_loc
                    dist = self.distances[man_loc][nut_loc]
                    if dist < min_dist_to_nut:
                        min_dist_to_nut = dist
                        nearest_loose_goal_nut_loc = nut_loc

        # If nearest_loose_goal_nut_loc is still None or min_dist_to_nut is inf,
        # it means the man's location is not reachable from any loose goal nut location.
        if nearest_loose_goal_nut_loc is None or min_dist_to_nut == float('inf'):
             return UNSOLVABLE_HEURISTIC # Treat as unsolvable


        # 6. Check if man is carrying a usable spanner
        carrying_usable = any(s in usable_spanners_in_state for s in carrying_spanners)

        # 7. Calculate cost to reach first nut location with spanner
        cost_to_reach_first_nut_with_spanner = 0

        if carrying_usable:
            # Already carrying usable spanner, just need to walk to the nearest nut
            cost_to_reach_first_nut_with_spanner = self.distances[man_loc][nearest_loose_goal_nut_loc]
            # Check if reachable (already done implicitly by the check for nearest_loose_goal_nut_loc)
            # if cost_to_reach_first_nut_with_spanner == float('inf'):
            #      return UNSOLVABLE_HEURISTIC
        else:
            # Need to find and pick up a usable spanner first
            min_dist_to_spanner = float('inf')
            nearest_usable_spanner_loc = None

            # Find nearest usable spanner location
            usable_spanner_at_loc_found_anywhere = False
            for spanner, loc in spanner_locations_in_state.items():
                 if spanner in usable_spanners_in_state:
                     usable_spanner_at_loc_found_anywhere = True # At least one usable spanner exists at a location
                     if man_loc in self.distances and loc in self.distances.get(man_loc, {}):
                        dist = self.distances[man_loc][loc]
                        if dist < min_dist_to_spanner:
                            min_dist_to_spanner = dist
                            nearest_usable_spanner_loc = loc

            if nearest_usable_spanner_loc is None:
                # No usable spanner available at any location *reachable from man_loc*.
                # If there are loose goal nuts, this is unsolvable from here.
                # Note: If usable_spanner_at_loc_found_anywhere is False, it means no usable spanners are at locations at all.
                # If usable_spanner_at_loc_found_anywhere is True but nearest_usable_spanner_loc is None,
                # it means usable spanners exist at locations, but those locations are unreachable from man_loc.
                # In either case, if there are loose goal nuts, the state is likely unsolvable.
                 return UNSOLVABLE_HEURISTIC


            # Cost to get spanner: walk to spanner + pickup action
            cost_to_get_spanner = min_dist_to_spanner + 1

            # Cost to travel from spanner location to the nearest nut location
            # Need to ensure nearest_usable_spanner_loc is connected to nearest_loose_goal_nut_loc
            if nearest_usable_spanner_loc not in self.distances or nearest_loose_goal_nut_loc not in self.distances.get(nearest_usable_spanner_loc, {}):
                 # Spanner location is not connected to nut location
                 return UNSOLVABLE_HEURISTIC

            dist_spanner_to_nut = self.distances[nearest_usable_spanner_loc][nearest_loose_goal_nut_loc]

            if dist_spanner_to_nut == float('inf'):
                 # Spanner location is not reachable from nut location
                 return UNSOLVABLE_HEURISTIC


            cost_to_reach_first_nut_with_spanner = cost_to_get_spanner + dist_spanner_to_nut

        # 8. Heuristic value
        # The number of loose nuts represents the minimum number of tighten actions.
        # The calculated cost is the estimated cost to enable the *first* tighten action.
        # This is an additive heuristic.
        heuristic_value = len(loose_goal_nuts) + cost_to_reach_first_nut_with_spanner

        return heuristic_value
