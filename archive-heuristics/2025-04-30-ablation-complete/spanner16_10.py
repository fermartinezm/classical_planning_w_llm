from collections import deque
# Assuming Heuristic base class is available like this
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper functions for graph and distances
def bfs(graph, start_node):
    """Performs BFS to find shortest distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

def build_location_graph(static_facts):
    """Builds an undirected graph of locations based on 'link' facts."""
    graph = {}
    locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'link':
            l1, l2 = parts[1], parts[2]
            locations.add(l1)
            locations.add(l2)
            graph.setdefault(l1, []).append(l2)
            graph.setdefault(l2, []).append(l1) # Links are bidirectional
    # Ensure all locations mentioned in links are in the graph dict
    for loc in locations:
         graph.setdefault(loc, [])
    return graph, list(locations) # Return locations list for BFS starting points

def precompute_distances(graph, locations):
    """Precomputes all-pairs shortest paths using BFS."""
    all_distances = {}
    for start_loc in locations:
        all_distances[start_loc] = bfs(graph, start_loc)
    return all_distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the total number of actions required to tighten all
    goal nuts. It sums the estimated cost for each loose goal nut independently.
    The estimated cost for a single loose nut includes the 'tighten_nut' action
    plus the estimated cost to get the man to the nut's location while carrying
    a usable spanner.

    # Assumptions:
    - The man can carry multiple spanners simultaneously. (Based on example state representation)
    - A spanner becomes unusable only after successfully tightening a nut.
    - Nuts do not move from their initial locations.
    - All nuts specified in the goal must be tightened.
    - There are always enough usable spanners available (either carried or on the ground)
      to tighten all loose goal nuts in solvable problems.
    - The location graph is connected such that all necessary locations (man start,
      spanner locations, nut locations) are reachable from each other.

    # Heuristic Initialization
    - Identifies the man object name from the initial state.
    - Extracts goal nuts and their static locations from the initial state.
    - Builds the location graph from 'link' static facts.
    - Precomputes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the state to determine:
       - The man's current location.
       - The set of spanners the man is currently carrying.
       - The set of all spanners that are currently usable.
       - The locations of spanners that are on the ground.
       - The set of nuts that have already been tightened.
    2. Identify the set of goal nuts that are currently loose (i.e., are in the task's
       goal conditions but not in the set of tightened nuts in the current state).
    3. Initialize the total heuristic cost to 0.
    4. For each loose goal nut `N` at its static location `L_N`:
       a. Add 1 to the cost for this nut (for the `tighten_nut` action).
       b. Calculate the cost to get the man to `L_N` while carrying a usable spanner:
          i. Check if the man is currently carrying any spanner that is also usable.
          ii. If yes, the cost is the shortest distance from the man's current location
              to `L_N`.
          iii. If no, the man needs to pick up a usable spanner from the ground.
               - Find all usable spanners that are currently on the ground and their locations.
               - If no usable spanners are on the ground, the problem is unsolvable
                 from this state; return infinity.
               - Otherwise, calculate the minimum cost among all usable ground spanners `S`
                 at location `L_S`: distance(man's location, `L_S`) + 1 (pickup action) +
                 distance(`L_S`, `L_N`). This minimum cost is the travel cost.
          iv. If the calculated travel cost is infinity (meaning `L_N` or `L_S` is unreachable),
              return infinity.
       c. Add the calculated travel cost to the cost for this nut.
       d. Add the total cost for this nut to the overall heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting task information and precomputing distances.
        """
        # Store initial state to find man and goal nut locations
        self.task_initial_state = task.initial_state
        self.goals = task.goals
        static_facts = task.static

        # Store goal nuts and their static locations (from initial state)
        self.goal_nuts = {}
        self.man_name = None
        spanner_names_in_init = set()
        goal_nut_names = set()

        # First pass: Collect goal nut names and spanner names from initial state
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                goal_nut_names.add(args[0])

        for fact in self.task_initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 if obj in goal_nut_names:
                     self.goal_nuts[obj] = loc
                 # Crude check for spanner name pattern - assumes spanner names start with 'spanner'
                 if obj.startswith('spanner'):
                     spanner_names_in_init.add(obj)
             elif parts[0] == 'carrying':
                 # Man might be carrying a spanner initially
                 spanner_names_in_init.add(parts[2])

        # Second pass: Identify the man object name
        # Assume the man is the only 'at' object in initial state that is not a goal nut or a spanner
        for fact in self.task_initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 if obj not in goal_nut_names and obj not in spanner_names_in_init:
                     self.man_name = obj
                     break # Found the man

        if self.man_name is None:
             # Fallback/Warning if man not found by the above logic
             # This heuristic might fail or be inaccurate without the man's name.
             # In a real system, robust parsing would be needed.
             print("Warning: Could not identify the man object name from initial state.")
             # Set to a dummy value to avoid errors later, though heuristic will be wrong
             self.man_name = "unknown_man"


        # Build location graph and precompute distances
        self.location_graph, self.locations = build_location_graph(static_facts)
        self.distances = precompute_distances(self.location_graph, self.locations)

        # Optional: Check if all necessary locations (goal nut locations, initial man location) are in the graph
        # This helps debug problem files or graph building
        all_locations_in_graph = set(self.locations)
        if self.man_name != "unknown_man": # Check only if man name was found
            man_initial_loc = None
            for fact in self.task_initial_state:
                 parts = get_parts(fact)
                 if parts[0] == 'at' and parts[1] == self.man_name:
                      man_initial_loc = parts[2]
                      break
            if man_initial_loc and man_initial_loc not in all_locations_in_graph:
                 print(f"Warning: Man's initial location {man_initial_loc} not found in location graph.")

        for nut, loc in self.goal_nuts.items():
             if loc not in all_locations_in_graph:
                  print(f"Warning: Goal nut location {loc} for {nut} not found in location graph.")
                  # This will correctly result in infinite distance later if unreachable.


    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returns infinity if locations are not in graph or unreachable."""
        # Ensure both locations are valid nodes in our distance map
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse state to get relevant facts
        man_location = None
        carried_spanners = set()
        usable_spanners_all = set()
        spanner_locations_on_ground = {} # For spanners not carried
        tightened_nuts = set()

        # Need man's name to parse state correctly
        if self.man_name == "unknown_man":
             # Cannot compute heuristic without man's name identified
             return float('inf') # Indicate unsolvable or setup issue

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj == self.man_name:
                     man_location = loc
                elif obj.startswith('spanner'): # Crude check for spanner
                     spanner_locations_on_ground[obj] = loc
            elif parts[0] == 'carrying':
                m, s = parts[1], parts[2]
                if m == self.man_name:
                    carried_spanners.add(s)
            elif parts[0] == 'usable':
                s = parts[1]
                usable_spanners_all.add(s)
            elif parts[0] == 'tightened':
                n = parts[1]
                tightened_nuts.add(n)

        # If man_location wasn't found in the state, something is wrong
        if man_location is None:
             print(f"Error: Man '{self.man_name}' location not found in state.")
             return float('inf') # Indicate invalid state or parsing error


        total_cost = 0

        # Identify usable spanners currently carried by the man
        usable_carried_spanners = carried_spanners.intersection(usable_spanners_all)

        # Identify usable spanners on the ground and their locations
        usable_ground_spanner_locs = {
            s: loc for s, loc in spanner_locations_on_ground.items()
            if s in usable_spanners_all
        }

        # Identify goal nuts that are currently loose
        loose_goal_nuts = {
            nut_name for nut_name in self.goal_nuts
            if nut_name not in tightened_nuts
        }

        # If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts:
            return 0

        # Estimate cost for each loose goal nut
        for nut_name in loose_goal_nuts:
            # Get the static location of the nut
            nut_location = self.goal_nuts.get(nut_name)
            if nut_location is None:
                 # This nut is a goal nut but its location wasn't found in initial state?
                 # Should not happen if __init__ works correctly.
                 print(f"Error: Location for goal nut '{nut_name}' not found.")
                 return float('inf')

            # Base cost for tightening the nut
            cost_for_this_nut = 1 # tighten_nut action

            # Cost to get man and a usable spanner to the nut's location
            cost_to_get_spanner_and_reach_nut = float('inf')

            # Case 1: Man is already carrying a usable spanner
            if usable_carried_spanners:
                 # He can just walk to the nut's location
                 travel_cost = self.get_distance(man_location, nut_location)
                 if travel_cost != float('inf'):
                     cost_to_get_spanner_and_reach_nut = travel_cost
            else:
                 # Case 2: Man needs to pick up a usable spanner from the ground
                 if not usable_ground_spanner_locs:
                     # No usable spanners available anywhere on the ground
                     # If he's not carrying one either (checked above), this nut cannot be tightened.
                     return float('inf') # Indicate unsolvable subproblem

                 # Find the minimum cost to go to a usable spanner, pick it up, and go to the nut
                 min_travel_pickup_cost = float('inf')
                 for spanner_name, spanner_loc in usable_ground_spanner_locs.items():
                     # Cost = travel from man to spanner + pickup action + travel from spanner to nut
                     travel_to_spanner = self.get_distance(man_location, spanner_loc)
                     travel_from_spanner_to_nut = self.get_distance(spanner_loc, nut_location)

                     # Ensure both legs of the journey are possible
                     if travel_to_spanner != float('inf') and travel_from_spanner_to_nut != float('inf'):
                         cost = travel_to_spanner + 1 + travel_from_spanner_to_nut
                         min_travel_pickup_cost = min(min_travel_pickup_cost, cost)

                 # If no reachable usable spanner on the ground allows reaching the nut
                 if min_travel_pickup_cost == float('inf'):
                      return float('inf') # Indicate unsolvable subproblem

                 cost_to_get_spanner_and_reach_nut = min_travel_pickup_cost

            # Add the cost to get spanner and reach nut to the nut's total cost
            # This cost should not be inf if we passed the checks above, but add defensively
            if cost_to_get_spanner_and_reach_nut == float('inf'):
                 return float('inf')

            cost_for_this_nut += cost_to_get_spanner_and_reach_nut
            total_cost += cost_for_this_nut

        return total_cost
