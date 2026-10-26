import math
from collections import deque

# Assume Heuristic base class is available, e.g., from heuristics.heuristic_base
# If not, define a dummy base class for the code to be syntactically correct.
try:
    # This assumes the heuristic is placed in a file structure
    # where 'heuristics.heuristic_base' is accessible.
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string like '(predicate arg1 arg2)'.
    Removes leading/trailing whitespace and parentheses.
    """
    return fact.strip()[1:-1].split()

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL 'spanner' domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It simulates a greedy strategy where the man agent repeatedly chooses the "cheapest"
    nut-tightening task to perform next. The cost of a task includes the actions:
    walking to a usable spanner, picking it up (if not already carrying one), walking
    to the target loose nut, and finally tightening the nut. The heuristic correctly
    accounts for the fact that spanners become unusable after a single use. It aims
    to provide an informative (though not necessarily admissible) estimate for
    Greedy Best-First Search.

    # Assumptions
    - There is exactly one agent of type 'man' in the problem instance.
    - Links between locations defined by the `(link ?l1 ?l2)` predicate are symmetric
      (bidirectional) and represent unit cost edges for the 'walk' action.
    - Nuts do not change their location throughout the planning process.
    - The goal specification consists solely of `(tightened ?n)` predicates for a
      predefined set of nuts.

    # Heuristic Initialization
    - The constructor (`__init__`) performs pre-computation based on the task definition:
    - It extracts the set of `goal_nuts` from the `task.goals`.
    - It identifies all unique location names by scanning `link` predicates in `task.static`
      and `at` predicates in `task.initial_state`.
    - It builds an adjacency list (`link_map`) representing the connectivity graph of locations
      based on the static `link` predicates.
    - It computes all-pairs shortest path distances between locations using Breadth-First Search (BFS)
      on the `link_map`. These distances (`self.distances`) represent the minimum number of 'walk'
      actions required between any two locations.
    - It attempts to identify the name of the single 'man' agent (`self.man_name`) by checking
      for `(carrying ...)` predicates in the initial state or looking for known default names
      like 'bob'. This name is crucial for correctly parsing the state during heuristic evaluation.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** The `__call__` method receives a state node. It parses the `node.state` (a set of fact strings) to determine the current situation:
        - Find the man's current location (`man_location`).
        - Identify the spanner the man is carrying (`carried_spanner`), if any, and check if it's currently `usable`.
        - Create a dictionary (`usable_spanners_at`) mapping the names of all `usable` spanners currently on the ground to their respective locations.
        - Create a dictionary (`loose_nuts_at`) mapping the names of `loose` nuts (that are also part of the `goal_nuts` set) to their locations.
    2.  **Check Goal Completion:** If the `loose_nuts_at` dictionary is empty, it signifies that all required nuts are already tightened (or were not loose). The heuristic value is 0, as the goal (regarding nuts) is satisfied.
    3.  **Initialize Simulation:** If there are loose goal nuts remaining, initialize the total estimated heuristic cost `h = 0`. Create mutable copies of the relevant state parts for a step-by-step simulation:
        - `current_lm`: Man's current location, updated after each simulated step.
        - `sim_usable_spanners_at`: Copy of ground usable spanners.
        - `sim_carried_spanner_is_usable`: Boolean flag for the carried spanner's usability.
        - `sim_remaining_nuts_at`: Copy of loose goal nuts.
    4.  **Greedy Simulation Loop:** Enter a loop that continues as long as `sim_remaining_nuts_at` is not empty. Inside the loop, simulate one step of tightening the "cheapest" nut:
        a.  **Find Possible Actions:** Calculate the cost for every possible immediate action sequence that results in tightening one of the `sim_remaining_nuts_at`. Store these possibilities in a list `possible_actions`.
            i.  **Using Carried Spanner:** If `sim_carried_spanner_is_usable` is true, calculate the cost for the man to walk from `current_lm` to each remaining nut's location `ln` and tighten it. Cost = `dist(current_lm, ln) + 1` (tighten action). Add valid options (where `dist != inf`) to `possible_actions`.
            ii. **Using Ground Spanner:** For each usable spanner `s` at location `ls` in `sim_usable_spanners_at`, calculate the cost to walk from `current_lm` to `ls`, pick up `s` (+1 cost), walk from `ls` to each remaining nut's location `ln`, and tighten it (+1 cost). Total Cost = `dist(current_lm, ls) + 1 + dist(ls, ln) + 1`. Add valid options to `possible_actions`.
        b.  **Check Reachability:** If `possible_actions` is empty after checking all combinations, it means no remaining nut can be tightened from the current simulated state (due to lack of usable spanners or reachability issues). The heuristic deems the goal unreachable; return `float('inf')`.
        c.  **Select Best Action:** Find the action in `possible_actions` that has the minimum cost. Let this be `best_action_details`.
        d.  **Update Heuristic Cost:** Add the cost of the best action (`best_action_details['cost']`) to the total heuristic value `h`.
        e.  **Update Simulated State:** Modify the simulation variables to reflect the effect of the chosen action:
            - Update `current_lm` to the location of the nut that was just tightened (`best_action_details['nut_loc']`).
            - Remove the tightened nut (`best_action_details['nut']`) from `sim_remaining_nuts_at`.
            - Mark the spanner used as unusable: If the carried spanner was used, set `sim_carried_spanner_is_usable = False`. If a ground spanner was used, remove it from `sim_usable_spanners_at`.
    5.  **Return Result:** Once the loop terminates (all nuts in `sim_remaining_nuts_at` have been processed), return the total accumulated cost `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-computing distances and identifying goal nuts and the man agent.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Identify goal nuts from the task's goal specification
        self.goal_nuts = set()
        for fact in self.goals:
            parts = get_parts(fact)
            # Check for facts like '(tightened nut_name)'
            if parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        # 2. Identify all unique locations mentioned in the problem
        self.locations = set()
        # Scan initial state and static facts for locations
        facts_to_scan = task.initial_state.union(self.static)
        for fact in facts_to_scan:
             parts = get_parts(fact)
             pred = parts[0]
             args = parts[1:]
             if pred == 'link' and len(args) == 2:
                 self.locations.add(args[0])
                 self.locations.add(args[1])
             elif pred == 'at' and len(args) == 2:
                 # The second argument of 'at' is a location
                 self.locations.add(args[1])

        # 3. Build the location connectivity graph (adjacency list)
        self.link_map = {loc: [] for loc in self.locations}
        for fact in self.static:
            parts = get_parts(fact)
            # Check for facts like '(link loc1 loc2)'
            if parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                # Add edges for both directions, assuming symmetry
                self.link_map.setdefault(l1, []).append(l2)
                self.link_map.setdefault(l2, []).append(l1)

        # 4. Compute all-pairs shortest path distances using BFS
        self.distances = self._compute_all_pairs_shortest_paths()

        # 5. Attempt to identify the man's name from the initial state
        self.man_name = self._find_man_name(task.initial_state)
        # A warning could be logged here if man_name is None, as it might cause issues later.


    def _find_man_name(self, initial_state):
        """
        Attempts to find the name of the 'man' agent from the initial state predicates.
        Returns the name if found, otherwise None.
        """
        # Prioritize finding the agent involved in a 'carrying' predicate
        for fact in initial_state:
             parts = get_parts(fact)
             # Check for '(carrying agent spanner)'
             if parts[0] == 'carrying' and len(parts) == 3:
                 return parts[1] # Assume the first argument is the man

        # Fallback: Check for common default names like 'bob' at a location
        for fact in initial_state:
             parts = get_parts(fact)
             # Check for '(at bob location)'
             if parts[0] == 'at' and len(parts) == 3 and parts[1] == 'bob':
                 return 'bob'

        # If no specific man identifier found, return None.
        # The __call__ method will need to handle this potential ambiguity.
        return None


    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances (number of 'walk' actions) between all
        pairs of locations using Breadth-First Search (BFS).
        Returns a dictionary of dictionaries: distances[loc1][loc2] = distance.
        """
        distances = {loc: {other: float('inf') for other in self.locations} for loc in self.locations}
        if not self.locations: # Handle case with no locations
             return distances

        for start_node in self.locations:
            # Ensure the start node is valid and exists in the distances structure
            if start_node not in distances: continue

            distances[start_node][start_node] = 0
            queue = deque([start_node])
            # visited_dist tracks distances from start_node in the current BFS
            visited_dist = {start_node: 0}

            while queue:
                current_node = queue.popleft()
                current_dist = visited_dist[current_node]

                # Explore neighbors using the pre-built adjacency list (link_map)
                for neighbor in self.link_map.get(current_node, []):
                    # Process neighbor only if it's a known location and not yet visited in this BFS run
                    if neighbor in self.locations and neighbor not in visited_dist:
                        visited_dist[neighbor] = current_dist + 1
                        distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Calculate the heuristic value (estimated cost to goal) for the given state node.
        """
        state = node.state
        man_name = self.man_name

        # If man_name wasn't determined during initialization, try a dynamic check now.
        # This is less efficient but necessary if init failed to find the name.
        if man_name is None:
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'carrying' and len(parts) == 3:
                    man_name = parts[1]; break
                if parts[0] == 'at' and len(parts) == 3 and parts[1] == 'bob': # Fallback check
                    man_name = 'bob'; break
            # If still no man_name found, the heuristic cannot function correctly.
            if man_name is None:
                # Log error or return infinity to indicate failure.
                # print("Error: SpannerHeuristic cannot identify the man agent.")
                return float('inf')

        # --- 1. Parse current state ---
        man_location = None
        carried_spanner = None
        usable_spanners_at = {} # Maps usable ground spanner names to locations
        loose_nuts_at = {}      # Maps loose goal nut names to locations
        spanner_is_usable = set() # Set of all usable spanner names (carried or ground)
        nut_locations = {}      # Locations of all nuts
        spanner_locations = {}  # Locations of all spanners

        # First pass: Collect locations, carrying status, and usability facts
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]
            num_args = len(args)

            if predicate == 'at' and num_args == 2:
                obj, loc = args
                if obj == man_name:
                    man_location = loc
                # Store locations, assuming names help identify type (fragile)
                # A better approach would use type info if available from the task object
                if 'spanner' in obj: spanner_locations[obj] = loc
                if 'nut' in obj: nut_locations[obj] = loc
            elif predicate == 'carrying' and num_args == 2:
                man, spanner = args
                if man == man_name:
                    carried_spanner = spanner
            elif predicate == 'usable' and num_args == 1:
                spanner_is_usable.add(args[0])
            # 'loose' facts processed in the second pass

        # Critical check: Man must be located somewhere.
        if man_location is None:
             return float('inf') # State seems invalid or parsing failed

        # Second pass: Filter and organize the collected information
        carried_spanner_is_usable = (carried_spanner is not None and carried_spanner in spanner_is_usable)

        # Identify usable spanners currently on the ground
        for spanner, loc in spanner_locations.items():
            if spanner != carried_spanner and spanner in spanner_is_usable:
                usable_spanners_at[spanner] = loc

        # Identify which goal nuts are currently loose
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]
            if predicate == 'loose' and len(args) == 1:
                nut = args[0]
                # Check if this loose nut is one of the goal nuts and its location is known
                if nut in self.goal_nuts and nut in nut_locations:
                    loose_nuts_at[nut] = nut_locations[nut]

        # --- 2. Check goal condition ---
        if not loose_nuts_at:
            # All required nuts are already tightened.
            return 0

        # --- 3. Initialize simulation state ---
        h = 0 # Accumulated heuristic cost
        current_lm = man_location # Man's location during simulation
        # Create copies for simulation to avoid modifying original dictionaries
        sim_usable_spanners_at = usable_spanners_at.copy()
        sim_carried_spanner_is_usable = carried_spanner_is_usable
        sim_remaining_nuts_at = loose_nuts_at.copy()

        # --- 4. Greedy simulation loop ---
        while sim_remaining_nuts_at:
            min_step_cost = float('inf')
            best_action_details = None # Stores info about the cheapest action found
            possible_actions = [] # List to hold potential actions for this step

            # --- 4a. Evaluate using carried spanner ---
            if sim_carried_spanner_is_usable:
                for nut, nut_loc in sim_remaining_nuts_at.items():
                    # Get distance, default to infinity if locations are disconnected
                    dist = self.distances.get(current_lm, {}).get(nut_loc, float('inf'))
                    if dist != float('inf'):
                        cost = dist + 1 # Cost = walk distance + 1 (tighten action)
                        possible_actions.append({
                            'cost': cost, 'nut': nut, 'nut_loc': nut_loc,
                            'spanner_type': 'carried', 'spanner_name': carried_spanner
                        })

            # --- 4b. Evaluate using ground spanners ---
            for spanner_name, spanner_loc in sim_usable_spanners_at.items():
                dist_to_spanner = self.distances.get(current_lm, {}).get(spanner_loc, float('inf'))
                if dist_to_spanner == float('inf'):
                    continue # Cannot reach this spanner from current location

                for nut, nut_loc in sim_remaining_nuts_at.items():
                    dist_spanner_to_nut = self.distances.get(spanner_loc, {}).get(nut_loc, float('inf'))
                    if dist_spanner_to_nut != float('inf'): # Can reach nut from spanner location
                        # Cost = walk to spanner + pickup + walk to nut + tighten
                        cost = dist_to_spanner + 1 + dist_spanner_to_nut + 1
                        possible_actions.append({
                            'cost': cost, 'nut': nut, 'nut_loc': nut_loc,
                            'spanner_type': 'ground', 'spanner_name': spanner_name
                        })

            # --- 4c. Check if any action is possible ---
            if not possible_actions:
                # No way to tighten any remaining nut from the current simulated state
                return float('inf') # Goal considered unreachable by this heuristic

            # --- 4d. Select the best (cheapest) action ---
            best_action_details = min(possible_actions, key=lambda x: x['cost'])
            min_step_cost = best_action_details['cost']

            # --- 4e. Update heuristic cost and simulated state ---
            h += min_step_cost
            current_lm = best_action_details['nut_loc'] # Man moves to the nut's location
            del sim_remaining_nuts_at[best_action_details['nut']] # Nut is now considered tightened

            # Mark the used spanner as unusable for future steps in the simulation
            if best_action_details['spanner_type'] == 'carried':
                sim_carried_spanner_is_usable = False
            else: # A ground spanner was used
                del sim_usable_spanners_at[best_action_details['spanner_name']]

        # --- 5. Return final heuristic value ---
        # The loop finished, meaning all goal nuts were processed in the simulation.
        return h
