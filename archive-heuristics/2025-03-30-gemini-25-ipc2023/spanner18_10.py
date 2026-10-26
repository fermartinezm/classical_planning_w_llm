import heapq
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import itertools
import math

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'spanner'.

    # Summary
    This heuristic estimates the remaining cost to tighten all goal nuts that are currently loose.
    It calculates the minimum estimated cost to tighten each loose goal nut individually and sums these costs.
    The cost for a single nut considers the actions needed: walking to a usable spanner (if not carried), picking it up, walking to the nut's location, and tightening the nut. It assumes the 'closest' sequence of actions for each nut independently, ignoring potential resource conflicts (like a single spanner being needed for multiple nuts or spanners becoming unusable).

    # Assumptions
    - There is exactly one 'man' agent in the problem.
    - Links between locations are bidirectional.
    - The goal is solely defined by '(tightened nut)' predicates.
    - Spanners become unusable after one use.
    - The heuristic does not need to be admissible.

    # Heuristic Initialization
    - Extracts all locations and the connections (links) between them from the static facts.
    - Computes all-pairs shortest path distances between locations using Breadth-First Search (BFS).
    - Identifies the 'man' object (relies on finding an object involved in 'at' and 'carrying' predicates, potentially fragile if object names are unusual or type information isn't available).
    - Identifies the set of nuts that need to be tightened according to the goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify State:** Determine the current location of the man, whether the man is carrying a usable spanner, the locations of all loose nuts that are part of the goal, and the locations of all usable spanners currently on the ground.
    2.  **Goal Check:** If there are no loose goal nuts, the heuristic value is 0.
    3.  **Solvability Check:** Count the total number of usable spanners (carried + on the ground). If this number is less than the number of loose goal nuts, the goal is unreachable, return infinity.
    4.  **Cost per Nut:** For each loose goal nut `n` at location `ln`:
        a. Initialize `min_cost_for_nut` to infinity.
        b. **Option 1 (Use Carried Spanner):** If the man is currently carrying a usable spanner, calculate the cost: `distance(man_loc, nut_loc) + 1` (for walk + tighten). Update `min_cost_for_nut`.
        c. **Option 2 (Use Ground Spanner):** Iterate through all usable spanners `s` located on the ground at `ls`. Calculate the cost: `distance(man_loc, spanner_loc) + 1 + distance(spanner_loc, nut_loc) + 1` (for walk_to_spanner + pickup + walk_to_nut + tighten). Update `min_cost_for_nut` with the minimum cost found across all ground spanners.
        d. If `min_cost_for_nut` remains infinity after checking both options (e.g., nut or required spanners are unreachable), the state is considered a dead end, return infinity.
    5.  **Total Heuristic Value:** Sum the `min_cost_for_nut` calculated for all loose goal nuts. This sum represents the estimated total actions required.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing distances and identifying key objects.

        Args:
            task: The planning task object containing initial state, goals, operators, and static facts.
        """
        super().__init__(task)
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # Precompute distances between locations
        self._build_graph_and_distances()

        # Identify the man object (NOTE: This implementation makes assumptions)
        self.man = self._find_man()
        if not self.man:
             print("Warning: Could not reliably identify the 'man' object. Heuristic might be inaccurate.")
             # As a fallback, try common names or patterns if needed, but it's risky.
             # For now, we proceed assuming it might be found later or error out.


        # Identify goal nuts
        self.goal_nuts = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == 'tightened':
                self.goal_nuts.add(parts[1])

        # Identify all spanner objects (NOTE: Relies on assumptions)
        self.all_spanners = self._find_all_spanners()


    def _find_man(self):
        """
        Attempts to identify the single 'man' object in the problem.

        NOTE: This is potentially fragile as it relies on finding an object
        that is involved in 'at' and 'carrying' predicates in the initial state
        or can perform actions like 'walk'. It does not use explicit type information
        which might not be available in the provided Task structure.
        It might incorrectly identify an object if the problem structure is unusual.
        """
        man_candidates = set()
        # Check objects involved in 'carrying' in the initial state
        for fact in self.task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'carrying':
                man_candidates.add(parts[1])
                # If found, likely the man, return early
                return parts[1]

        # If not found via carrying, check objects 'at' locations that might be men
        for fact in self.task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                 potential_man = parts[1]
                 # Check if this object appears as the agent in typical man actions
                 # (e.g., first param of carrying, third param of walk)
                 # This requires checking operator structure, which is complex here.
                 # Simple check: If only one object is 'at' something initially, assume it's the man.
                 # Better check: See if it's involved in 'carrying' later.
                 # Let's assume the first object found 'at' a location is a candidate.
                 man_candidates.add(potential_man)

        # If multiple candidates, this approach is ambiguous.
        # If exactly one candidate, return it.
        if len(man_candidates) == 1:
            return list(man_candidates)[0]

        # If still ambiguous, return None or raise an error.
        # Trying a common name like 'bob' based on examples
        if 'bob' in man_candidates:
             return 'bob'

        # Last resort: return the first candidate found, if any
        if man_candidates:
            return list(man_candidates)[0]

        return None # Could not identify

    def _find_all_spanners(self):
        """
        Attempts to identify all spanner objects.

        NOTE: Relies on finding objects involved in 'usable' or 'carrying' predicates
        in the initial state or mentioned in goal state involving spanners (if any).
        Does not use explicit type information.
        """
        spanners = set()
        for fact in self.task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'usable':
                spanners.add(parts[1])
            elif parts[0] == 'carrying':
                spanners.add(parts[2]) # The object being carried is a spanner
            elif parts[0] == 'at':
                 # Check if object 'at' location might be a spanner based on name or later usage
                 obj = parts[1]
                 # A simple heuristic: if it starts with 'spanner', assume it is one.
                 if obj.startswith('spanner'):
                     spanners.add(obj)

        # Also consider spanners mentioned in operators if needed, but let's stick to state facts for now.
        return spanners


    def _build_graph_and_distances(self):
        """Builds the location graph and computes all-pairs shortest paths using BFS."""
        self.locations = set()
        self.links = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                # Assuming links are bidirectional as per standard PDDL practice
                self.links.setdefault(loc1, []).append(loc2)
                self.links.setdefault(loc2, []).append(loc1)

        self.dist = {}
        if not self.locations: # Handle empty problems or problems without locations
            return

        for start_node in self.locations:
            # Initialize distances from start_node to all other nodes as infinity
            self.dist[start_node] = {loc: float('inf') for loc in self.locations}
            # Distance to itself is 0
            self.dist[start_node][start_node] = 0

            queue = deque([start_node])
            visited = {start_node} # Keep track of visited nodes for BFS

            while queue:
                current_node = queue.popleft()
                current_dist = self.dist[start_node][current_node]

                # Explore neighbors
                for neighbor in self.links.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.

        Args:
            node: The state node in the search graph.

        Returns:
            An estimated cost (number of actions) to reach the goal, or infinity if unreachable.
        """
        state = node.state

        # --- 1. Identify State ---
        man_loc = None
        carried_spanner = None
        nut_locations = {} # nut -> location
        spanner_locations = {} # spanner -> location
        loose_nuts = set()
        usable_spanners = set()

        if not self.man:
             # If man wasn't identified in init, try again or fail
             # This indicates a problem setup issue or limitation of _find_man
             # For safety, return infinity as state cannot be properly evaluated
             print("Error: Man object unknown, cannot compute heuristic.")
             return float('inf')


        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at':
                obj, loc = args[0], args[1]
                if obj == self.man:
                    man_loc = loc
                elif obj in self.goal_nuts:
                    nut_locations[obj] = loc
                elif obj in self.all_spanners:
                    spanner_locations[obj] = loc
            elif predicate == 'carrying' and args[0] == self.man:
                carried_spanner = args[1]
            elif predicate == 'loose' and args[0] in self.goal_nuts:
                loose_nuts.add(args[0])
            elif predicate == 'usable':
                usable_spanners.add(args[0])

        if man_loc is None:
             # Should not happen if man exists and state is valid
             print(f"Error: Could not find location for man '{self.man}' in state: {state}")
             return float('inf')

        # Filter for loose goal nuts
        loose_goal_nuts_map = {}
        for nut in loose_nuts:
            if nut in nut_locations:
                loose_goal_nuts_map[nut] = nut_locations[nut]
            else:
                # This implies a loose goal nut doesn't have a location in the current state.
                # This might happen if the nut location is static (not typical for 'locatable')
                # or the state is inconsistent. Assume dead end.
                print(f"Error: Location unknown for loose goal nut '{nut}' in state: {state}")
                return float('inf')

        # --- 2. Goal Check ---
        if not loose_goal_nuts_map:
            return 0

        # Identify carried usable spanner and ground usable spanners
        carrying_usable = False
        if carried_spanner and carried_spanner in usable_spanners:
            carrying_usable = True

        ground_usable_spanners = {}
        for spanner, loc in spanner_locations.items():
            if spanner in usable_spanners:
                ground_usable_spanners[spanner] = loc

        # --- 3. Solvability Check ---
        num_total_usable = len(ground_usable_spanners) + (1 if carrying_usable else 0)
        if len(loose_goal_nuts_map) > num_total_usable:
            return float('inf') # Not enough usable spanners

        # --- 4. Cost per Nut & 5. Total Heuristic Value ---
        total_heuristic_cost = 0
        for nut, nut_loc in loose_goal_nuts_map.items():
            min_cost_for_nut = float('inf')

            # Ensure nut_loc is valid before proceeding
            if nut_loc not in self.locations:
                 print(f"Error: Nut location '{nut_loc}' not found in precomputed locations.")
                 return float('inf') # Invalid state or graph issue

            # Option 1: Use carried usable spanner
            if carrying_usable:
                # Get distance, handle potential disconnectivity
                walk_cost = self.dist.get(man_loc, {}).get(nut_loc, float('inf'))
                if walk_cost != float('inf'):
                    min_cost_for_nut = min(min_cost_for_nut, walk_cost + 1) # walk + tighten

            # Option 2: Use a ground usable spanner
            for spanner, spanner_loc in ground_usable_spanners.items():
                 # Ensure spanner_loc is valid
                 if spanner_loc not in self.locations:
                     print(f"Error: Spanner location '{spanner_loc}' not found in precomputed locations.")
                     continue # Skip this spanner, maybe try others

                 # Get distances, handle potential disconnectivity
                 walk1_cost = self.dist.get(man_loc, {}).get(spanner_loc, float('inf'))
                 walk2_cost = self.dist.get(spanner_loc, {}).get(nut_loc, float('inf'))

                 if walk1_cost != float('inf') and walk2_cost != float('inf'):
                     # walk_to_spanner + pickup + walk_to_nut + tighten
                     cost = walk1_cost + 1 + walk2_cost + 1
                     min_cost_for_nut = min(min_cost_for_nut, cost)

            if min_cost_for_nut == float('inf'):
                # This specific nut cannot be tightened from the current state (e.g., unreachable)
                return float('inf') # State is likely a dead end

            total_heuristic_cost += min_cost_for_nut

        # Ensure heuristic is non-negative
        return max(0, total_heuristic_cost)

