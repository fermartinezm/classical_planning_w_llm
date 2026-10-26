import itertools
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is available

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    return fact[1:-1].split()

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the cost to tighten all goal nuts that are currently loose.
    The estimate is based on the number of tighten actions, the number of
    spanner pickup actions required, and the walking distance to the
    first required location (either a nut location if holding a usable spanner,
    or a spanner location if needing to pick one up).

    # Assumptions
    - There is exactly one 'man' agent in the problem.
    - The location graph defined by `link` predicates is connected for relevant locations.
    - The goal is solely defined by `(tightened nut)` predicates.
    - The man can only carry one spanner at a time (implied by actions, though not strictly enforced by state representation).

    # Heuristic Initialization
    - Parses static `link` facts to build an adjacency list representation of the location graph.
    - Computes all-pairs shortest paths using Breadth-First Search (BFS) and stores the distances.
    - Identifies the set of nuts that need to be tightened to satisfy the goal conditions.
    - Attempts to identify the single 'man' agent present in the problem instance.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Man and Goal Nuts:** The constructor identifies the man agent and the set of nuts required by the goal state.
    2.  **Parse Current State:** In the `__call__` method, parse the current state to find:
        a. The man's current location (`lm`).
        b. Which spanner (if any) the man is carrying (`carried_spanners`).
        c. The set of all spanners currently marked as `usable` (`usable_spanners_all`).
        d. The set of goal nuts that are currently `loose` (`current_loose_goal_nuts`).
        e. The locations of all loose goal nuts (`nut_locs`).
        f. The locations of all usable spanners (`spanner_locs`).
    3.  **Count Remaining Goal Nuts (`N`):** Determine the number of goal nuts that are still loose (`N = len(current_loose_goal_nuts)`).
    4.  **Check Goal Completion:** If `N == 0`, all goal nuts are tightened. Return heuristic value 0.
    5.  **Count Tighten Actions:** `N` tighten actions will be required. Initialize `cost = N`.
    6.  **Check Spanner Availability:** Find the total number of usable spanners (`U_total = len(usable_spanners_all)`). If `U_total < N`, the goal is impossible to reach from this state because not enough usable spanners exist. Return `float('inf')`.
    7.  **Check Carried Usable Spanner (`CU`):** Determine if the man is currently carrying a spanner that is also usable. Set `CU = 1` if yes, `CU = 0` otherwise. (Assumes man carries at most one).
    8.  **Count Pickup Actions (`P`):** Calculate the minimum number of spanner pickup actions needed. Since each tighten uses up a spanner's usability, and the man starts with at most one usable spanner (`CU`), `P = max(0, N - CU)` pickups are required. Add `P` to the `cost`.
    9.  **Estimate Initial Walking Cost (`N_walk_first`):** Estimate the walking distance required to get to the location of the *first* necessary action (either tightening a nut or picking up a spanner).
        a.  If `CU > 0` (man has a usable spanner): The first action is tightening. Find the minimum distance from the man's current location (`lm`) to the location (`ln`) of any loose goal nut. `N_walk_first = min(dist(lm, ln))`.
        b.  If `CU == 0` (man needs a spanner): The first action is picking up a spanner (this is required because `N > 0` implies `P > 0`). Find the minimum distance from the man's current location (`lm`) to the location (`ls`) of any *available* (not carried) usable spanner. `N_walk_first = min(dist(lm, ls))`.
        c.  Handle edge cases: If the required locations (nuts or available spanners) don't exist when needed, return `float('inf')`.
    10. **Add Walking Cost:** Add `N_walk_first` to the total `cost`.
    11. **Return Total Cost:** The final heuristic value is `h = N + P + N_walk_first`. Return this value as an integer.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information:
        - Builds the location graph from 'link' predicates.
        - Computes all-pairs shortest paths between locations.
        - Identifies goal nuts.
        - Identifies the man agent.
        """
        self.goals = task.goals
        static_facts = task.static

        # Identify goal nuts
        self.goal_nuts = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == 'tightened':
                self.goal_nuts.add(parts[1])

        # Build location graph
        self.adj = {}
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.adj.setdefault(loc1, []).append(loc2)
                self.adj.setdefault(loc2, []).append(loc1)

        # Identify the man agent - requires assumptions without type info
        self.man = None
        # Try finding man from 'at' or 'carrying' in initial state
        potential_men = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 # Check if it looks like a man based on common names or if it's the agent parameter in actions
                 # Simple check: assume it's the agent if it's not a spanner or nut
                 # This is weak. A better approach needs object types.
                 # Let's assume the man is the first argument of 'at' if it's not a spanner/nut object.
                 # We need the list of spanners/nuts. Let's find them first.
                 pass # Defer man identification after finding other objects
             elif parts[0] == 'carrying':
                 potential_men.add(parts[1]) # Man is the first arg of carrying

        # Find all spanners and nuts mentioned in the initial state or goals
        all_spanners = set()
        all_nuts = set(self.goal_nuts)
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] in ['at', 'carrying', 'usable', 'loose', 'tightened']:
                 # Check arguments - if they contain 'spanner' or 'nut' in name, assume type
                 for arg in parts[1:]:
                     # This is heuristic based on naming conventions in examples
                     if 'spanner' in arg:
                         all_spanners.add(arg)
                     elif 'nut' in arg:
                         all_nuts.add(arg)

        # Now try to identify the man from 'at' predicates again
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 agent = parts[1]
                 if agent not in all_spanners and agent not in all_nuts:
                     potential_men.add(agent)

        # Assume there's only one man
        if len(potential_men) == 1:
            self.man = potential_men.pop()
        elif len(potential_men) > 1:
             print(f"Warning: Found multiple potential man agents: {potential_men}. Using the first one found.")
             self.man = list(potential_men)[0]

        if not self.man:
            # If still not found, maybe the man is only mentioned in static facts? Unlikely.
            # Or only in operators? Need operator parsing.
            # Last resort: Check objects list if available (not provided by Task class here)
            # If the examples always use 'bob', we could default to that, but it's brittle.
            # Check example 1: bob - man. Example 2: bob - man. Let's default to 'bob' if needed.
            if 'bob' in potential_men: # Prioritize 'bob' if found among potentials
                self.man = 'bob'
            elif not self.man:
                 # If no potential men found at all, maybe the problem is trivial or malformed?
                 # Or the man starts carrying something but isn't 'at' anywhere?
                 # Let's assume 'bob' if absolutely no other clue.
                 print("Warning: Could not reliably identify the man agent. Assuming 'bob'.")
                 self.man = 'bob'


        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        # Ensure all locations mentioned in init state are included, even if not in links
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                self.locations.add(parts[2])

        for start_node in self.locations:
            # Initialize distances from start_node
            self.distances[start_node] = {loc: float('inf') for loc in self.locations}
            if start_node not in self.locations: continue # Skip if start node isn't actually a location
            self.distances[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node}

            while queue:
                current_node = queue.popleft()
                current_dist = self.distances[start_node][current_node]

                # Use self.adj which contains linked neighbours
                for neighbor in self.adj.get(current_node, []):
                    if neighbor in self.locations and neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)

    def get_dist(self, loc1, loc2):
        """ Safely get precomputed shortest path distance between two locations. """
        if loc1 == loc2:
            return 0
        # Check if loc1 is a known starting point for BFS
        if loc1 not in self.distances:
            # print(f"Warning: Location {loc1} not found in distance matrix keys.")
            return float('inf')
        # Get distance, default to infinity if loc2 is unreachable from loc1
        dist = self.distances[loc1].get(loc2, float('inf'))
        # print(f"Dist({loc1}, {loc2}) = {dist}")
        return dist


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state

        # --- State Parsing ---
        lm = None
        carried_spanners = set()
        usable_spanners_all = set()
        loose_nuts = set()
        nut_locs = {}
        spanner_locs = {} # Location of spanners not carried

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at':
                obj, loc = args[0], args[1]
                if obj == self.man:
                    lm = loc
                # Check if obj is a nut (heuristic check)
                elif 'nut' in obj or obj in self.goal_nuts:
                    nut_locs[obj] = loc
                # Check if obj is a spanner (heuristic check)
                elif 'spanner' in obj:
                    spanner_locs[obj] = loc
            elif predicate == 'carrying':
                man, spanner = args[0], args[1]
                if man == self.man:
                    carried_spanners.add(spanner)
            elif predicate == 'usable':
                spanner = args[0]
                usable_spanners_all.add(spanner)
            elif predicate == 'loose':
                nut = args[0]
                # Only consider loose nuts that are part of the goal
                if nut in self.goal_nuts:
                    loose_nuts.add(nut)

        # If man's location is unknown, cannot compute heuristic
        if lm is None:
             # This might happen in unusual states or if man identification failed.
             # print("Warning: Man location (lm) not found in state.")
             return float('inf')

        # --- Heuristic Calculation ---

        # 1. Identify remaining loose goal nuts
        current_loose_goal_nuts = self.goal_nuts.intersection(loose_nuts)
        N = len(current_loose_goal_nuts)

        # 2. Check Goal Completion
        if N == 0:
            return 0

        # 3. Count Tighten Actions
        cost = N

        # 4. Count Usable Spanners & Check Feasibility
        U_total = len(usable_spanners_all)
        if U_total < N:
            # Not enough usable spanners exist in the current state to finish the remaining nuts
            return float('inf')

        # 5. Check Carried Usable Spanner (Assume at most 1 carried)
        carried_usable_spanners = carried_spanners.intersection(usable_spanners_all)
        CU = 1 if carried_usable_spanners else 0

        # 6. Count Pickup Actions
        P = max(0, N - CU)
        cost += P

        # 7. Estimate Initial Walking Cost
        N_walk_first = float('inf') # Initialize walk cost to infinity

        # Find locations of loose goal nuts that are actually 'at' somewhere
        L_nuts_locs = {nut_locs[n] for n in current_loose_goal_nuts if n in nut_locs}
        if len(L_nuts_locs) != N:
             # This implies a loose goal nut exists but its location is unknown in the state.
             # This shouldn't happen in valid states.
             # print(f"Warning: Location missing for some loose goal nuts. Found locs for {len(L_nuts_locs)} out of {N}.")
             return float('inf')

        # Find locations of available (not carried) usable spanners
        available_usable_spanners = usable_spanners_all - carried_spanners
        L_spanners_avail_locs = {spanner_locs[s] for s in available_usable_spanners if s in spanner_locs}

        if CU > 0: # Man has a usable spanner, first action is tighten
            if not L_nuts_locs:
                 # This case should not be reachable if N > 0
                 # print("Warning: N > 0 but no locations found for loose goal nuts.")
                 return float('inf')
            # Calculate min distance to any loose goal nut
            min_dist_to_nut = min((self.get_dist(lm, ln) for ln in L_nuts_locs), default=float('inf'))
            N_walk_first = min_dist_to_nut

        else: # Man does not have a usable spanner (CU == 0)
            if P > 0: # Needs to pick one up (requires N > 0)
                if not L_spanners_avail_locs:
                    # Needs pickup, but no usable spanners available at any location. Goal unreachable from here.
                    # This check is slightly different from U_total < N. U_total could be >= N,
                    # but all usable spanners might be carried by someone else (if multi-agent)
                    # or inaccessible. In our single-man case, this means U_total >= N, CU=0,
                    # but all usable spanners are somehow not 'at' a location? Seems unlikely.
                    # print("Warning: Need to pickup spanner (P>0), but none are available at locations.")
                    return float('inf')
                # Calculate min distance to any available usable spanner
                min_dist_to_spanner = min((self.get_dist(lm, ls) for ls in L_spanners_avail_locs), default=float('inf'))
                N_walk_first = min_dist_to_spanner
            # else P == 0: This implies N=0 (already handled) or N=CU=1 (handled by CU>0 case).
            # So if CU=0, P must be > 0 if N > 0.

        # If N_walk_first remained infinity, something is wrong (e.g., man isolated)
        if N_walk_first == float('inf'):
            # print(f"Warning: Could not calculate N_walk_first. Man at {lm}, Nuts at {L_nuts_locs}, Spanners at {L_spanners_avail_locs}")
            return float('inf') # Indicate unreachability or error

        cost += N_walk_first

        # Ensure cost is non-negative and finite before returning int
        if cost == float('inf'):
            return float('inf')
        if cost < 0:
             # Should not happen with BFS distances and positive counts
             # print(f"Warning: Negative heuristic value calculated: {cost}")
             cost = 0

        return int(round(cost))

