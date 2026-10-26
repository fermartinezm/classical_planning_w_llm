import itertools
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import heapq # Used for finding closest nuts efficiently

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at bob shed)" -> ["at", "bob", "shed"]
    return fact[1:-1].split()

# Helper function to match facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern. Wildcards (*) are allowed.
    Example: match("(at bob shed)", "at", "bob", "*") -> True
             match("(at spanner1 loc1)", "at", "nut*", "*") -> False
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts
    that are currently loose. It considers the man's location, carried spanners,
    available usable spanners on the ground, and the locations of loose nuts.
    It prioritizes using carried spanners first, then fetches spanners from
    the ground, estimating the travel, pickup, and tightening costs. It aims
    for efficient computation and good guidance for a greedy best-first search,
    and does not need to be admissible.

    # Assumptions
    - There is only one man agent in the problem instance.
    - Links between locations ('link' predicate) represent traversable paths and are assumed to be bidirectional for distance calculation.
    - The problem instance is solvable, meaning there are enough usable spanners for the goal nuts and paths exist between relevant locations. The heuristic returns large finite values for apparent dead ends (like disconnected locations) rather than infinity.
    - The primary cost drivers are the number of nuts to tighten and the travel involved (walking, picking up, tightening). Each tighten action consumes one usable spanner.

    # Heuristic Initialization
    - Extracts the set of all nuts that must be tightened to satisfy the goal conditions (`goal_nuts`).
    - Parses static 'link' predicates from the task's static facts to build a graph representation of the locations.
    - Identifies all unique location names mentioned in links or 'at' predicates in the initial state or static facts.
    - Precomputes all-pairs shortest path distances (number of 'walk' actions) between all known locations using Breadth-First Search (BFS). Stores these distances in `self.distances`. Disconnected locations will have infinite distance.
    - Identifies the single 'man' object by inspecting operator definitions (preferred) or assuming a common name like 'bob' as a fallback.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** In the `__call__` method, iterate through the facts in the current state (`node.state`) to determine:
        - The man's current location (`man_loc`).
        - The set of spanners the man is currently carrying (`carried_spanners`).
        - The set of spanners that are currently usable (`usable_spanners`).
        - The locations of all objects identified as spanners (`spanner_locs`). Spanners are identified if they appear in `usable` or `carrying` predicates.
        - The locations of all nuts relevant to the goal (`nut_locs`).
        - The set of nuts that are currently loose (`loose_nuts`).
    2.  **Identify Pending Goals:** Determine the set of nuts that are both in the goal (`self.goal_nuts`) and currently loose (`loose_nuts`). Store these as `current_goal_loose_nuts` (mapping nut name to its location).
    3.  **Check Goal Completion:** If `current_goal_loose_nuts` is empty, the goal state (regarding nuts) is reached, so return 0.
    4.  **Identify Available Resources:** Determine which carried spanners are usable (`carried_usable_spanners`) and which usable spanners are on the ground (`location_usable_spanners`, mapping spanner name to its location).
    5.  **Initialize Cost:** Set `total_cost = 0`. Create mutable copies/lists of the loose nuts and available spanners for processing.
    6.  **Phase 1: Use Carried Spanners:**
        a. Calculate how many nuts can potentially be tightened using the `carried_usable_spanners` (`num_can_carry_tighten`), limited by the number of loose nuts and the number of carried usable spanners.
        b. Use a min-heap (`heapq`) to efficiently find the `num_can_carry_tighten` loose nuts closest to the man's current location (`man_loc`).
        c. For each selected nut (`nut` at `loc_n`):
           - Calculate the cost: `distance(man_loc, loc_n) + 1` (representing estimated walk actions + 1 tighten action). The distance is looked up from precomputed values.
           - Add this cost to `total_cost`.
        d. Keep track of which nuts were handled in this phase (`handled_nuts_phase1`) to exclude them from Phase 2.
    7.  **Phase 2: Fetch Spanners from Ground:**
        a. If there are still loose nuts remaining after Phase 1:
        b. Create a mutable list of usable spanners available on the ground (`available_loc_spanners`).
        c. Iterate through each remaining loose nut (`nut` at `loc_n`).
        d. For the current nut, find the best available usable spanner (`spanner` at `loc_s`) from the `available_loc_spanners` list. "Best" is defined as minimizing the estimated cost for this specific nut: `distance(man_loc, loc_s) + 1 (pickup) + distance(loc_s, loc_n) + 1 (tighten)`. Distances are retrieved from precomputed values.
        e. If a suitable spanner is found:
           - Add the calculated minimum cost (`best_spanner_cost`) to `total_cost`.
           - Remove the chosen spanner from the `available_loc_spanners` list to prevent it from being assigned to multiple nuts within this heuristic calculation.
        f. If no usable spanner is found on the ground for a required nut (e.g., `available_loc_spanners` is empty), add a large penalty (e.g., 1000) to `total_cost` for this nut to signify high cost or a potential dead end.
    8.  **Return Final Cost:** Return the accumulated `total_cost`. A helper function `get_dist` handles distance lookups, returning a large finite value (1000) for infinite distances (disconnected locations) or invalid location inputs.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Extract goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

        # Extract locations and links
        self.locations = set()
        adj = {} # Adjacency list for BFS

        # Process static facts to find links and locations
        for fact in static_facts:
             if match(fact, "link", "*", "*"):
                 parts = get_parts(fact)
                 loc1, loc2 = parts[1], parts[2]
                 self.locations.add(loc1)
                 self.locations.add(loc2)
                 adj.setdefault(loc1, []).append(loc2)
                 adj.setdefault(loc2, []).append(loc1) # Assume bidirectional links

        # Also find locations mentioned in 'at' predicates in init/static facts
        # This ensures locations without links are also included
        all_facts_for_locs = task.initial_state | static_facts
        for fact in all_facts_for_locs:
            if match(fact, "at", "*", "*"):
                 # The second argument of 'at' is the location
                 loc = get_parts(fact)[2]
                 self.locations.add(loc)

        # Ensure all identified locations have an entry in adj for BFS consistency
        for loc in self.locations:
            adj.setdefault(loc, [])

        # Compute all-pairs shortest paths using BFS
        self.distances = {loc: {other: float('inf') for other in self.locations} for loc in self.locations}
        for start_node in self.locations:
            # Check if start_node is valid and exists in adj
            if start_node not in self.locations: continue

            self.distances[start_node][start_node] = 0
            queue = deque([start_node])
            # Use visited dict to store distances during BFS for efficiency
            visited = {start_node: 0}

            while queue:
                current_node = queue.popleft()
                current_dist = visited[current_node]

                # Use adj.get() for safety in case a location has no links
                for neighbor in adj.get(current_node, []):
                    if neighbor in self.locations and neighbor not in visited:
                        visited[neighbor] = current_dist + 1
                        self.distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)

        # Identify the man agent
        self.man = None
        man_candidates = set()
        # Look for the agent parameter in action definitions provided by the task
        if hasattr(task, 'operators'):
            for op in task.operators:
                # Operator name includes parameters, e.g., "(walk shed location1 bob)"
                op_parts = op.name[1:-1].split()
                action_name = op_parts[0]
                # Identify the man parameter based on its position in known actions
                if action_name == 'walk' and len(op_parts) == 4:
                    man_candidates.add(op_parts[3])
                elif action_name == 'pickup_spanner' and len(op_parts) == 4:
                    man_candidates.add(op_parts[3])
                elif action_name == 'tighten_nut' and len(op_parts) == 5:
                    man_candidates.add(op_parts[3])

        if len(man_candidates) == 1:
            self.man = list(man_candidates)[0]
        elif 'bob' in man_candidates: # Prioritize common name if ambiguous
            self.man = 'bob'
        elif man_candidates: # Pick one if multiple and 'bob' not present
             self.man = list(man_candidates)[0]
             # Optional: print a warning if the choice is ambiguous
             # print(f"Warning: Multiple man candidates found ({man_candidates}). Using '{self.man}'.")
        else:
             # Fallback if no man identified from operators (e.g., if task.operators not available/parsed)
             # Try finding from initial state 'at' facts (less reliable)
             for fact in task.initial_state:
                 # Assuming man is the object in (at man_obj loc)
                 if match(fact, "at", "*", "*"):
                     obj = get_parts(fact)[1]
                     # Very weak check based on common names, improve if possible
                     if 'bob' in obj or 'man' in obj:
                         self.man = obj
                         break
             if not self.man:
                 print("Error: Could not identify the man agent. Heuristic calculation might be incorrect.")
                 # Assign None, will be handled in __call__
                 self.man = None


    def __call__(self, node):
        state = node.state

        # Handle case where man couldn't be identified in init
        if self.man is None:
            print("Error: Man agent not identified during initialization. Returning high heuristic value.")
            return 10000 # Return a large value as heuristic cannot be computed reliably

        # 1. Parse Current State
        man_loc = None
        carried_spanners = set()
        usable_spanners = set()
        spanner_locs = {} # spanner -> location (potential spanners)
        nut_locs = {} # nut -> location
        loose_nuts = set()
        all_spanners_in_state = set() # Track all objects acting as spanners

        for fact in state:
            # Use match for robust parsing against patterns
            if match(fact, "at", self.man, "*"):
                man_loc = get_parts(fact)[2]
            elif match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                if obj in self.goal_nuts:
                    nut_locs[obj] = loc
                else:
                    # Assume other objects might be spanners, track their location
                    spanner_locs[obj] = loc
            elif match(fact, "carrying", self.man, "*"):
                spanner = get_parts(fact)[2]
                carried_spanners.add(spanner)
                all_spanners_in_state.add(spanner) # Mark as spanner
            elif match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                usable_spanners.add(spanner)
                all_spanners_in_state.add(spanner) # Mark as spanner
            elif match(fact, "loose", "*"):
                 nut = get_parts(fact)[1]
                 # Only consider loose nuts that are part of the goal
                 if nut in self.goal_nuts:
                     loose_nuts.add(nut)

        # Refine spanner locations: only keep locations for objects confirmed as spanners
        spanner_locs = {s: loc for s, loc in spanner_locs.items() if s in all_spanners_in_state}

        # 2. Identify Pending Goals
        current_goal_loose_nuts = {} # nut -> location
        for nut in loose_nuts:
            if nut in nut_locs:
                current_goal_loose_nuts[nut] = nut_locs[nut]
            else:
                 # This case (loose goal nut without location) should ideally not happen in consistent states.
                 # If it does, we cannot calculate cost accurately for this nut. Skip or add penalty?
                 # Let's add a penalty if location is missing.
                 print(f"Warning: Location for loose goal nut '{nut}' not found in state. Adding penalty.")
                 # We'll handle this by the distance function returning high value if loc is None.
                 current_goal_loose_nuts[nut] = None # Mark location as unknown

        # 3. Check Goal Completion
        if not current_goal_loose_nuts:
            # Check if all goal nuts are tightened (the primary goal)
            all_tightened = True
            for nut in self.goal_nuts:
                if f"(tightened {nut})" not in state:
                    all_tightened = False
                    break
            if all_tightened:
                 return 0
            else:
                 # Goals remain but no loose nuts? Maybe goal includes other conditions.
                 # Or maybe current_goal_loose_nuts logic missed something.
                 # For this domain, goal is typically just (tightened nut*), so empty loose means done.
                 # If goals included other things, this check might need adjustment.
                 # Let's assume 0 if no goal nuts are loose.
                 if not any(nut in loose_nuts for nut in self.goal_nuts):
                     return 0


        # Check if man's location is known; essential for calculations.
        if man_loc is None:
             print(f"Error: Could not find location for man '{self.man}' in state. Returning high value.")
             return 10000 # High value if state seems invalid or man location missing

        # 4. Identify Available Resources
        carried_usable_spanners = carried_spanners.intersection(usable_spanners)
        # Spanners on the ground that are usable
        location_usable_spanners = {s: loc for s, loc in spanner_locs.items() if s in usable_spanners and s not in carried_spanners}

        # 5. Initialize Cost
        total_cost = 0
        # Use lists for easy removal during assignment/processing
        loose_nuts_items = list(current_goal_loose_nuts.items()) # list of (nut, loc_n)
        carried_usable_list = list(carried_usable_spanners)
        loc_usable_list = list(location_usable_spanners.items()) # list of (spanner, loc_s)

        # Helper function for safe distance lookup using precomputed distances
        # Returns a large finite value for infinity or invalid locations.
        PENALTY_COST = 1000
        def get_dist(loc1, loc2):
            if loc1 is None or loc2 is None:
                # print(f"Warning: Calculating distance with None location ({loc1}, {loc2}).")
                return PENALTY_COST
            if loc1 not in self.locations or loc2 not in self.locations:
                # print(f"Warning: Location unknown or invalid in distance calculation: {loc1} or {loc2}")
                return PENALTY_COST
            # Lookup distance, default to infinity if not found (should be precomputed)
            dist = self.distances.get(loc1, {}).get(loc2, float('inf'))
            # Return large finite number for infinity (e.g., disconnected graph)
            return dist if dist != float('inf') else PENALTY_COST

        # 6. Phase 1: Use Carried Spanners
        num_can_carry_tighten = min(len(loose_nuts_items), len(carried_usable_list))

        handled_nuts_phase1 = set()
        if num_can_carry_tighten > 0:
            # Find the closest nuts to the man among the loose ones
            nuts_by_distance = []
            for nut, loc_n in loose_nuts_items:
                 dist = get_dist(man_loc, loc_n)
                 # Use tuple (distance, nut_name) for stable sorting if distances are equal
                 heapq.heappush(nuts_by_distance, (dist, nut, loc_n))

            # Process the closest nuts up to the limit
            for i in range(num_can_carry_tighten):
                if not nuts_by_distance: break # Safety check
                dist, nut, loc_n = heapq.heappop(nuts_by_distance)
                # Cost = walk_actions + tighten_action
                # Check if distance is penalty cost, indicating issue
                if dist >= PENALTY_COST:
                    cost = PENALTY_COST # Propagate high cost if path is bad/unknown
                else:
                    cost = dist + 1
                total_cost += cost
                handled_nuts_phase1.add(nut)

            # Update the list of loose nuts for Phase 2
            loose_nuts_items = [(n, l) for n, l in loose_nuts_items if n not in handled_nuts_phase1]

        # 7. Phase 2: Fetch Spanners from Ground
        if loose_nuts_items:
            # Create a mutable list of available spanners on the ground
            available_loc_spanners = loc_usable_list[:] # Copy the list

            for nut, loc_n in loose_nuts_items:
                best_spanner_cost = float('inf')
                chosen_spanner_info = None # Stores (spanner_name, spanner_loc)
                best_spanner_index = -1

                # Find the best available ground spanner for this nut
                for i, (spanner, loc_s) in enumerate(available_loc_spanners):
                    # Cost: walk to spanner + pickup + walk to nut + tighten
                    dist_man_spanner = get_dist(man_loc, loc_s)
                    dist_spanner_nut = get_dist(loc_s, loc_n)

                    # If any path is problematic, the cost will be high due to PENALTY_COST
                    cost = dist_man_spanner + 1 + dist_spanner_nut + 1

                    if cost < best_spanner_cost:
                        best_spanner_cost = cost
                        chosen_spanner_info = (spanner, loc_s)
                        best_spanner_index = i

                if chosen_spanner_info is not None:
                    # Add the cost, even if it's the penalty cost
                    total_cost += best_spanner_cost
                    # Remove the chosen spanner from the available pool for subsequent nuts
                    # This prevents double-counting the same spanner in the heuristic estimate
                    del available_loc_spanners[best_spanner_index]
                else:
                    # No usable spanner found on the ground for this nut.
                    # Add penalty cost.
                    # print(f"Warning: No available ground spanner found for nut {nut}. Adding penalty.")
                    total_cost += PENALTY_COST

        # 8. Return Final Cost
        # Ensure the heuristic is non-negative
        return max(0, total_cost)

