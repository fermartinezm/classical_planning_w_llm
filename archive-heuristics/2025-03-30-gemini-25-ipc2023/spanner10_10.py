import collections
import itertools
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    # Example: "(at bob shed)" -> ["at", "bob", "shed"]
    return fact[1:-1].split()

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts
    specified in the goal state. It calculates the cost by simulating a greedy
    strategy where the single man sequentially addresses each required nut tightening.
    For each nut, it estimates the cost of walking to a usable spanner (if not already
    carrying one), picking it up, walking to the nut's location, and tightening it.
    The heuristic accounts for the fact that each tightening action consumes the
    usability of one spanner.

    # Assumptions
    - There is exactly one man agent in the problem.
    - Nuts do not change their location.
    - The `link` predicates define a static, undirected graph of locations.
    - The goal consists solely of `(tightened ?n)` predicates.
    - The heuristic assumes that if the number of initially usable spanners is less
      than the number of nuts to be tightened, the goal is unreachable from the start.
      It returns infinity for states where the number of remaining usable spanners
      is less than the number of remaining nuts to tighten.

    # Heuristic Initialization
    - Extracts all object names by type (location, man, nut, spanner).
    - Parses static `link` predicates to build an adjacency list representation
      of the location graph.
    - Precomputes all-pairs shortest path distances between locations using BFS.
      Stores distances in a dictionary `self.distances`. Unreachable pairs have
      infinite distance.
    - Stores the fixed location of each nut in `self.nut_locations`.
    - Identifies the set of nuts that need to be tightened according to the goal
      predicates (`self.goal_nuts`).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Remaining Goals:** Determine the set of nuts (`nuts_to_tighten`)
        that are specified in the goal but are not yet `(tightened)` in the
        current `state`. If this set is empty, the goal is reached, return 0.
    2.  **Get Current State Info:**
        - Find the man's current location (`man_location`).
        - Identify all currently usable spanners (`usable_spanners_state`).
        - Determine which usable spanners are currently carried by the man
          (`usable_carried`).
        - Determine which usable spanners are on the ground and their locations
          (`usable_on_ground`). Map: `location -> set of spanners`.
    3.  **Check Solvability (Spanner Count):** Calculate the total number of
        usable spanners available (carried + on ground). If this number is less
        than `len(nuts_to_tighten)`, return `float('inf')` as the state is
        considered unsolvable.
    4.  **Simulate Tightening Process:**
        - Initialize heuristic value `h = 0`.
        - Keep track of the man's simulated location (`current_man_loc`),
          initialized to the actual `man_location`.
        - Maintain mutable copies of the sets of usable spanners (carried and on
          ground) to simulate their consumption.
        - Iterate through `nuts_to_tighten` (e.g., in a fixed sorted order for
          determinism):
            a.  Get the location of the current nut `n` (`nut_loc`).
            b.  **Option 1: Use a Carried Spanner:** If the set of usable carried
                spanners (copy) is not empty:
                i.   Select one usable carried spanner `s`.
                ii.  Calculate the walking distance: `dist = self.distances[(current_man_loc, nut_loc)]`.
                iii. Add `dist + 1` (walk + tighten action) to `h`.
                iv.  Update `current_man_loc = nut_loc`.
                v.   Remove `s` from the usable carried set (copy).
            c.  **Option 2: Pick Up a Spanner:** If no usable carried spanner is
                available:
                i.   Find the usable spanner `s` on the ground at location `ls`
                     that minimizes the combined travel cost:
                     `dist(current_man_loc, ls) + dist(ls, nut_loc)`.
                ii.  If no usable spanners are left on the ground, return `float('inf')`
                     (should be caught by the initial check, but for safety).
                iii. Let the minimum combined travel distance be `min_travel_dist = dist(current_man_loc, ls) + dist(ls, nut_loc)`.
                iv.  Add `min_travel_dist + 1 (pickup) + 1 (tighten)` to `h`.
                v.   Update `current_man_loc = nut_loc`.
                vi.  Remove the chosen spanner `s` from the usable on ground set (copy).
            d.  If any required distance calculation results in `float('inf')`, it means
                the required location is unreachable; return `float('inf')`.
    5.  **Return Heuristic Value:** Return the total accumulated cost `h`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # Extract object types (optional, but good for clarity)
        self.locations = set()
        self.nuts = set()
        self.spanners = set()
        self.men = set()
        # Note: Parsing types from the domain file is complex here.
        # We infer types from predicates in init/static facts.
        # Assuming task.facts contains all potential objects might be needed,
        # or inferring from predicates like 'at', 'link', 'loose', 'usable'.

        # Store nut locations (assuming they are static and defined in init)
        self.nut_locations = {}
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and parts[1].startswith('nut'): # Basic type inference
                 self.nuts.add(parts[1])
                 self.nut_locations[parts[1]] = parts[2]
                 self.locations.add(parts[2])
             elif parts[0] == 'at' and parts[1].startswith('spanner'):
                 self.spanners.add(parts[1])
                 self.locations.add(parts[2])
             elif parts[0] == 'at' and parts[1].startswith('bob'): # Assuming 'bob' or similar is man
                 self.men.add(parts[1])
                 self.locations.add(parts[2])
             elif parts[0] == 'link':
                 self.locations.add(parts[1])
                 self.locations.add(parts[2])

        # Identify goal nuts
        self.goal_nuts = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'tightened':
                self.goal_nuts.add(parts[1])
                # Ensure goal nuts are known nuts
                if parts[1] not in self.nut_locations:
                     # This might happen if a nut starts tightened but is still in goal?
                     # Or if nut location isn't in initial 'at' facts.
                     # We need its location. Let's assume it must be in nut_locations.
                     pass


        # Build adjacency list and compute distances
        self.adj = collections.defaultdict(list)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.adj[l1].append(l2)
                self.adj[l2].append(l1) # Assuming bidirectional links
                self.locations.add(l1)
                self.locations.add(l2)

        self.distances = self._compute_all_pairs_shortest_paths()

        # Assume only one man
        if not self.men:
             # Try finding man in initial state if not found before
             for fact in task.initial_state:
                 parts = get_parts(fact)
                 if parts[0] == 'at' and len(parts) == 3: # Basic check for (at obj loc)
                     # Heuristic guess: if obj is not nut/spanner/location, it might be man
                     if parts[1] not in self.nuts and parts[1] not in self.spanners and parts[1] not in self.locations:
                         self.men.add(parts[1])
                         break # Assume first one found is the man

        if len(self.men) != 1:
            print(f"Warning: Expected 1 man, found {len(self.men)}. Heuristic may be inaccurate.")
            # If no man found, heuristic cannot work.
            if not self.men:
                 raise ValueError("SpannerHeuristic: No man object found in the problem instance.")
        self.man_name = list(self.men)[0]


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        distances = collections.defaultdict(lambda: float('inf'))
        for start_node in self.locations:
            distances[(start_node, start_node)] = 0
            queue = collections.deque([(start_node, 0)])
            visited = {start_node: 0}

            while queue:
                current_loc, dist = queue.popleft()
                for neighbor in self.adj[current_loc]:
                    if neighbor not in visited or visited[neighbor] > dist + 1:
                        visited[neighbor] = dist + 1
                        distances[(start_node, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))
        return distances

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given node's state."""
        state = node.state

        # 1. Identify loose nuts that are part of the goal
        tightened_nuts_in_state = {get_parts(f)[1] for f in state if f.startswith('(tightened ')}
        nuts_to_tighten = self.goal_nuts - tightened_nuts_in_state

        if not nuts_to_tighten:
            return 0 # Goal reached

        # 2. Get current state info
        man_location = None
        carried_spanners_names = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == self.man_name:
                man_location = parts[2]
            elif parts[0] == 'carrying' and parts[1] == self.man_name:
                carried_spanners_names.add(parts[2])
        
        if man_location is None:
             # This indicates an issue, maybe man is not 'at' anywhere?
             # Or the man_name identification failed.
             print(f"Warning: Could not find location for man '{self.man_name}' in state.")
             return float('inf') # Cannot proceed without man's location


        usable_spanners_state = {get_parts(f)[1] for f in state if f.startswith('(usable ')}
        usable_carried = usable_spanners_state.intersection(carried_spanners_names)

        usable_on_ground = collections.defaultdict(set)
        # Need spanner locations from the current state
        for fact in state:
            parts = get_parts(fact)
            # Check if it's an 'at' predicate for a known spanner
            if parts[0] == 'at' and parts[1] in self.spanners:
                spanner_name = parts[1]
                spanner_loc = parts[2]
                # Check if this spanner is usable and not carried
                if spanner_name in usable_spanners_state and spanner_name not in usable_carried:
                    usable_on_ground[spanner_loc].add(spanner_name)

        # 3. Check solvability (spanner count)
        num_usable_available = len(usable_carried) + sum(len(s) for s in usable_on_ground.values())
        if num_usable_available < len(nuts_to_tighten):
            return float('inf') # Not enough usable spanners

        # 4. Simulate tightening process
        h = 0
        current_man_loc = man_location
        # Use copies to simulate consumption during calculation
        usable_carried_copy = set(usable_carried)
        usable_on_ground_copy = collections.defaultdict(set)
        for loc, spanners in usable_on_ground.items():
            usable_on_ground_copy[loc].update(spanners)

        # Process nuts one by one (sort for determinism)
        for nut_name in sorted(list(nuts_to_tighten)):
            if nut_name not in self.nut_locations:
                 print(f"Warning: Location for nut '{nut_name}' not found.")
                 return float('inf') # Cannot proceed if nut location unknown
            nut_loc = self.nut_locations[nut_name]
            cost_this_nut = 0

            # Option 1: Use a carried usable spanner
            if usable_carried_copy:
                spanner_to_use = usable_carried_copy.pop() # Take one (arbitrary)
                
                # Walk from current location to nut location
                walk_dist = self.distances.get((current_man_loc, nut_loc), float('inf'))
                if walk_dist == float('inf'): return float('inf') # Unreachable

                cost_this_nut += walk_dist
                cost_this_nut += 1 # Tighten action

                # Update man's location for the next step in the heuristic calculation
                current_man_loc = nut_loc

            # Option 2: Pick up a usable spanner from the ground
            else:
                best_spanner = None
                best_spanner_loc = None
                min_pickup_cost = float('inf')

                # Find the spanner minimizing travel: man -> spanner -> nut
                possible_spanners = []
                for loc, spanners in usable_on_ground_copy.items():
                    if spanners:
                        possible_spanners.extend([(s, loc) for s in spanners])

                if not possible_spanners:
                    # This case should be caught by the initial check, but added for safety
                    return float('inf')

                found_path = False
                for spanner_name, spanner_loc in possible_spanners:
                    dist_man_to_spanner = self.distances.get((current_man_loc, spanner_loc), float('inf'))
                    dist_spanner_to_nut = self.distances.get((spanner_loc, nut_loc), float('inf'))

                    if dist_man_to_spanner == float('inf') or dist_spanner_to_nut == float('inf'):
                        continue # This spanner path is not viable

                    found_path = True
                    # Cost: walk to spanner + pickup + walk to nut
                    current_total_pickup_walk_cost = dist_man_to_spanner + 1 + dist_spanner_to_nut
                    if current_total_pickup_walk_cost < min_pickup_cost:
                        min_pickup_cost = current_total_pickup_walk_cost
                        best_spanner = spanner_name
                        best_spanner_loc = spanner_loc

                if not found_path:
                     # No usable spanner on the ground is reachable and can reach the nut
                     return float('inf')

                # Add costs for this nut: pickup sequence + tighten
                cost_this_nut += min_pickup_cost # Already includes walk + pickup + walk
                cost_this_nut += 1 # Tighten action

                # Update state for heuristic calculation
                usable_on_ground_copy[best_spanner_loc].remove(best_spanner)
                if not usable_on_ground_copy[best_spanner_loc]: # Clean up empty set entry
                     del usable_on_ground_copy[best_spanner_loc]
                current_man_loc = nut_loc # Man ends up at nut location

            h += cost_this_nut

        # Ensure heuristic is 0 only for goal states (already handled by initial check)
        # Ensure heuristic is finite for solvable states (returns inf if unsolvable detected)
        return int(round(h)) # Return integer estimate

