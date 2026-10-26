import math

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the remaining cost to reach the goal state (all goal nuts tightened)
    by summing three main components:
    1. The number of loose goal nuts: Each loose goal nut requires at least one `tighten_nut` action.
    2. The travel cost for the man to reach the closest location containing a loose goal nut: This estimates the minimum travel needed to get to where the first task can be performed.
    3. The cost to obtain a usable spanner if the man is not currently carrying one: This cost includes one `pickup_spanner` action plus the travel cost to reach the closest location with an available usable spanner.

    The heuristic is designed for greedy best-first search and is not admissible; it may overestimate the true cost but aims to guide the search effectively by prioritizing states where goal conditions are closer or resources are readily available.

    Assumptions:
    - There is exactly one man in the domain.
    - Nut locations are static and are provided in the initial state.
    - Spanner usability is static in the sense that a spanner is either initially usable or not; it becomes unusable after one use but cannot become usable again.
    - The problem is solvable, implying there are enough usable spanners initially across all locations to tighten all goal nuts.
    - Locations mentioned in initial state facts (man's start, spanner locations, nut locations) and static link facts constitute all relevant locations in the problem graph.

    Heuristic Initialization:
    The constructor performs precomputation based on the static task information:
    - It identifies the man's name, all spanner names, nut names, and location names by parsing facts in the initial state, goals, and static information.
    - It extracts the static locations of all nuts from the initial state.
    - It stores the names of the goal nuts from the task's goal conditions.
    - It builds a graph representation of the locations using the static `link` facts.
    - It computes all-pairs shortest path distances between all known locations using Breadth-First Search (BFS). These distances are stored for quick lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic (__call__ method):
    1.  Extract the man's current location from the input state. If the man's location cannot be found, the state is invalid or unsolvable, return infinity.
    2.  Determine if the man is currently carrying a spanner and if that spanner is usable in the current state.
    3.  Identify all goal nuts that are currently in a `(loose ?n)` state by checking the input state against the precomputed set of goal nut names.
    4.  Count the number of loose goal nuts (`num_loose_nuts`).
    5.  If `num_loose_nuts` is 0, the goal state has been reached, so the heuristic value is 0.
    6.  Initialize the heuristic value `h` with `num_loose_nuts`. This accounts for the minimum number of `tighten_nut` actions required.
    7.  Find the locations of all loose goal nuts using the precomputed `nut_locations` dictionary.
    8.  Calculate the minimum shortest path distance from the man's current location to any of the locations containing a loose goal nut. Add this minimum distance to `h`. This estimates the travel cost to reach the vicinity of the first required task. If no loose nut location is reachable, the state is unsolvable, return infinity.
    9.  Check if the man is *not* currently carrying a usable spanner.
    10. If a usable spanner is needed:
        a. Add 1 to `h` to account for the necessary `pickup_spanner` action.
        b. Identify all usable spanners that are currently at a location (i.e., not being carried by the man) by examining the input state.
        c. Find the locations of these available usable spanners.
        d. Calculate the minimum shortest path distance from the man's current location to any of these available usable spanner locations. Add this minimum distance to `h`. This estimates the travel cost to acquire a spanner. If no usable spanner is available at a reachable location, the state is unsolvable, return infinity.
    11. Return the final calculated value of `h`.
    """
    def __init__(self, task):
        """
        Initializes the spanner heuristic by precomputing static information.

        @param task: The planning task object containing initial state, goals, operators, and static facts.
        """
        self.task = task
        self.man_name = None
        self.spanner_names = set()
        self.nut_names = set()
        self.location_names = set()
        self.nut_locations = {} # {nut_name: location}
        self.goal_nut_names = set()

        # Infer object names and types by examining all facts in initial state, goals, and static info
        all_relevant_facts = set(task.initial_state) | set(task.static) | set(task.goals)

        # First pass: Collect all potential objects and locations based on predicate structure
        for fact_str in all_relevant_facts:
            parts = fact_str.strip('()').split()
            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assume second argument of 'at' is always a location
                self.location_names.add(loc)
                # Assume first argument of 'at' is a locatable object (man, spanner, nut)
                # We'll refine type below
            elif predicate == 'carrying' and len(parts) == 3:
                man, spanner = parts[1], parts[2]
                # Assume the carrier in 'carrying' is the man
                self.man_name = man
                self.spanner_names.add(spanner)
            elif predicate == 'usable' and len(parts) == 2:
                spanner = parts[1]
                self.spanner_names.add(spanner)
            elif predicate == 'loose' and len(parts) == 2:
                nut = parts[1]
                self.nut_names.add(nut)
            elif predicate == 'tightened' and len(parts) == 2:
                nut = parts[1]
                self.nut_names.add(nut)
            elif predicate == 'link' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                # Assume arguments of 'link' are locations
                self.location_names.add(loc1)
                self.location_names.add(loc2)

        # Extract nut locations from initial state (assuming they are there and static)
        for fact in task.initial_state:
             if fact.startswith('(at '):
                 parts = fact.strip('()').split()
                 obj = parts[1]
                 loc = parts[2]
                 if obj in self.nut_names:
                     self.nut_locations[obj] = loc

        # Extract goal nut names
        for goal_fact in task.goals:
            if goal_fact.startswith('(tightened '):
                self.goal_nut_names.add(goal_fact.strip('()').split()[1])

        # Build location graph (adjacency list)
        self.location_links = {} # adjacency list
        for fact in task.static:
            if fact.startswith('(link '):
                parts = fact.strip('()').split()
                loc1 = parts[1]
                loc2 = parts[2]
                # Ensure locations from links are added (already done, but defensive)
                self.location_names.add(loc1)
                self.location_names.add(loc2)
                self.location_links.setdefault(loc1, set()).add(loc2)
                self.location_links.setdefault(loc2, set()).add(loc1) # links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.location_distances = {} # dict mapping (loc1, loc2) to distance

        for start_loc in self.location_names:
            q = [(start_loc, 0)]
            visited = {start_loc}
            self.location_distances[(start_loc, start_loc)] = 0
            while q:
                current_loc, dist = q.pop(0)
                # Ensure current_loc is in links before accessing
                if current_loc in self.location_links:
                    for neighbor in self.location_links[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.location_distances[(start_loc, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returns inf if no path."""
        if loc1 is None or loc2 is None:
             return float('inf')
        if loc1 == loc2:
            return 0
        # Use .get() with default inf for safety
        return self.location_distances.get((loc1, loc2), float('inf'))

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state represented as a frozenset of strings.
        @return: The estimated cost to reach the goal, or float('inf') if unsolvable.
        """
        # Extract info from state
        man_loc = None
        carrying_spanner_name = None
        usable_spanner_at_locations = {} # {spanner_name: location}
        is_usable_in_state = {} # {spanner_name: boolean}

        for fact in state:
            if fact.startswith('(at '):
                parts = fact.strip('()').split()
                if len(parts) == 3:
                    obj = parts[1]
                    loc = parts[2]
                    if obj == self.man_name:
                        man_loc = loc
                    elif obj in self.spanner_names:
                        # Store location of spanners that are 'at' a location
                        usable_spanner_at_locations[obj] = loc
            elif fact.startswith('(carrying '):
                parts = fact.strip('()').split()
                if len(parts) == 3:
                    carrier = parts[1]
                    spanner = parts[2]
                    if carrier == self.man_name and spanner in self.spanner_names:
                         carrying_spanner_name = spanner
            elif fact.startswith('(usable '):
                 parts = fact.strip('()').split()
                 if len(parts) == 2:
                     spanner = parts[1]
                     if spanner in self.spanner_names:
                         is_usable_in_state[spanner] = True
            # loose facts are handled when identifying loose goal nuts

        # Check if man_loc was found (should always be in a valid state)
        if man_loc is None:
             return float('inf') # Should not happen in a reachable state

        # Identify loose goal nuts
        loose_goal_nut_facts = {f for f in state if f.startswith('(loose ') and f.strip('()').split()[1] in self.goal_nut_names}
        num_loose_nuts = len(loose_goal_nut_facts)

        # If goal reached
        if num_loose_nuts == 0:
            return 0

        h = num_loose_nuts # Base cost: one tighten action per nut

        # Find locations of loose goal nuts
        loose_nut_locations = set()
        for nut_fact in loose_goal_nut_facts:
             nut_name = nut_fact.strip('()').split()[1]
             # Get nut location from precomputed static info
             if nut_name in self.nut_locations:
                 loose_nut_locations.add(self.nut_locations[nut_name])
             # else: This nut's location is unknown? Problematic. Assume nut_locations is complete.

        # Cost to reach a nut location
        if loose_nut_locations:
            min_dist_to_nut = float('inf')
            for nut_loc in loose_nut_locations:
                dist = self.get_distance(man_loc, nut_loc)
                min_dist_to_nut = min(min_dist_to_nut, dist)

            if min_dist_to_nut == float('inf'):
                 # Cannot reach any nut location - unsolvable
                 return float('inf')
            else:
                 h += min_dist_to_nut
        # else: num_loose_nuts > 0 but no loose_nut_locations? Should not happen if nut_locations is complete.

        # Cost to get a spanner if needed
        is_man_carrying_usable = False
        if carrying_spanner_name and is_usable_in_state.get(carrying_spanner_name, False):
            is_man_carrying_usable = True

        if not is_man_carrying_usable:
            h += 1 # Cost for pickup action

            # Find available usable spanners at locations
            available_usable_spanner_locations = set()
            for s_name, s_loc in usable_spanner_at_locations.items():
                # Check if spanner is usable in the current state
                if is_usable_in_state.get(s_name, False):
                     available_usable_spanner_locations.add(s_loc)

            if available_usable_spanner_locations:
                min_dist_to_spanner = float('inf')
                for spanner_loc in available_usable_spanner_locations:
                     dist = self.get_distance(man_loc, spanner_loc)
                     min_dist_to_spanner = min(min_dist_to_spanner, dist)

                if min_dist_to_spanner == float('inf'):
                     # Cannot reach any available spanner location - unsolvable
                     return float('inf')
                else:
                     h += min_dist_to_spanner
            else:
                # No usable spanners available anywhere (not carried, not at location)
                # This state is unsolvable if more nuts need tightening
                # (which is true because num_loose_nuts > 0)
                return float('inf')

        return h

