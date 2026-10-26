import collections
import math

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the total number of actions required to reach the goal
    by summing the estimated costs for tightening each loose goal nut independently.
    The cost for a single nut is estimated as the minimum actions needed to get the man
    to the nut's location with a usable spanner, plus the tighten action.

    Assumptions:
    - The problem is solvable.
    - The location graph is connected for all relevant locations (man, nuts, spanners).
    - The man can carry at most one spanner at a time (implied by domain).
    - A spanner becomes permanently unusable after one tighten action.
    - Object names follow conventions (e.g., 'bob' for man, 'spannerX', 'nutX').
    - Facts are represented as strings like '(predicate obj1 obj2)'.

    Heuristic Initialization:
    - Parses static facts (`task.static`) to build the location graph based on 'link' predicates.
    - Parses initial state facts (`task.initial_state`) to identify the man, spanners, nuts, and locations.
    - Computes all-pairs shortest path distances between all identified locations using BFS.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the set of goal nuts that are currently 'loose' in the state (`UntightenedGoals`).
    2. If `UntightenedGoals` is empty, the goal is reached, return 0.
    3. Find the man's current location (`ManLoc`) from the state.
    4. Determine if the man is currently carrying a spanner that is 'usable' from the state.
    5. Find the current locations of all spanners that are 'usable' from the state.
    6. Find the current locations of all nuts from the state.
    7. Initialize total heuristic value to 0.
    8. For each nut `n` in `UntightenedGoals`:
        a. Get the location of nut `n` (`NutLoc`).
        b. Calculate the minimum estimated cost to get the man to `NutLoc` carrying a usable spanner:
           - If man is currently carrying a usable spanner: The cost is the distance from `ManLoc` to `NutLoc` (`dist(ManLoc, NutLoc)`).
           - If man is NOT currently carrying a usable spanner: He needs to acquire one. The minimum cost to do this and reach `NutLoc` is the minimum over all usable spanners `s` at locations `L_s` of the cost to go from `ManLoc` to `L_s`, pick up `s`, and then go from `L_s` to `NutLoc`. This cost is `dist(ManLoc, L_s)` (walk to spanner) + 1 (pickup) + `dist(L_s, NutLoc)` (carry to nut).
        c. Add the minimum estimated cost calculated in 8b plus 1 (for the 'tighten_nut' action) to the total heuristic value.
    9. Return the total heuristic value.
    """
    def __init__(self, task):
        """
        Initializes the heuristic. Precomputes distances between locations.
        """
        self.task = task
        self.man_name = None
        self.location_names = set()
        self.spanner_names = set()
        self.nut_names = set()
        self.link_graph = collections.defaultdict(set)
        self.distances = {} # Stores shortest path distances between locations

        # Extract object names and links from static and initial state
        self._process_static_info(task.static)
        self._process_initial_state_for_objects(task.initial_state)

        # Ensure all locations mentioned in links or initial state are in the set
        for loc1, neighbors in self.link_graph.items():
            self.location_names.add(loc1)
            self.location_names.update(neighbors)
        # Add locations from initial state facts like (at obj loc)
        for fact in task.initial_state:
             pred, objs = self._parse_fact(fact)
             if pred == 'at' and len(objs) == 2:
                 self.location_names.add(objs[1])

        # Compute distances
        self._compute_all_pairs_shortest_paths()

    def _parse_fact(self, fact_string):
        """Helper to parse PDDL fact string."""
        # Removes outer parentheses and splits by space
        parts = fact_string[1:-1].split()
        if not parts: # Handle empty fact string like '()'
            return None, []
        predicate = parts[0]
        objects = parts[1:]
        return predicate, objects

    def _process_static_info(self, static_facts):
        """Extracts static information like links."""
        for fact in static_facts:
            pred, objs = self._parse_fact(fact)
            if pred == 'link' and len(objs) == 2:
                loc1, loc2 = objs
                self.link_graph[loc1].add(loc2)
                self.link_graph[loc2].add(loc1) # Links are bidirectional

    def _process_initial_state_for_objects(self, initial_state_facts):
        """Extracts object names and types from initial state."""
        # This is a simplified approach assuming typical naming conventions
        # A full PDDL parser would be more robust.
        for fact in initial_state_facts:
            pred, objs = self._parse_fact(fact)
            if pred == 'at' and len(objs) == 2:
                obj_name, loc_name = objs
                # Infer type based on naming convention
                if 'bob' in obj_name.lower() or 'man' in obj_name.lower():
                    self.man_name = obj_name
                elif 'spanner' in obj_name.lower():
                    self.spanner_names.add(obj_name)
                elif 'nut' in obj_name.lower():
                    self.nut_names.add(obj_name)
            elif pred == 'carrying' and len(objs) == 2:
                 self.man_name = objs[0]
                 self.spanner_names.add(objs[1])
            elif pred == 'usable' and len(objs) == 1:
                 self.spanner_names.add(objs[0])
            elif (pred == 'loose' or pred == 'tightened') and len(objs) == 1:
                 self.nut_names.add(objs[0])

        # Ensure man_name is found (critical for heuristic)
        if self.man_name is None:
             # print("Error: Man object not identified.")
             # This case indicates a problem with the input format or domain definition
             # In a real scenario, proper PDDL parsing is needed.
             # For this problem, we assume man is identifiable.
             pass


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all locations using BFS."""
        locations = list(self.location_names)
        for start_loc in locations:
            self.distances[start_loc] = {}
            q = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                current_loc, dist = q.popleft()
                self.distances[start_loc][current_loc] = dist
                for neighbor in self.link_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))

    def _get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Return infinity if locations are not in the computed distances (e.g., disconnected graph)
        # or if loc1/loc2 are not valid locations found during init.
        return self.distances.get(loc1, {}).get(loc2, math.inf)


    def __call__(self, state, task):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        untightened_goals = set()
        for goal_fact in task.goals:
            pred, objs = self._parse_fact(goal_fact)
            if pred == 'tightened' and len(objs) == 1:
                nut_name = objs[0]
                # Check if this nut is currently loose in the state
                if f'(loose {nut_name})' in state:
                     untightened_goals.add(nut_name)

        if not untightened_goals:
            return 0

        # Get man's location
        man_loc = None
        for fact in state:
            pred, objs = self._parse_fact(fact)
            if pred == 'at' and len(objs) == 2 and objs[0] == self.man_name:
                man_loc = objs[1]
                break
        # If man_loc is not found, something is wrong with the state representation
        if man_loc is None:
             # print(f"Error: Man '{self.man_name}' location not found in state.")
             return math.inf # Should not happen in valid states

        # Check if man is carrying a usable spanner
        man_carrying_usable_spanner = False
        carried_spanner_name = None
        for fact in state:
            pred, objs = self._parse_fact(fact)
            if pred == 'carrying' and len(objs) == 2 and objs[0] == self.man_name:
                carried_spanner_name = objs[1]
                if f'(usable {carried_spanner_name})' in state:
                    man_carrying_usable_spanner = True
                break # Assuming man carries at most one spanner

        # Get locations of all usable spanners
        usable_spanner_locs = {} # {spanner_name: location_name}
        for fact in state:
            pred, objs = self._parse_fact(fact)
            if pred == 'at' and len(objs) == 2 and objs[0] in self.spanner_names:
                spanner_name = objs[0]
                spanner_loc = objs[1]
                if f'(usable {spanner_name})' in state:
                    usable_spanner_locs[spanner_name] = spanner_loc

        # Get locations of all nuts
        nut_locs = {} # {nut_name: location_name}
        for fact in state:
            pred, objs = self._parse_fact(fact)
            if pred == 'at' and len(objs) == 2 and objs[0] in self.nut_names:
                nut_name = objs[0]
                nut_loc = objs[1]
                nut_locs[nut_name] = nut_loc

        total_heuristic = 0

        for nut_name in untightened_goals:
            nut_loc = nut_locs.get(nut_name)
            if nut_loc is None:
                 # Should not happen for goal nuts in valid problems
                 # print(f"Error: Location for nut '{nut_name}' not found in state.")
                 return math.inf # Indicate problem

            # Calculate minimum cost to get man to nut_loc carrying a usable spanner
            cost_get_man_spanner_to_nut_loc = math.inf

            # Case 1: Man is carrying a usable spanner
            if man_carrying_usable_spanner:
                # Cost is just walking to the nut location
                cost_get_man_spanner_to_nut_loc = self._get_distance(man_loc, nut_loc)

            # Case 2: Man is not carrying a usable spanner
            else:
                # Find the minimum cost to fetch *any* usable spanner and bring it to nut_loc
                min_cost_fetch_and_carry = math.inf
                # Iterate through all usable spanners
                for s_name, s_loc in usable_spanner_locs.items():
                     # Cost = walk from man_loc to spanner_loc + pickup + carry from spanner_loc to nut_loc
                     cost_fetch_and_carry = self._get_distance(man_loc, s_loc) + 1 + self._get_distance(s_loc, nut_loc)
                     min_cost_fetch_and_carry = min(min_cost_fetch_and_carry, cost_fetch_and_carry)

                cost_get_man_spanner_to_nut_loc = min_cost_fetch_and_carry


            # Total cost for this nut = cost to get man/spanner to nut_loc + tighten action
            # If cost_get_man_spanner_to_nut_loc is still inf, it means no path/spanner exists.
            if cost_get_man_spanner_to_nut_loc == math.inf:
                 # This nut is unreachable with a usable spanner from the current state.
                 return math.inf

            total_heuristic += cost_get_man_spanner_to_nut_loc + 1 # Add 1 for the tighten action

        return total_heuristic
