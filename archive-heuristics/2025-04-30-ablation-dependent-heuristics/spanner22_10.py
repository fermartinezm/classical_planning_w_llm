import collections

def parse_fact(fact_string):
    """Parses a PDDL fact string into a list of strings."""
    # Remove leading '(' and trailing ')' and potential outer quotes
    cleaned_string = fact_string.strip()
    if cleaned_string.startswith("'("):
         cleaned_string = cleaned_string[2:]
    elif cleaned_string.startswith("("):
         cleaned_string = cleaned_string[1:]

    if cleaned_string.endswith(")'"):
         cleaned_string = cleaned_string[:-2]
    elif cleaned_string.endswith(")"):
         cleaned_string = cleaned_string[:-1]

    # Split by space
    return cleaned_string.split()

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal by summing three components:
    1. The number of loose nuts that need to be tightened (minimum tighten actions).
    2. The minimum number of walk actions required to reach any location containing a loose goal nut from the man's current location.
    3. The number of additional usable spanners the man needs to pick up to tighten all remaining loose goal nuts (minimum pickup actions).

    Assumptions:
    - The man object name can be reliably identified (e.g., by looking for the object involved in 'carrying' facts in the initial state or operators).
    - Nuts do not move from their initial locations.
    - The location graph defined by 'link' facts is static.
    - All locations relevant to the problem (man's start, nut locations, spanner locations) are part of the connected graph or reachable in solvable problems.
    - There is only one man.
    - The state representation uses strings like "'(predicate arg1 arg2)'" or "(predicate arg1 arg2)".

    Heuristic Initialization:
    - Parses the goal facts.
    - Identifies the man object name.
    - Parses static 'link' facts to build the location graph.
    - Computes all-pairs shortest paths (distances) between locations using BFS.
    - Parses initial 'at' facts for nuts to store their fixed locations.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state. If yes, return 0.
    2. Identify all loose nuts that are part of the goal. Count them (`num_loose_goal_nuts`) and record their locations (`Locs_nut`).
    3. If there are no loose goal nuts, return 0 (all relevant nuts are already tightened).
    4. Find the man's current location (`loc_m`). If location is unknown or not in graph, return infinity.
    5. Calculate the minimum distance from `loc_m` to any location in `Locs_nut` (`min_dist_to_nut_loc`) using the precomputed distances. If no reachable nut location, return infinity.
    6. Check if the man is currently carrying a usable spanner (`num_carried_usable` is 1 if yes, 0 otherwise).
    7. Calculate the number of additional usable spanners the man needs to acquire: `needed_spanners = max(0, num_loose_goal_nuts - num_carried_usable)`.
    8. The heuristic value is the sum of these three components: `num_loose_goal_nuts + min_dist_to_nut_loc + needed_spanners`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Args:
            task: The planning task object.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Keep initial state to find nut locations

        # 1. Identify the man object name
        self.man_name = None
        # Look in initial state for carrying fact
        for fact_string in self.initial_state:
            if fact_string.startswith("'(carrying") or fact_string.startswith("(carrying"):
                parsed = parse_fact(fact_string)
                if len(parsed) > 1:
                    self.man_name = parsed[1]
                    break
        # If not found, look in operator names (less reliable, but covers cases)
        if self.man_name is None:
             for op in task.operators:
                 # Operator name format might vary, try common ones
                 # Example: (walk shed location1 bob)
                 # Example: (pickup_spanner location1 spanner1 bob)
                 # Example: (tighten_nut gate spanner1 bob nut1)
                 parsed_name = parse_fact(op.name)
                 # Man is typically the parameter typed 'man'
                 # In these operators, it's the 3rd parameter (index 2)
                 if parsed_name[0] in ('walk', 'pickup_spanner', 'tighten_nut'):
                     if len(parsed_name) > 2: # walk needs 3, pickup needs 3, tighten needs 4
                          self.man_name = parsed_name[2] # This is correct for walk/pickup/tighten
                          break # Assuming one man

        if self.man_name is None:
             # Fallback: Assume a common name like 'bob' if detection fails.
             # This is based on the provided examples.
             self.man_name = 'bob'


        # 2. Build location graph and compute distances
        self.locations = set()
        links = []
        for fact_string in self.static:
            if fact_string.startswith("'(link") or fact_string.startswith("(link"):
                parsed = parse_fact(fact_string)
                if len(parsed) == 3:
                    l1, l2 = parsed[1], parsed[2]
                    self.locations.add(l1)
                    self.locations.add(l2)
                    links.append((l1, l2))

        self.graph = {loc: set() for loc in self.locations}
        for l1, l2 in links:
            self.graph[l1].add(l2)
            self.graph[l2].add(l1) # Links are bidirectional

        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = self._bfs(start_node)

        # 3. Store nut locations (nuts are static)
        self.nut_locations = {}
        # Identify potential nut objects by looking at goals and initial state facts
        potential_nut_objects = set()
        for fact_string in self.goals:
             parsed = parse_fact(fact_string)
             if parsed[0] == 'tightened' and len(parsed) > 1:
                  potential_nut_objects.add(parsed[1])
        for fact_string in self.initial_state:
             parsed = parse_fact(fact_string)
             if parsed[0] in ('loose', 'tightened') and len(parsed) > 1:
                  potential_nut_objects.add(parsed[1])

        # Find locations for these potential nut objects in the initial state
        for fact_string in self.initial_state:
             if fact_string.startswith("'(at") or fact_string.startswith("(at"):
                  parsed = parse_fact(fact_string)
                  if len(parsed) == 3:
                       obj, loc = parsed[1], parsed[2]
                       if obj in potential_nut_objects:
                            self.nut_locations[obj] = loc


    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not in the graph, no distances can be computed
             return distances

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node not in self.graph:
                 continue # Should not happen if locations are from links

            for neighbor in self.graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for a given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Identify loose goal nuts and their locations
        num_loose_goal_nuts = 0
        Locs_nut = set()
        loose_facts = {f for f in state if f.startswith("'(loose") or f.startswith("(loose")}

        for goal_fact in self.goals:
            # goal_fact is like '(tightened nut1)'
            parsed_goal = parse_fact(goal_fact)
            if parsed_goal[0] == 'tightened' and len(parsed_goal) > 1:
                nut_name = parsed_goal[1]
                # Check if this nut is currently loose in the state
                if f"'(loose {nut_name})'" in loose_facts or f"(loose {nut_name})" in loose_facts:
                    num_loose_goal_nuts += 1
                    # Get the location of this nut (precomputed)
                    if nut_name in self.nut_locations:
                         Locs_nut.add(self.nut_locations[nut_name])
                    # else: This nut doesn't have a location? Problematic. Assume nuts have locations.
                    # If a nut location is not found, we cannot calculate distance, return inf?
                    # For now, assume all goal nuts have precomputed locations.

        # 3. If no loose goal nuts, goal is effectively reached for relevant nuts
        if num_loose_goal_nuts == 0:
             return 0 # All goal nuts are already tightened

        # 4. Find man's current location
        loc_m = None
        for fact_string in state:
            if fact_string.startswith("'(at") or fact_string.startswith("(at"):
                parsed = parse_fact(fact_string)
                if len(parsed) == 3 and parsed[1] == self.man_name:
                    loc_m = parsed[2]
                    break
        # If man's location not found or not in graph, return infinity
        if loc_m is None or loc_m not in self.dist:
             # print(f"Warning: Man {self.man_name} location {loc_m} not found in graph/state.")
             return float('inf')

        # 5. Calculate minimum distance to a loose goal nut location
        min_dist_to_nut_loc = float('inf')
        reachable_nut_locations = [ln for ln in Locs_nut if ln in self.dist[loc_m] and self.dist[loc_m][ln] != float('inf')]

        if not reachable_nut_locations:
             # If there are loose goal nuts but none are reachable, return infinity
             if num_loose_goal_nuts > 0:
                 # print(f"Warning: No reachable loose goal nut location from {loc_m}.")
                 return float('inf')
             else:
                 # This case should be caught by num_loose_goal_nuts == 0 check earlier, but defensive
                 return 0


        min_dist_to_nut_loc = min(self.dist[loc_m][ln] for ln in reachable_nut_locations)


        # 6. Check if man is carrying a usable spanner
        num_carried_usable = 0
        carried_spanner = None
        for fact_string in state:
             if fact_string.startswith("'(carrying") or fact_string.startswith("(carrying"):
                  parsed = parse_fact(fact_string)
                  if len(parsed) == 3 and parsed[1] == self.man_name:
                       carried_spanner = parsed[2]
                       break # Assuming man carries at most one spanner

        if carried_spanner is not None:
             if f"'(usable {carried_spanner})'" in state or f"(usable {carried_spanner})" in state:
                  num_carried_usable = 1

        # 7. Calculate needed spanners
        needed_spanners = max(0, num_loose_goal_nuts - num_carried_usable)

        # 8. Compute heuristic value
        # Heuristic = num_tighten_actions + num_walk_actions + num_pickup_actions
        # num_tighten_actions = num_loose_goal_nuts
        # num_walk_actions >= min_dist_to_nut_loc (first step towards a nut)
        # num_pickup_actions >= needed_spanners
        heuristic_value = num_loose_goal_nuts + min_dist_to_nut_loc + needed_spanners

        return heuristic_value
