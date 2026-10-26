import collections

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the total cost to tighten all goal nuts that are currently loose.
    For each loose goal nut, it estimates the cost as the sum of:
    1. The shortest path distance for the man to reach the nut's location.
    2. The cost to acquire a usable spanner if the man is not already carrying one.
       If a usable spanner is needed, this cost is estimated as the minimum cost
       to travel from the man's current location to a location with an available
       usable spanner, pick it up (1 action), and then travel from that spanner's
       location to the nut's location. If no usable spanners are available anywhere,
       a large penalty is added for this nut, indicating it's likely unreachable.

    Assumptions:
    - The domain structure is as defined in the PDDL file.
    - There is exactly one man object, and its name is 'bob' (based on examples).
    - Nut locations are static and provided in the initial state via '(at nutX locationY)' facts.
    - Goal nuts are specified using the '(tightened ?nut)' predicate in the task goals.
    - Usable spanners are required to tighten nuts and become unusable after one use.
    - The man can carry multiple spanners.
    - The location graph defined by 'link' facts is used for movement costs.
    - All locations mentioned in 'link' or initial 'at' facts are considered.
    - Unreachable locations or lack of necessary items (usable spanners) results in a high heuristic value for the affected goal nut.

    Heuristic Initialization:
    - Parses static facts to build the location graph representing connections between locations.
    - Identifies all unique locations mentioned in static 'link' facts and initial 'at' facts.
    - Computes shortest path distances between all pairs of identified locations using Breadth-First Search (BFS). These distances represent the minimum number of 'walk' actions required.
    - Identifies the man's name (hardcoded as 'bob').
    - Parses the initial state to record the static locations of all nuts.
    - Parses the task's goal facts to identify the set of nuts that need to be tightened.

    Step-By-Step Thinking for Computing Heuristic (for a given state):
    1. Initialize the total heuristic value `h_value` to 0.
    2. Define a large penalty value (e.g., 1000) to represent high cost or unsolvability for a single nut.
    3. Parse the current state to extract dynamic information:
       - The man's current location.
       - The set of spanners the man is currently carrying.
       - The set of spanners that are currently usable (anywhere).
       - A mapping of spanners currently at locations to their respective locations.
       - The set of nuts that are currently loose.
       - The set of nuts that are currently tightened.
    4. Determine which usable spanners are currently being carried by the man.
    5. Determine which usable spanners are currently available at locations, along with their locations.
    6. Iterate through each nut that is part of the task's goal set (`self.goal_nuts`).
    7. For the current goal nut `n`:
       - Check if `n` is present in the set of `tightened_nuts` extracted from the current state. If it is, this goal is achieved for this nut, and its contribution to the heuristic is 0. Continue to the next goal nut.
       - If `n` is not tightened (i.e., it is loose), calculate its estimated cost:
         - Retrieve the static location of nut `n` (`self.nut_locations.get(n)`). If the location is unknown (should not happen in valid problems), add the large penalty and skip this nut. Let `nut_location` be this location.
         - Retrieve the man's current location (`man_location`). If the man's location is unknown (should not happen), add the large penalty and skip this nut.
         - Determine the cost for this nut:
           - If the man is currently carrying *any* spanner that is also marked as usable (`usable_spanners_carried` is not empty), the spanner acquisition cost for this nut is effectively 0. The total estimated cost for this nut is the shortest path distance from the man's current location to the nut's location (`self.get_distance(man_location, nut_location)`). If the nut location is unreachable, use the large penalty.
           - If the man is *not* carrying any usable spanner:
             - The man needs to pick one up. Find usable spanners `s` that are currently at locations `loc_s` (`usable_spanners_at_loc`).
             - If there are usable spanners available at locations:
               - Calculate the cost of a path that involves picking up a spanner: `Distance(man_location, loc_s) + 1 (pickup action) + Distance(loc_s, nut_location)`.
               - Find the minimum such path cost over all available usable spanners `s` at locations `loc_s`. Let this be `min_path_via_spanner`.
               - If a valid path via a spanner exists (`min_path_via_spanner` is not infinity), the estimated cost for this nut is `min_path_via_spanner`.
               - If no usable spanners are available at locations (and the man isn't carrying one), this nut cannot be tightened from this state. The estimated cost for this nut is the `large_penalty`.
             - If there are no usable spanners available at locations (and the man isn't carrying one), this nut cannot be tightened. The estimated cost for this nut is the `large_penalty`.
         - Add the calculated estimated cost for nut `n` to the `h_value`.
    8. Return the total `h_value`.
    """
    def __init__(self, task):
        """
        Initializes the spanner heuristic.

        Heuristic Initialization:
        - Parses static facts to build the location graph.
        - Computes shortest path distances between all pairs of locations using BFS.
        - Identifies the man's name (assumed to be 'bob' based on domain examples).
        - Identifies the locations of all nuts from the initial state (nut locations are static).
        - Identifies the set of goal nuts from the task's goal facts.
        """
        self.task = task
        self.location_graph = collections.defaultdict(set)
        self.locations = set()
        self.all_pairs_distances = {} # {(loc1, loc2): distance}
        self.man_name = 'bob' # Assume man is named 'bob' based on examples

        # Parse static facts to build location graph
        for fact_str in task.static:
            pred, *args = self._parse_fact(fact_str)
            if pred == 'link':
                loc1, loc2 = args
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1)
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Parse initial state to find nut locations and collect all locations
        self.nut_locations = {} # {nut_name: location_name}
        all_mentioned_locations = set(self.locations) # Start with locations from links

        for fact_str in task.initial_state:
            pred, *args = self._parse_fact(fact_str)
            if pred == 'at':
                obj, loc = args
                all_mentioned_locations.add(loc) # Add location from 'at' fact
                if obj.startswith('nut'): # Assume nuts start with 'nut'
                    self.nut_locations[obj] = loc
                # We don't need initial spanner/man locations here, they are dynamic.

        self.locations = all_mentioned_locations # Update self.locations to include all relevant ones

        # Compute all-pairs shortest paths
        self.all_pairs_distances = {}
        for start_loc in self.locations:
            self.all_pairs_distances[start_loc] = self._bfs(start_loc)

        # Identify goal nuts
        self.goal_nuts = set()
        # The task.goals is a frozenset of goal facts
        for fact_str in task.goals:
             pred, *args = self._parse_fact(fact_str)
             if pred == 'tightened':
                 nut_name = args[0]
                 self.goal_nuts.add(nut_name)

        # Ensure all goal nuts have known locations (checked during h calculation)


    def _parse_fact(self, fact_string):
        """Helper to parse a PDDL fact string into a tuple."""
        # Removes leading/trailing parentheses and splits by space
        # Handles cases like '(at obj loc)'
        parts = fact_string[1:-1].split()
        # Remove potential empty strings from split and strip whitespace
        parts = [p.strip() for p in parts if p.strip()]
        return tuple(parts)

    def _bfs(self, start_node):
        """Performs BFS from start_node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not in the known locations, cannot reach anything
             return distances

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Ensure current_node is in the graph keys before accessing neighbors
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Gets the precomputed shortest distance between two locations."""
        if loc1 not in self.locations or loc2 not in self.locations:
             # One or both locations are not in the known set of locations
             return float('inf')
        # BFS result is stored as distances from start_loc to others
        return self.all_pairs_distances.get(loc1, {}).get(loc2, float('inf'))


    def h(self, state):
        """
        Computes the domain-dependent heuristic for the spanner domain.

        Summary:
        The heuristic estimates the total cost to tighten all goal nuts that are currently loose.
        For each loose goal nut, it estimates the cost as the sum of:
        1. The shortest path distance for the man to reach the nut's location.
        2. The cost to acquire a usable spanner if the man is not already carrying one.
           If a usable spanner is needed, this cost is estimated as the minimum cost
           to travel from the man's current location to a location with an available
           usable spanner, pick it up (1 action), and then travel from that spanner's
           location to the nut's location. If no usable spanners are available anywhere,
           a large penalty is added for this nut, indicating it's likely unreachable.

        Assumptions:
        - The domain structure is as defined in the PDDL file.
        - There is exactly one man object, and its name is 'bob' (based on examples).
        - Nut locations are static and provided in the initial state via '(at nutX locationY)' facts.
        - Goal nuts are specified using the '(tightened ?nut)' predicate in the task goals.
        - Usable spanners are required to tighten nuts and become unusable after one use.
        - The man can carry multiple spanners.
        - The location graph defined by 'link' facts is used for movement costs.
        - All locations mentioned in 'link' or initial 'at' facts are considered.
        - Unreachable locations or lack of necessary items (usable spanners) results in a high heuristic value for the affected goal nut.

        Step-By-Step Thinking for Computing Heuristic (for a given state):
        1. Initialize the total heuristic value `h_value` to 0.
        2. Define a large penalty value (e.g., 1000) to represent high cost or unsolvability for a single nut.
        3. Parse the current state to extract dynamic information:
           - The man's current location.
           - The set of spanners the man is currently carrying.
           - The set of spanners that are currently usable (anywhere).
           - A mapping of spanners currently at locations to their respective locations.
           - The set of nuts that are currently loose.
           - The set of nuts that are currently tightened.
        4. Determine which usable spanners are currently being carried by the man.
        5. Determine which usable spanners are currently available at locations, along with their locations.
        6. Iterate through each nut that is part of the task's goal set (`self.goal_nuts`).
        7. For the current goal nut `n`:
           - Check if `n` is present in the set of `tightened_nuts` extracted from the current state. If it is, this goal is achieved for this nut, and its contribution to the heuristic is 0. Continue to the next goal nut.
           - If `n` is not tightened (i.e., it is loose), calculate its estimated cost:
             - Retrieve the static location of nut `n` (`self.nut_locations.get(n)`). If the location is unknown (should not happen in valid problems), add the large penalty and skip this nut. Let `nut_location` be this location.
             - Retrieve the man's current location (`man_location`). If the man's location is unknown (should not happen), add the large penalty and skip this nut.
             - Determine the cost for this nut:
               - If the man is currently carrying *any* spanner that is also marked as usable (`usable_spanners_carried` is not empty), the spanner acquisition cost for this nut is effectively 0. The total estimated cost for this nut is the shortest path distance from the man's current location to the nut's location (`self.get_distance(man_location, nut_location)`). If the nut location is unreachable, use the large penalty.
               - If the man is *not* carrying any usable spanner:
                 - The man needs to pick one up. Find usable spanners `s` that are currently at locations `loc_s` (`usable_spanners_at_loc`).
                 - If there are usable spanners available at locations:
                   - Calculate the cost of a path that involves picking up a spanner: `Distance(man_location, loc_s) + 1 (pickup action) + Distance(loc_s, nut_location)`.
                   - Find the minimum such path cost over all available usable spanners `s` at locations `loc_s`. Let this be `min_path_via_spanner`.
                   - If a valid path via a spanner exists (`min_path_via_spanner` is not infinity), the estimated cost for this nut is `min_path_via_spanner`.
                   - If no usable spanners are available at locations (and the man isn't carrying one), this nut cannot be tightened from this state. The estimated cost for this nut is the `large_penalty`.
                 - If there are no usable spanners available at locations (and the man isn't carrying one), this nut cannot be tightened. The estimated cost for this nut is the `large_penalty`.
             - Add the calculated estimated cost for nut `n` to the `h_value`.
    8. Return the total `h_value`.
    """
        h_value = 0
        large_penalty = 1000 # Use a large finite number for unreachable goals

        # --- Step 3: Parse the current state ---
        man_location = None
        carried_spanners = set()
        usable_spanners_in_state = set() # Usable spanners anywhere
        spanners_at_locations = {} # {spanner_name: location_name}
        loose_nuts = set()
        tightened_nuts = set()

        for fact_str in state:
            pred, *args = self._parse_fact(fact_str)
            if pred == 'at':
                obj, loc = args
                if obj == self.man_name:
                    man_location = loc
                elif obj.startswith('spanner'):
                    spanners_at_locations[obj] = loc
                # Nut locations are static, so we use self.nut_locations
            elif pred == 'carrying':
                # Assuming the first arg is the man
                if args[0] == self.man_name:
                    carried_spanners.add(args[1])
            elif pred == 'usable':
                usable_spanners_in_state.add(args[0])
            elif pred == 'loose':
                loose_nuts.add(args[0])
            elif pred == 'tightened':
                tightened_nuts.add(args[0])

        # --- Step 4 & 5: Identify usable spanners ---
        usable_spanners_carried = carried_spanners.intersection(usable_spanners_in_state)
        usable_spanners_at_loc = {
            s: loc for s, loc in spanners_at_locations.items()
            if s in usable_spanners_in_state
        }

        # --- Step 6 & 7: Iterate through goal nuts and compute cost ---
        for nut in self.goal_nuts:
            # Check if the nut is already tightened
            if nut in tightened_nuts:
                continue # Cost for this nut is 0

            # Nut is loose, calculate its cost
            nut_location = self.nut_locations.get(nut)
            if nut_location is None:
                 # Should not happen in valid problems, but handle defensively
                 h_value += large_penalty
                 continue # Cannot reach a nut whose location is unknown

            # Retrieve man's current location. If unknown, problem state is weird.
            if man_location is None:
                 h_value += large_penalty
                 continue # Man's location is unknown

            # Determine the cost for this nut
            cost_for_this_nut = 0

            if usable_spanners_carried:
                 # Man has a usable spanner, just needs to get to the nut
                 dist_man_to_nut = self.get_distance(man_location, nut_location)
                 if dist_man_to_nut == float('inf'):
                      cost_for_this_nut = large_penalty # Nut location unreachable
                 else:
                      cost_for_this_nut = dist_man_to_nut
            else:
                 # Man needs a spanner. Find the best path via a spanner.
                 min_path_via_spanner = float('inf')
                 if usable_spanners_at_loc:
                      for spanner, spanner_loc in usable_spanners_at_loc.items():
                           dist_man_to_spanner = self.get_distance(man_location, spanner_loc)
                           dist_spanner_to_nut = self.get_distance(spanner_loc, nut_location)
                           if dist_man_to_spanner != float('inf') and dist_spanner_to_nut != float('inf'):
                                # Cost is travel to spanner + pickup + travel to nut
                                min_path_via_spanner = min(min_path_via_spanner, dist_man_to_spanner + 1 + dist_spanner_to_nut)

                 if min_path_via_spanner != float('inf'):
                      cost_for_this_nut = min_path_via_spanner
                 else:
                      # No usable spanners available anywhere (carried or at location)
                      cost_for_this_nut = large_penalty # Cannot tighten this nut

            h_value += cost_for_this_nut

        # --- Step 8: Return total heuristic value ---
        return h_value
