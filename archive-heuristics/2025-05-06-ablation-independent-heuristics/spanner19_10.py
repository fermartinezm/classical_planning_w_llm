from heuristics.heuristic_base import Heuristic
from task import Task
from collections import deque

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
    Estimates the cost to reach the goal (tighten all required nuts) by
    greedily assigning usable spanners to loose nuts. It calculates the minimum
    cost to pick up the best available spanner and take it to the location of
    the best available loose nut, tighten it, and repeats this process from the
    new location for the remaining nuts. The total heuristic is the sum of
    these minimum costs. Shortest path distances between locations are precomputed
    using BFS.

    Assumptions:
    - The domain follows the PDDL definition provided.
    - Links between locations are bidirectional.
    - Each 'tighten_nut' action consumes one usable spanner.
    - The goal is a conjunction of (tightened nut_name) facts.
    - The state representation is a frozenset of strings as shown in the example.
    - Static facts are provided in task.static.
    - Object types are available in task.objects.

    Heuristic Initialization:
    1. Identify all locations and links from static facts and object types.
    2. Build an adjacency list representation of the location graph.
    3. Compute all-pairs shortest paths between locations using BFS.
    4. Identify the man object name and the set of goal nuts from the task definition.
    5. Store object types for parsing state facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state (frozenset of facts).
    2. Parse the state to extract:
       - The man's current location.
       - Locations of all nuts and spanners.
       - Which items are carried by the man.
       - Which items are usable.
       - Which nuts are loose.
    3. Identify the set of goal nuts that are currently loose and their locations.
    4. Identify the set of usable spanners and their initial status (carried or at location) in this state.
    5. If there are more loose goal nuts than available usable spanners, return infinity (dead end).
    6. If there are no loose goal nuts, return 0 (goal reached).
    7. Initialize the total heuristic value to 0.
    8. Initialize the current man location to the man's location extracted from the state.
    9. Create working sets of remaining loose goal nuts and remaining usable spanners.
    10. While there are remaining loose goal nuts:
        a. Find the minimum cost to tighten one of the remaining loose nuts using one of the remaining usable spanners, starting from the current man location.
           - Iterate through each remaining loose nut N at location LN.
           - Iterate through each remaining usable spanner S.
           - Determine the cost to use S for N:
             - If S was initially carried by the man in this state: Cost = shortest_path(current_man_loc, LN) + 1 (tighten).
             - If S was initially at location LS in this state: Cost = shortest_path(current_man_loc, LS) + 1 (pickup) + shortest_path(LS, LN) + 1 (tighten).
           - Keep track of the minimum cost found and the corresponding nut (N_best) and spanner (S_best).
        b. If no valid pair (nut, spanner) could be assigned (e.g., no paths exist), return infinity (dead end).
        c. Add the minimum cost (min_cost_for_next_nut) to the total heuristic value.
        d. Update the current man location to the location of N_best (where the nut was tightened).
        e. Remove N_best from the set of remaining loose nuts.
        f. Remove S_best from the set of remaining usable spanners.
    11. Return the total heuristic value.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static
        self.object_types = task.objects # Store object types

        # 1. Identify locations and links, build graph
        self.location_graph = {}
        self.locations = set()
        self.man_name = None

        # Find the man object name
        for obj_name, obj_type in self.object_types.items():
            if obj_type == 'man':
                self.man_name = obj_name
                break

        # Identify locations from object types
        for obj_name, obj_type in self.object_types.items():
            if obj_type == 'location':
                self.locations.add(obj_name)
                self.location_graph.setdefault(obj_name, []) # Ensure all locations are keys

        # Add links to the graph
        for fact_str in self.static:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'link' and len(args) == 2:
                loc1, loc2 = args
                # Ensure they are actually locations (should be based on object_types)
                if loc1 in self.locations and loc2 in self.locations:
                    self.location_graph[loc1].append(loc2)
                    self.location_graph[loc2].append(loc1) # Links are bidirectional
                # else: Warning about link between non-locations?

        # 2. Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_node in self.locations:
            self.shortest_paths[start_node] = self._bfs(start_node)

        # 3. Identify goal nuts
        self.goal_nuts = set()
        for goal_fact in self.goals:
            predicate, args = self._parse_fact(goal_fact)
            if predicate == 'tightened' and len(args) == 1:
                nut_name = args[0]
                # Ensure it's actually a nut
                if self.object_types.get(nut_name) == 'nut':
                    self.goal_nuts.add(nut_name)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all other nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node in distances: # Handle case where start_node might not be in self.locations (e.g., malformed problem)
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                if current_node in self.location_graph: # Handle locations with no links
                    for neighbor in self.location_graph[current_node]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_node] + 1
                            queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Safely get shortest path distance."""
        if loc1 not in self.shortest_paths or loc2 not in self.shortest_paths.get(loc1, {}):
             # Should not happen if all relevant locations are included
             # in self.locations during init, but as a safeguard:
             # print(f"Warning: Distance requested between unknown locations {loc1} and {loc2}")
             return float('inf')
        return self.shortest_paths[loc1][loc2]

    def _parse_fact(self, fact_str):
        """Parses a fact string into predicate and arguments."""
        # Example: '(at bob shed)' -> ['at', 'bob', 'shed']
        # Example: '(tightened nut1)' -> ['tightened', 'nut1']
        parts = fact_str.strip('()').split()
        if not parts:
            return None, []
        return parts[0], parts[1:]


    def __call__(self, node):
        state = node.state

        man_location = None
        nut_locations = {} # {nut_name: location} for all nuts in state
        spanner_locations = {} # {spanner_name: location} for all spanners in state
        carried_items = {} # {item_name: carrier_name}
        usable_items = set() # {item_name}
        loose_nuts_in_state = set() # {nut_name}

        # First pass: Extract all relevant information from the state
        for fact_str in state:
            predicate, args = self._parse_fact(fact_str)

            if predicate == 'at' and len(args) == 2:
                obj_name, loc_name = args
                obj_type = self.object_types.get(obj_name)
                if obj_type == 'man' and obj_name == self.man_name:
                    man_location = loc_name
                elif obj_type == 'spanner':
                    spanner_locations[obj_name] = loc_name
                elif obj_type == 'nut':
                    nut_locations[obj_name] = loc_name # Store location for all nuts

            elif predicate == 'carrying' and len(args) == 2:
                 carrier_name, item_name = args
                 carried_items[item_name] = carrier_name

            elif predicate == 'usable' and len(args) == 1:
                 item_name = args[0]
                 usable_items.add(item_name)

            elif predicate == 'loose' and len(args) == 1:
                 nut_name = args[0]
                 loose_nuts_in_state.add(nut_name)

        # Identify loose goal nuts and their locations
        loose_goal_nuts = {} # {nut_name: location}
        for nut_name in self.goal_nuts:
             if nut_name in loose_nuts_in_state:
                  if nut_name in nut_locations:
                       loose_goal_nuts[nut_name] = nut_locations[nut_name]
                  else:
                       # Loose goal nut exists but has no location? Dead end.
                       # This shouldn't happen in valid PDDL states, but handle defensively.
                       return float('inf')

        # Identify usable spanners and their status (carried or at location)
        available_usable_spanners = {} # {spanner_name: {'loc': location} or {'carried': True}}
        carried_usable_spanners = set() # {spanner_name}

        for spanner_name in usable_items:
             obj_type = self.object_types.get(spanner_name)
             if obj_type == 'spanner': # Ensure it's actually a spanner
                  if spanner_name in carried_items and carried_items[spanner_name] == self.man_name:
                       available_usable_spanners[spanner_name] = {'carried': True}
                       carried_usable_spanners.add(spanner_name)
                  elif spanner_name in spanner_locations:
                       available_usable_spanners[spanner_name] = {'loc': spanner_locations[spanner_name]}
                  # else: usable spanner exists but is neither carried nor at a location? Ignore.

        # Check if goal is reached
        if not loose_goal_nuts:
            return 0

        # Check for dead end: not enough usable spanners for loose goal nuts
        if len(loose_goal_nuts) > len(available_usable_spanners):
             return float('inf')

        # Check for dead end: man's location is unknown
        if man_location is None:
             return float('inf')

        # 3. Implement greedy matching
        h_value = 0
        current_man_loc = man_location
        remaining_loose_nuts = set(loose_goal_nuts.keys())
        remaining_usable_spanners = set(available_usable_spanners.keys())

        # Need a copy of available_usable_spanners info as we iterate
        spanner_status_map = dict(available_usable_spanners) # shallow copy is fine

        while remaining_loose_nuts:
            min_cost_for_next_nut = float('inf')
            best_nut = None
            best_spanner = None

            # Iterate through all remaining loose nuts
            for nut_name in remaining_loose_nuts:
                nut_loc = loose_goal_nuts[nut_name]

                # Iterate through all remaining usable spanners
                for spanner_name in remaining_usable_spanners:
                    spanner_info = spanner_status_map[spanner_name] # Get original status for this state

                    cost = float('inf')
                    if spanner_name in carried_usable_spanners: # Check if it was initially carried in this state
                        # Spanner is carried by the man (at current_man_loc)
                        # Cost = walk from current_man_loc to nut_loc + tighten
                        dist_to_nut = self.get_distance(current_man_loc, nut_loc)
                        if dist_to_nut != float('inf'):
                             cost = dist_to_nut + 1 # 1 for tighten action
                    elif 'loc' in spanner_info:
                        # Spanner is at a location
                        spanner_loc = spanner_info['loc']
                        # Cost = walk from current_man_loc to spanner_loc + pickup + walk from spanner_loc to nut_loc + tighten
                        dist_to_spanner = self.get_distance(current_man_loc, spanner_loc)
                        dist_spanner_to_nut = self.get_distance(spanner_loc, nut_loc)

                        if dist_to_spanner != float('inf') and dist_spanner_to_nut != float('inf'):
                             cost = dist_to_spanner + 1 + dist_spanner_to_nut + 1 # 1 for pickup, 1 for tighten

                    # Update minimum cost for this iteration
                    if cost < min_cost_for_next_nut:
                        min_cost_for_next_nut = cost
                        best_nut = nut_name
                        best_spanner = spanner_name

            # If we couldn't find a way to tighten any remaining nut with any remaining spanner
            if min_cost_for_next_nut == float('inf'):
                 return float('inf') # Dead end

            # Add the cost of tightening the best pair
            h_value += min_cost_for_next_nut

            # Update state for the next iteration of the greedy choice
            current_man_loc = loose_goal_nuts[best_nut] # Man is now at the nut location
            remaining_loose_nuts.remove(best_nut)
            remaining_usable_spanners.remove(best_spanner)
            # Note: carried_usable_spanners and spanner_status_map are based on the *initial* state
            # for this heuristic call and are not updated within the greedy loop.
            # The removal from remaining_usable_spanners handles consumption.

        return h_value
