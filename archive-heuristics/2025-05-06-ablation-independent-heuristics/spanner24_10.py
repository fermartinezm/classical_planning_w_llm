import heapq
from collections import deque, defaultdict
import logging

from heuristics.heuristic_base import Heuristic
# Assuming Task class is available from task.py
# from task import Operator, Task


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum estimated
        cost for each loose target nut. The cost for a single nut is estimated
        as 1 (for the tighten action) plus the minimum cost to get the man
        to the nut's location while carrying a usable spanner. This minimum
        cost considers using an already carried spanner (just walking) or
        picking up the nearest usable spanner on the ground (walking to spanner,
        picking up, walking to nut).

    Assumptions:
        - The PDDL domain is 'spanner' as provided.
        - The goal is a conjunction of (tightened nut) facts.
        - Nut locations are static (do not change during planning).
        - Spanner locations are static unless carried by the man.
        - The man can carry multiple spanners simultaneously.
        - The 'link' predicates define a connected graph of locations.
        - Action costs are 1.

    Heuristic Initialization:
        - Parses the task facts, initial state, and static facts to identify
          locations and build the graph of linked locations.
        - Computes all-pairs shortest paths between locations using BFS.
        - Identifies names of man, spanners, and nuts based on predicate usage.
        - Stores static nut locations from the initial state.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state and identify the man's location, carried
           spanners, usable spanners, and ground spanners with their locations.
        2. Identify the set of target nuts that are currently loose. If this set
           is empty, the goal is reached, return 0.
        3. Count the total number of usable spanners available (carried or on ground).
        4. If the number of loose target nuts exceeds the total number of usable
           spanners, the goal is unreachable, return infinity.
        5. Initialize the total heuristic value to 0.
        6. For each loose target nut:
            a. Get its static location.
            b. Calculate the minimum cost to get the man to this nut's location
               while carrying *a* usable spanner. This cost is the minimum of:
               - The distance from the man's current location to the nut's
                 location, if the man is currently carrying *any* usable spanner.
                 (This assumes one of the carried spanners can be used, regardless
                 of how many are carried or needed for other nuts - a relaxation).
               - The minimum cost over all usable spanners on the ground:
                 distance from man's current location to the spanner's location
                 + 1 (for pickup action) + distance from the spanner's location
                 to the nut's location.
               - If no usable spanners are available (neither carried nor on ground),
                 this cost is infinity.
            c. Add 1 (for the tighten action) plus the calculated minimum spanner
               delivery/availability cost to the total heuristic value.
        7. Return the total heuristic value.

    Note: This heuristic sums the costs for each nut independently. This
    overestimates the true cost by potentially double-counting shared travel
    paths (e.g., walking to a location where multiple nuts are) or the benefit
    of picking up multiple spanners on a single trip. However, it provides
    a lower bound on the actions required for each individual nut and captures
    the core dependencies (man at location, carrying spanner, spanner usable,
    nut loose), making it a useful non-admissible heuristic for greedy search.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Data structures for location graph and distances
        self.locations = set()
        self.location_graph = defaultdict(set)
        self.distances = {} # Dict of dicts: distances[loc1][loc2] = dist

        # Object names by type (inferred)
        self.man_names = set()
        self.spanner_names = set()
        self.nut_names = set()
        self.nut_locations = {} # Static nut locations

        self._parse_facts_for_structure()
        self._compute_all_pairs_shortest_paths()

    def _parse_facts_for_structure(self):
        """Parses facts to build graph and get object names/locations."""
        # Use all possible facts to identify objects and predicates
        all_facts_strings = self.task.facts

        # Identify predicates and their argument types based on domain definition
        # Hardcoding based on the provided domain file
        predicates_info = {
            'at': ['locatable', 'location'],
            'carrying': ['man', 'spanner'],
            'usable': ['spanner'],
            'link': ['location', 'location'],
            'tightened': ['nut'],
            'loose': ['nut'],
        }

        # Collect all object names seen in any fact
        all_object_names = set()
        for fact_str in all_facts_strings:
             parts = fact_str.strip('()').split()
             # Skip predicate name
             all_object_names.update(parts[1:])

        # Try to classify objects based on predicates they appear in
        obj_potential_types = defaultdict(set)
        for fact_str in all_facts_strings:
             parts = fact_str.strip('()').split()
             predicate = parts[0]
             if predicate in predicates_info:
                 arg_types = predicates_info[predicate]
                 for i, obj_name in enumerate(parts[1:]):
                     if i < len(arg_types):
                         obj_potential_types[obj_name].add(arg_types[i])

        # Classify objects - simple approach: if an object appears as a 'man', it's a man.
        # If it appears as a 'spanner', it's a spanner, etc.
        # Assuming no object belongs to multiple conflicting types.
        for obj_name, types in obj_potential_types.items():
            if 'man' in types:
                self.man_names.add(obj_name)
            if 'spanner' in types:
                self.spanner_names.add(obj_name)
            if 'nut' in types:
                self.nut_names.add(obj_name)
            if 'location' in types:
                self.locations.add(obj_name)
            # 'locatable' is a supertype, ignore for specific classification here

        # Build location graph from static facts
        for fact in self.static_facts:
            if fact.startswith('(link '):
                parts = fact.strip('()').split()
                loc1 = parts[1]
                loc2 = parts[2]
                # Ensure they are actually identified as locations, though link implies it
                if loc1 in self.locations and loc2 in self.locations:
                    self.location_graph[loc1].add(loc2)
                    self.location_graph[loc2].add(loc1) # Links are bidirectional

        # Get static nut locations from initial state
        for fact in self.initial_state:
             if fact.startswith('(at '):
                 parts = fact.strip('()').split()
                 obj = parts[1]
                 loc = parts[2]
                 if obj in self.nut_names:
                     self.nut_locations[obj] = loc
                 # Man's initial location will be found in __call__ from the state.
                 # Man's name should be identified from carrying facts or by exclusion if only one locatable left.
                 # If man_names is still empty, try to find the single locatable in initial state that isn't a spanner or nut
                 if not self.man_names and obj not in self.spanner_names and obj not in self.nut_names and loc in self.locations:
                      self.man_names.add(obj) # Assume the remaining locatable is the man


        # Ensure all locations from the graph are in self.locations
        # (In case a location is linked but doesn't appear in any other fact type)
        for loc in self.location_graph:
             self.locations.add(loc)
        for neighbors in self.location_graph.values():
             self.locations.update(neighbors)

        # Handle potential locations mentioned only in initial state 'at' facts
        # e.g., (at spanner1 location1) where location1 is not in any link fact
        for fact in self.initial_state:
             if fact.startswith('(at '):
                 parts = fact.strip('()').split()
                 loc = parts[2]
                 self.locations.add(loc)

        # Rebuild graph ensuring all identified locations are keys/values
        temp_graph = defaultdict(set)
        for loc in self.locations:
             temp_graph[loc] = self.location_graph.get(loc, set())
        self.location_graph = temp_graph


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        for start_node in self.locations:
            self.distances[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}

            while q:
                current_node, dist = q.popleft()
                self.distances[start_node][current_node] = dist

                for neighbor in self.location_graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # If a location is not in our computed distances (e.g., isolated node, or error)
        # treat as unreachable.
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # logging.warning(f"Distance requested for unconnected or unknown locations: {loc1} to {loc2}")
             return float('inf')
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Computes the spanner heuristic for the given state node."""
        state = node.state

        # 1. Identify current state facts
        man_loc = None
        carried_spanners = set()
        usable_spanners = set()
        ground_spanners_locations = {} # map spanner name to location

        for fact in state:
            if fact.startswith('(at '):
                parts = fact.strip('()').split()
                obj = parts[1]
                loc = parts[2]
                if obj in self.man_names:
                     man_loc = loc
                elif obj in self.spanner_names:
                     ground_spanners_locations[obj] = loc
                # Nut locations are static, retrieved in __init__

            elif fact.startswith('(carrying '):
                parts = fact.strip('()').split()
                # man = parts[1] # Assuming only one man, already identified
                spanner = parts[2]
                carried_spanners.add(spanner)

            elif fact.startswith('(usable '):
                parts = fact.strip('()').split()
                spanner = parts[1]
                usable_spanners.add(spanner)

        # Ensure man_loc was found (should always be the case in valid states)
        if man_loc is None:
             # This state is likely invalid or represents an unreachable scenario
             # where the man's location is not asserted.
             # For heuristic purposes, treat as infinite cost.
             # logging.error("Man's location not found in state.")
             return float('inf')


        # 2. Identify loose target nuts
        current_loose_target_nuts = set()
        for goal_fact in self.goals:
            if goal_fact.startswith('(tightened '):
                nut = goal_fact.strip('()').split()[1]
                # Check if the corresponding loose fact exists in the current state
                if f'(loose {nut})' in state:
                     current_loose_target_nuts.add(nut)

        # 3. Check if goal is reached
        if not current_loose_target_nuts:
            return 0

        # 4. Get usable spanners currently carried or on ground
        usable_carried = carried_spanners.intersection(usable_spanners)
        usable_ground_locations = {s: loc for s, loc in ground_spanners_locations.items() if s in usable_spanners}

        num_loose_target_nuts = len(current_loose_target_nuts)
        num_usable_spanners = len(usable_carried) + len(usable_ground_locations)

        # 5. Check reachability based on spanners
        if num_usable_spanners < num_loose_target_nuts:
            return float('inf')

        # 6. Calculate heuristic by summing costs for each loose nut
        total_heuristic = 0
        for nut in current_loose_target_nuts:
            nut_loc = self.nut_locations.get(nut) # Get static location of the nut

            # Should always find nut location if it's a target nut from initial state
            if nut_loc is None:
                 # logging.error(f"Location not found for target nut: {nut}")
                 return float('inf') # Should not happen in valid problems

            cost_to_deliver_spanner_to_nut_loc = float('inf')

            # Option 1: Use a carried usable spanner
            # This option is available if there is *at least one* usable spanner carried.
            # The cost is just the walk from man_loc to nut_loc.
            # We don't decrement the count of carried spanners here because this is a relaxation.
            if usable_carried:
                cost_to_deliver_spanner_to_nut_loc = min(cost_to_deliver_spanner_to_nut_loc, self.get_distance(man_loc, nut_loc))

            # Option 2: Pick up a usable ground spanner
            # This option is available if there is *at least one* usable spanner on the ground.
            # We find the minimum cost over all usable ground spanners.
            if usable_ground_locations:
                min_ground_spanner_delivery_cost = float('inf')
                for s, s_loc in usable_ground_locations.items():
                    # Cost = walk man_loc to s_loc + pickup (cost 1) + walk s_loc to nut_loc
                    walk_cost = self.get_distance(man_loc, s_loc) + self.get_distance(s_loc, nut_loc)
                    cost = walk_cost + 1 # Add 1 for the pickup action
                    min_ground_spanner_delivery_cost = min(min_ground_spanner_delivery_cost, cost)
                cost_to_deliver_spanner_to_nut_loc = min(cost_to_deliver_spanner_to_nut_loc, min_ground_spanner_delivery_cost)

            # If after checking both options, the cost is still infinity,
            # it means no usable spanners are available for this nut (which
            # should have been caught by the earlier check, but good safeguard).
            if cost_to_deliver_spanner_to_nut_loc == float('inf'):
                 return float('inf')

            # Total cost for this nut = tighten action (cost 1) + minimum delivery cost
            total_heuristic += 1 + cost_to_deliver_spanner_to_nut_loc

        # 7. Return the total heuristic value
        return total_heuristic
