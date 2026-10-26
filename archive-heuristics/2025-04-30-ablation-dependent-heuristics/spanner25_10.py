import collections
import sys

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the cost to reach the goal state (all goal nuts
    tightened) by summing three components:
    1. The number of loose goal nuts (minimum tighten actions).
    2. The number of spanner pickup actions required (based on the number
       of loose nuts and spanners currently carried).
    3. The estimated travel cost (minimum walk actions) to reach a relevant
       location (either a loose nut location or a usable spanner location
       if pickups are needed).

    Assumptions:
    - The input state is a frozenset of PDDL fact strings.
    - The Task object provides access to initial state, goals, static facts,
      and domain facts (though domain facts are used indirectly for type inference).
    - The location graph derived from static 'link' facts is connected for
      all relevant locations in solvable problems.
    - There are enough usable spanners available in total (carried or at
      locations) to tighten all goal nuts in solvable problems.

    Heuristic Initialization:
    The constructor precomputes static information:
    - Identifies the man, spanners, nuts, and locations by inspecting initial
      state, static, and goal facts.
    - Identifies the goal nuts from the task's goal state.
    - Builds a graph representing the locations and links based on static facts.
    - Computes all-pairs shortest paths (distances) between all pairs of
      locations using BFS. This distance information is stored in
      `self.distance[l1][l2]`.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify all goal nuts that are currently loose. Let this count be N_loose.
    2. If N_loose is 0, the state is a goal state, return 0.
    3. Find the current location of the man.
    4. Count the number of usable spanners the man is currently carrying. Let this be N_carried.
    5. Calculate the number of additional spanners the man needs to pick up: N_pickups = max(0, N_loose - N_carried).
    6. Find the current locations of all loose goal nuts.
    7. Find the current locations of all usable spanners that are currently at locations (not carried).
    8. Determine the set of 'required locations' the man needs to visit soon. This set includes all loose nut locations. If N_pickups > 0, it also includes all usable spanner locations (at locations).
    9. Calculate the estimated travel cost as the minimum distance from the man's current location to any location in the set of required locations. Use the precomputed distances. If the set of required locations is empty (should not happen if N_loose > 0), the travel cost is 0. If a required location is unreachable from the man's current location, the minimum distance will be infinity, and the heuristic will return infinity.
    10. The heuristic value is the sum of N_loose (for tighten actions), N_pickups (for pickup actions), and the estimated travel cost.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information like
        object types, goal nuts, and location distances.
        """
        self.task = task
        self.goal_nuts = self._get_goal_nuts(task.goals)
        # Infer objects by type from initial state, static, and goal facts
        self.man = self._get_object_by_type(task.initial_state, task.static, task.goals, 'man')
        self.spanners = self._get_objects_by_type(task.initial_state, task.static, task.goals, 'spanner')
        self.nuts = self._get_objects_by_type(task.initial_state, task.static, task.goals, 'nut')
        self.locations = self._get_objects_by_type(task.initial_state, task.static, task.goals, 'location')

        # Build location graph and compute distances
        self.distance = self._compute_distances(task.static, self.locations)


    def _get_object_name_from_fact(self, fact_string, position):
        """Extracts object name from a fact string at a given argument position."""
        # Example: '(at bob shed)' -> parts = ['at', 'bob', 'shed']
        # position 1 gives 'bob', position 2 gives 'shed'
        parts = fact_string.strip('()').split()
        if position < len(parts):
            return parts[position]
        return None

    def _get_predicate_name_from_fact(self, fact_string):
         """Extracts predicate name from a fact string."""
         parts = fact_string.strip('()').split()
         if parts:
             return parts[0]
         return None


    def _get_objects_by_type(self, initial_state_facts, static_facts, goal_facts, obj_type):
        """
        Infers objects of a specific type by looking at predicate signatures
        in initial state, static, and goal facts.
        This is a simplified approach based on common PDDL patterns and the
        structure of the spanner domain.
        """
        objects = set()
        relevant_facts = initial_state_facts | static_facts | goal_facts

        if obj_type == 'man':
            # Man appears in (at ?m ?l), (carrying ?m ?s), (walk ?s ?e ?m), (pickup_spanner ?l ?s ?m), (tighten_nut ?l ?s ?m ?n)
            for fact in relevant_facts:
                 pred = self._get_predicate_name_from_fact(fact)
                 if pred in ('at', 'carrying', 'walk', 'pickup_spanner', 'tighten_nut'):
                     parts = fact.strip('()').split()
                     if pred == 'at' and len(parts) == 3: objects.add(parts[1]) # (at ?locatable ?location) - first arg is locatable
                     if pred == 'carrying' and len(parts) == 3: objects.add(parts[1]) # (carrying ?man ?spanner) - first arg is man
                     if pred == 'walk' and len(parts) == 4: objects.add(parts[3]) # (walk ?start ?end ?man) - third arg is man
                     if pred == 'pickup_spanner' and len(parts) == 4: objects.add(parts[3]) # (pickup_spanner ?l ?s ?m) - third arg is man
                     if pred == 'tighten_nut' and len(parts) == 5: objects.add(parts[3]) # (tighten_nut ?l ?s ?m ?n) - third arg is man
            return list(objects) # Expecting one man

        elif obj_type == 'spanner':
             # Spanner appears in (at ?s ?l), (carrying ?m ?s), (usable ?s), (pickup_spanner ?l ?s ?m), (tighten_nut ?l ?s ?m ?n)
             for fact in relevant_facts:
                 pred = self._get_predicate_name_from_fact(fact)
                 if pred in ('at', 'carrying', 'usable', 'pickup_spanner', 'tighten_nut'):
                     parts = fact.strip('()').split()
                     if pred == 'at' and len(parts) == 3: objects.add(parts[1]) # (at ?locatable ?location) - could be spanner
                     if pred == 'carrying' and len(parts) == 3: objects.add(parts[2]) # (carrying ?man ?spanner) - second arg is spanner
                     if pred == 'usable' and len(parts) == 2: objects.add(parts[1]) # (usable ?spanner) - first arg is spanner
                     if pred == 'pickup_spanner' and len(parts) == 4: objects.add(parts[2]) # (pickup_spanner ?l ?s ?m) - second arg is spanner
                     if pred == 'tighten_nut' and len(parts) == 5: objects.add(parts[2]) # (tighten_nut ?l ?s ?m ?n) - second arg is spanner
             return list(objects)

        elif obj_type == 'nut':
             # Nut appears in (at ?n ?l), (tightened ?n), (loose ?n), (tighten_nut ?l ?s ?m ?n)
             for fact in relevant_facts:
                 pred = self._get_predicate_name_from_fact(fact)
                 if pred in ('at', 'tightened', 'loose', 'tighten_nut'):
                     parts = fact.strip('()').split()
                     if pred == 'at' and len(parts) == 3: objects.add(parts[1]) # (at ?locatable ?location) - could be nut
                     if pred == 'tightened' and len(parts) == 2: objects.add(parts[1]) # (tightened ?nut) - first arg is nut
                     if pred == 'loose' and len(parts) == 2: objects.add(parts[1]) # (loose ?nut) - first arg is nut
                     if pred == 'tighten_nut' and len(parts) == 5: objects.add(parts[4]) # (tighten_nut ?l ?s ?m ?n) - fourth arg is nut
             return list(objects)

        elif obj_type == 'location':
             # Location appears in (at ?obj ?l), (link ?l1 ?l2), (walk ?s ?e ?m), (pickup_spanner ?l ?s ?m), (tighten_nut ?l ?s ?m ?n)
             for fact in relevant_facts:
                 pred = self._get_predicate_name_from_fact(fact)
                 if pred in ('at', 'link', 'walk', 'pickup_spanner', 'tighten_nut'):
                     parts = fact.strip('()').split()
                     if pred == 'at' and len(parts) == 3: objects.add(parts[2]) # (at ?locatable ?location) - second arg is location
                     if pred == 'link' and len(parts) == 3: # (link ?l1 ?l2)
                         objects.add(parts[1])
                         objects.add(parts[2])
                     if pred == 'walk' and len(parts) == 4: # (walk ?start ?end ?man)
                         objects.add(parts[1]) # start
                         objects.add(parts[2]) # end
                     if pred == 'pickup_spanner' and len(parts) == 4: objects.add(parts[1]) # (pickup_spanner ?l ?s ?m) - first arg is location
                     if pred == 'tighten_nut' and len(parts) == 5: objects.add(parts[1]) # (tighten_nut ?l ?s ?m ?n) - first arg is location
             return list(objects)

        return [] # Unknown type


    def _get_object_by_type(self, initial_state_facts, static_facts, goal_facts, obj_type):
        """Returns the first object found of a specific type (useful for singletons like man)."""
        objects = self._get_objects_by_type(initial_state_facts, static_facts, goal_facts, obj_type)
        return objects[0] if objects else None


    def _get_goal_nuts(self, goals):
        """Extracts the names of the nuts that need to be tightened from the goal facts."""
        goal_nuts = set()
        # Goals are a frozenset of facts, e.g., frozenset({'(tightened nut1)', '(tightened nut2)'})
        for goal_fact in goals:
            if goal_fact.startswith('(tightened '):
                # Extract the nut name from '(tightened nut_name)'
                parts = goal_fact.strip('()').split()
                if len(parts) == 2:
                    nut_name = parts[1]
                    goal_nuts.add(nut_name)
        return goal_nuts

    def _compute_distances(self, static_facts, locations):
        """
        Builds a graph from link facts and computes shortest path distances
        between all pairs of locations using BFS.
        """
        graph = collections.defaultdict(set)
        for fact in static_facts:
            if fact.startswith('(link '):
                parts = fact.strip('()').split()
                if len(parts) == 3:
                    loc1 = parts[1]
                    loc2 = parts[2]
                    graph[loc1].add(loc2)
                    graph[loc2].add(loc1) # Links are bidirectional for walk

        distance = {}
        # Initialize distance map for all known locations
        for loc in locations:
             distance[loc] = {}

        # Run BFS from each location to find distances to all reachable locations
        for start_loc in locations:
            queue = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            distance[start_loc][start_loc] = 0 # Distance to self is 0

            while queue:
                current_loc, dist = queue.popleft()

                # Add neighbors to queue if not visited
                for neighbor in graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distance[start_loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

        # Any location not reached from start_loc will not have an entry in distance[start_loc].
        # Accessing distance[l1][l2] will raise KeyError if unreachable.
        # The heuristic calculation handles this by checking existence.
        return distance

    def get_object_location(self, state, obj_name):
        """Finds the current location of a locatable object in the state."""
        # Man's location is handled by get_man_location
        if obj_name == self.man:
            return self.get_man_location(state)

        # Check if the object is carried by the man
        if '(carrying ' + self.man + ' ' + obj_name + ')' in state:
            # If carried, its location is the man's location
            return self.get_man_location(state)

        # Check if the object is at a location
        for fact in state:
            if fact.startswith('(at '):
                parts = fact.strip('()').split()
                if len(parts) == 3 and parts[1] == obj_name:
                    return parts[2]

        # If not carried and not found at a location, something is wrong
        # with the state representation or the object doesn't exist/is not locatable.
        # In this domain, nuts and spanners are locatable and persist.
        # print(f"Warning: Location not found for object {obj_name} in state.")
        return None # Should not happen for locatable objects in a valid state


    def get_man_location(self, state):
        """Finds the current location of the man in the state."""
        for fact in state:
            if fact.startswith('(at '):
                parts = fact.strip('()').split()
                if len(parts) == 3 and parts[1] == self.man:
                    return parts[2]
        # Man must always be at a location in a valid state
        # print(f"Error: Man's location not found in state.")
        return None # Should not happen


    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        # 1. Identify loose goal nuts
        loose_goal_nuts = {n for n in self.goal_nuts if '(loose ' + n + ')' in state}

        # 2. If N_loose is 0, the state is a goal state, return 0.
        n_loose = len(loose_goal_nuts)
        if n_loose == 0:
            return 0

        # 3. Find the current location of the man.
        man_location = self.get_man_location(state)
        if man_location is None:
             # Should not happen in a valid state, but indicates a problem
             return float('inf') # Problem state

        # 4. Count usable spanners carried by the man.
        carried_usable_spanners = {s for s in self.spanners if '(carrying ' + self.man + ' ' + s + ')' in state and '(usable ' + s + ')' in state}
        n_carried = len(carried_usable_spanners)

        # 5. Calculate the number of additional spanners needed.
        n_pickups_needed = max(0, n_loose - n_carried)

        # 6. Find locations of loose goal nuts.
        loose_nut_locations = {self.get_object_location(state, n) for n in loose_goal_nuts}
        # Filter out None locations if any nut location wasn't found (shouldn't happen)
        loose_nut_locations = {loc for loc in loose_nut_locations if loc is not None}


        # 7. Find locations of usable spanners at locations.
        # Need to get usable spanners that are *at* a location, not carried.
        usable_spanners_at_loc_objs = {s for s in self.spanners if '(at ' + s + ' ' + self.get_object_location(state, s) + ')' in state and '(usable ' + s + ')' in state}
        usable_spanner_locations = {self.get_object_location(state, s) for s in usable_spanners_at_loc_objs}
        # Filter out None locations
        usable_spanner_locations = {loc for loc in usable_spanner_locations if loc is not None}


        # 8. Determine required locations for travel.
        required_locations = set(loose_nut_locations)
        if n_pickups_needed > 0:
            required_locations.update(usable_spanner_locations)

        # 9. Calculate estimated travel cost.
        travel_cost = 0
        if required_locations:
            # Find the minimum distance from the man's current location to any required location.
            min_dist = float('inf')
            # Check if man_location is a valid key in the distance map
            if man_location in self.distance:
                for loc in required_locations:
                    # Check if the target location is reachable from man_location
                    if loc in self.distance[man_location]:
                         min_dist = min(min_dist, self.distance[man_location][loc])

            if min_dist == float('inf'):
                 # This implies a required location is unreachable from the man's current location.
                 # Return infinity to prune this path.
                 return float('inf')
            travel_cost = min_dist
        # If required_locations is empty, travel_cost remains 0 (handled by n_loose == 0 case)


        # 10. Compute the total heuristic value.
        # Heuristic = N_loose (tighten actions) + N_pickups_needed (pickup actions) + travel_cost (walk actions)
        heuristic_value = n_loose + n_pickups_needed + travel_cost

        return heuristic_value
