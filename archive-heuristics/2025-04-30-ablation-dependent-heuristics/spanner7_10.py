import sys
from collections import deque

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the number of actions required to reach the goal state
    (all specified nuts tightened) from a given state. The heuristic is
    designed for greedy best-first search and does not need to be admissible.
    It is calculated as the sum of the minimum number of tighten actions,
    minimum number of pickup actions, and an estimate of the minimum walk
    actions required to get to a relevant location.

    Assumptions:
    - There is exactly one man in the domain. The heuristic attempts to identify
      the man by looking for the object involved in a 'carrying' predicate in
      the initial state, or as a fallback, the first object in an 'at' predicate
      that is not identified as a spanner or nut.
    - Object types (man, spanner, nut, location) are inferred by their usage
      in initial state and static facts based on predicate types ('carrying',
      'usable', 'loose', 'tightened', 'at', 'link'). This inference relies on
      typical PDDL conventions for this domain.
    - The location graph defined by 'link' facts is connected for all
      locations relevant to solvable problems. Unreachable required locations
      result in an infinite heuristic value.

    Heuristic Initialization:
    1. Parses the initial state and static facts to identify the names of the
       man, spanners, nuts, and locations based on the predicates they appear in.
    2. Builds a graph representing the locations and the links between them
       using the static 'link' facts.
    3. Computes all-pairs shortest path distances between all identified locations
       using Breadth-First Search (BFS). These distances represent the minimum
       number of 'walk' actions required to travel between locations.
    4. Stores the set of nuts that are part of the goal condition.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to identify:
       - The man's current location.
       - The set of usable spanners the man is currently carrying.
       - A mapping from spanner names to their locations for usable spanners
         that are currently at a location (not carried).
       - A mapping from nut names to their locations for nuts that are currently
         loose and are part of the goal condition.
    2. Determine the number of loose nuts that are part of the goal (`k`).
       If `k` is 0, the goal is reached, and the heuristic value is 0.
    3. Determine the number of usable spanners the man is carrying (`c`).
    4. Calculate the number of additional usable spanners the man needs to acquire
       from locations (`m = max(0, k - c)`). This is the minimum number of
       'pickup_spanner' actions required.
    5. Check if there are enough usable spanners available in the entire problem
       (carried + at locations) to tighten all `k` nuts. If `m` is greater than
       the total number of available usable spanners at locations, the state is
       likely unsolvable, and the heuristic returns infinity.
    6. Calculate the base cost, which is the sum of the minimum required 'tighten_nut'
       actions (`k`) and the minimum required 'pickup_spanner' actions (`m`).
    7. Identify the set of target locations the man needs to reach. This set includes
       the locations of all loose goal nuts. If additional spanners are needed (`m > 0`),
       it also includes the locations of all available usable spanners.
    8. Calculate the minimum distance (number of 'walk' actions) from the man's
       current location to any location in the set of target locations. If the man's
       location is unknown or no target locations are reachable, return infinity.
    9. The final heuristic value is the sum of the base cost (tighten + pickup actions)
       and the minimum travel distance (walk actions).
    """

    def __init__(self, task):
        """
        Initializes the spanner heuristic.

        Args:
            task: The planning task object containing initial state, goals, static facts, etc.
        """
        self.task = task

        # 1. Identify objects by type based on predicate usage in initial/static facts
        self.man_name = None
        self.spanner_names = set()
        self.nut_names = set()
        self.location_names = set()

        all_init_facts = set(task.initial_state)
        all_static_facts = set(task.static)
        all_facts = all_init_facts | all_static_facts

        # Identify spanners and nuts first based on specific predicates
        for fact in all_facts:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'carrying':
                # The second argument of carrying is a spanner
                if len(parts) > 2: self.spanner_names.add(parts[2])
            elif predicate == 'usable':
                 if len(parts) > 1: self.spanner_names.add(parts[1])
            elif predicate in ['loose', 'tightened']:
                 if len(parts) > 1: self.nut_names.add(parts[1])

        # Identify man: Assume the object in a 'carrying' fact in initial state is the man.
        # If no 'carrying' fact, try finding a unique object in 'at' that isn't a spanner/nut.
        for fact in all_init_facts:
            if fact.startswith('(carrying '):
                self.man_name = fact.strip('()').split()[1]
                break # Assume one man

        if self.man_name is None:
             # Fallback: Find an object in an 'at' fact that isn't a known spanner or nut
             for fact in all_init_facts:
                 if fact.startswith('(at '):
                     parts = fact.strip('()').split()
                     if len(parts) > 1:
                         obj = parts[1]
                         if obj not in self.spanner_names and obj not in self.nut_names:
                             self.man_name = obj
                             break # Assume this is the man

        # Identify locations from 'at' and 'link' facts
        for fact in all_facts:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'at':
                if len(parts) > 2: self.location_names.add(parts[2])
            elif predicate == 'link':
                if len(parts) > 2:
                    self.location_names.add(parts[1])
                    self.location_names.add(parts[2])

        # 2. Build location graph
        self.location_graph = {}
        for fact in all_static_facts:
            if fact.startswith('(link '):
                parts = fact.strip('()').split()
                if len(parts) > 2:
                    loc1 = parts[1]
                    loc2 = parts[2]
                    self.location_graph.setdefault(loc1, set()).add(loc2)
                    self.location_graph.setdefault(loc2, set()).add(loc1)

        # Ensure all identified locations are in the graph keys for BFS, even if they have no links
        for loc in self.location_names:
             self.location_graph.setdefault(loc, set())

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.location_names:
            self.distances[start_loc] = self._bfs_distances(start_loc)

        # 4. Store goal nuts
        self.goal_nuts = set()
        for goal_fact in task.goals:
            if goal_fact.startswith('(tightened '):
                nut_name = goal_fact.strip('()').split()[1]
                self.goal_nuts.add(nut_name)

    def _bfs_distances(self, start_loc):
        """Helper method to compute shortest distances from a start location."""
        distances = {loc: float('inf') for loc in self.location_names}
        if start_loc in self.location_names:
            distances[start_loc] = 0
            queue = deque([start_loc])
            while queue:
                current_loc = queue.popleft()
                if current_loc in self.location_graph:
                    for neighbor in self.location_graph[current_loc]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_loc] + 1
                            queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Retrieves the precomputed shortest distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
            # This might happen if a location in the state was not identified
            # during initialization, or if locations are disconnected.
            # For solvable problems, relevant locations should be connected.
            # Return infinity to indicate unreachable.
            return float('inf')
        return self.distances[loc1][loc2]

    def h(self, state):
        """
        Computes the heuristic value for a given state.

        Args:
            state: The current state represented as a frozenset of fact strings.

        Returns:
            An integer or float('inf') representing the estimated cost to reach the goal.
        """
        man_location = None
        carried_usable_spanners = set()
        available_usable_spanners_at_loc = {} # {spanner_name: location}
        loose_nuts_in_goal = {} # {nut_name: location}

        # 1. Parse state
        # Create a set of state facts for efficient lookup
        state_facts = set(state)

        for fact in state_facts:
            if fact.startswith('(at '):
                parts = fact.strip('()').split()
                if len(parts) > 2:
                    obj = parts[1]
                    loc = parts[2]
                    if obj == self.man_name:
                         man_location = loc
                    elif obj in self.spanner_names:
                         if '(usable ' + obj + ')' in state_facts:
                             available_usable_spanners_at_loc[obj] = loc
                    elif obj in self.nut_names:
                         if '(loose ' + obj + ')' in state_facts and obj in self.goal_nuts:
                             loose_nuts_in_goal[obj] = loc
            elif fact.startswith('(carrying '):
                parts = fact.strip('()').split()
                if len(parts) > 2:
                    s = parts[2] # The spanner
                    if s in self.spanner_names and '(usable ' + s + ')' in state_facts:
                         carried_usable_spanners.add(s)

        # 2. Number of nuts to tighten
        k = len(loose_nuts_in_goal)

        # If no nuts need tightening, goal is reached
        if k == 0:
            return 0

        # 3. Number of usable spanners carried
        c = len(carried_usable_spanners)

        # 4. Number of spanners needed from locations
        m = max(0, k - c)

        # 5. Check solvability w.r.t spanners
        if m > len(available_usable_spanners_at_loc):
             # Not enough usable spanners in the world to tighten all goal nuts
             return float('inf') # Indicate unsolvable state

        # 6. Base cost: minimum tighten and pickup actions
        h_value = k + m

        # 7. Identify target locations for travel
        target_locations = set()

        # Required locations are where loose goal nuts are
        target_locations.update(loose_nuts_in_goal.values())

        # If spanners are needed, required locations also include where available usable spanners are
        if m > 0:
            target_locations.update(available_usable_spanners_at_loc.values())

        # 8. Calculate minimum distance from man's current location to any target location
        min_dist_to_target = float('inf')

        if man_location is None:
            # Man's location is unknown - indicates an invalid state representation
            return float('inf')

        if not target_locations:
            # This case should be covered by k == 0 check, but defensive
            min_dist_to_target = 0
        else:
            for target_loc in target_locations:
                dist = self.get_distance(man_location, target_loc)
                min_dist_to_target = min(min_dist_to_target, dist)

        # If min_dist_to_target is inf, it means a required location is unreachable from man's current location
        if min_dist_to_target == float('inf'):
            return float('inf')

        # 9. Total heuristic value
        h_value += min_dist_to_target

        return h_value
