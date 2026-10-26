import collections

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Estimates the cost to reach the goal state by considering the number of
    nuts remaining to be tightened, the spanners needed, and the walking costs.
    Assumes all goal nuts are at a single, fixed location.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing distances and identifying
        key objects and locations.

        Heuristic Initialization:
        1. Parses static facts to build the graph of locations connected by links.
        2. Infers object types (man, spanner, nut, location) by examining which
           objects appear in specific argument positions of predicates in the
           initial state and goal facts.
        3. Computes all-pairs shortest paths between all identified locations
           using BFS.
        4. Identifies the set of goal nuts from the task's goal facts.
        5. Determines the single location for all goal nuts by finding the location
           of one goal nut in the initial state.
        """
        self.task = task
        self.locations = set()
        self.graph = collections.defaultdict(set)
        self.distances = {}

        self.man_name = None
        self.nut_goal_location = None
        self.nuts_in_problem = set()
        self.spanners_in_problem = set()

        # 1. Build graph and find all locations from static facts
        for fact in self.task.static:
            if fact.startswith('(link '):
                parts = fact[1:-1].split()
                if len(parts) == 3:
                    l1, l2 = parts[1], parts[2]
                    self.graph[l1].add(l2)
                    self.graph[l2].add(l1)
                    self.locations.add(l1)
                    self.locations.add(l2)

        # 3. Identify object types and names from initial state and goal
        all_objects_set = set()
        potential_man = set()
        potential_spanner = set()
        potential_nut = set()
        potential_location = self.locations.copy() # Start with locations from links

        # Scan initial state and goal for predicate usage
        for fact in self.task.initial_state.union(self.task.goals):
            parts = fact[1:-1].split()
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]
            all_objects_set.update(args)

            if predicate == 'at' and len(args) == 2:
                 obj, loc = args
                 # The second argument of 'at' is always a location
                 potential_location.add(loc)
                 # The first argument could be man, spanner, or nut
            elif predicate == 'carrying' and len(args) == 2:
                 m, s = args
                 # First arg is man, second is spanner
                 potential_man.add(m)
                 potential_spanner.add(s)
            elif predicate == 'usable' and len(args) == 1:
                 # Arg is a spanner
                 potential_spanner.add(args[0])
            elif predicate == 'tightened' and len(args) == 1:
                 # Arg is a nut
                 potential_nut.add(args[0])
            elif predicate == 'loose' and len(args) == 1:
                 # Arg is a nut
                 potential_nut.add(args[0])
            # 'link' facts handled already

        # Refine object types based on potential sets
        # An object is a nut if it's in potential_nut and not in others
        self.nuts_in_problem = potential_nut - potential_spanner - potential_man - potential_location
        # An object is a spanner if it's in potential_spanner and not in others (except maybe location if it's at one)
        self.spanners_in_problem = potential_spanner - potential_nut - potential_man # Spanners can be at locations, so don't exclude locations yet
        # An object is a man if it's in potential_man and not in others
        self.man_name = next(iter(potential_man - potential_spanner - potential_nut - potential_location), None) # Assuming one man

        # Re-check spanners: an object is a spanner if it's in potential_spanner AND it's not the man AND it's not a nut.
        self.spanners_in_problem = potential_spanner - {self.man_name} - self.nuts_in_problem

        # Re-check locations: any object that is not man, nut, or spanner, but appears in an (at obj loc) fact as 'loc' is a location.
        # Also, locations from links are locations.
        locations_from_at = set()
        for fact in self.task.initial_state:
             if fact.startswith('(at '):
                 parts = fact[1:-1].split()
                 if len(parts) == 3:
                     obj, loc = parts[1], parts[2]
                     # Check if 'loc' is a potential location based on exclusion
                     if loc in all_objects_set and loc not in self.nuts_in_problem and loc not in self.spanners_in_problem and loc != self.man_name:
                          locations_from_at.add(loc)
        self.locations.update(locations_from_at) # Add any locations found in 'at' facts

        # 2. Compute all-pairs shortest paths (after identifying all locations)
        self.distances = {} # Clear old distances if any
        for start_node in list(self.locations):
            q = collections.deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0
            while q:
                curr_node, dist = q.popleft()
                for neighbor in self.graph.get(curr_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))


        # 4. Identify goal nuts and their location
        goal_nuts = {fact[1:-1].split()[1] for fact in self.task.goals if fact.startswith('(tightened ') and len(fact[1:-1].split()) == 2}

        # Find location of one goal nut from initial state
        first_goal_nut = next(iter(goal_nuts), None)
        if first_goal_nut:
            for fact in self.task.initial_state:
                if fact.startswith('(at '):
                    parts = fact[1:-1].split()
                    if len(parts) == 3:
                        obj, loc = parts[1], parts[2]
                        if obj == first_goal_nut:
                            self.nut_goal_location = loc
                            break
        # Assume all goal nuts are at this same location self.nut_goal_location
        # If first_goal_nut is None, goal is empty, handled in __call__
        # If nut_goal_location is None, the goal nut is not placed in initial state? Problematic.
        # Assume solvable problems have goal nuts placed in initial state.


    def __call__(self, state):
        """
        spannerHeuristic estimates the cost to reach the goal state.

        Summary:
        The heuristic estimates the cost by summing the number of remaining
        tighten_nut actions, the number of pickup_spanner actions required
        to obtain enough spanners, and the estimated walking cost.
        It assumes all goal nuts are located at the same fixed location.

        Assumptions:
        - There is exactly one man object.
        - All nuts required to be tightened in the goal are located at the same
          single location throughout the problem. This location is determined
          from the initial state of one of the goal nuts.
        - The problem is solvable (i.e., there are enough usable spanners
          available initially to tighten all goal nuts).
        - The heuristic calculates walking costs using precomputed shortest paths
          on the location graph.
        - Object types (man, spanner, nut, location) are inferred from their
          usage in predicates in the initial state and goal.

        Heuristic Initialization:
        1. Parses static facts to build the graph of locations connected by links.
        2. Infers object types (man, spanner, nut, location) by examining which
           objects appear in specific argument positions of predicates in the
           initial state and goal facts.
        3. Computes all-pairs shortest paths between all identified locations
           using BFS.
        4. Identifies the set of goal nuts from the task's goal facts.
        5. Determines the single location for all goal nuts by finding the location
           of one goal nut in the initial state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state. If yes, return 0.
        2. Identify the set of loose nuts that need to be tightened: these are the
           nuts present in the task's goal state that are not marked as 'tightened'
           in the current state. Let N be the count of these nuts.
        3. If N is 0, the heuristic is 0 (already checked in step 1, but defensive).
        4. Find the man's current location in the state. If not found (shouldn't happen in valid states), return infinity.
        5. Find the set of usable spanners currently carried by the man.
        6. Find the set of usable spanners currently located at various locations
           (not carried by the man). Store their locations.
        7. Calculate the number of additional spanners the man needs to pick up:
           This is max(0, N - number of usable spanners currently carried).
           Let this be K.
        8. The base action cost is N (for tighten_nut) + K (for pickup_spanner).
        9. Estimate the walking cost:
           a. The man needs to walk from his current location to the location of
              the goal nuts (self.nut_goal_location) to use the spanners he is
              carrying. The cost is the shortest distance between his location
              and the goal nut location. If unreachable, return infinity.
           b. For the K spanners he needs to pick up, he must travel from the
              goal nut location to a spanner location, pick it up, and return
              to the goal nut location. To minimize walking, he should pick up
              spanners from the K usable spanners at locations that are closest
              to the goal nut location.
           c. Calculate the sum of (distance from goal nut location to spanner location +
              distance from spanner location back to goal nut location) for the K
              closest usable spanners at locations. If any required spanner location
              is unreachable from the nut location, or there aren't enough usable
              spanners at locations, return infinity.
        10. The total heuristic value is the base action cost plus the estimated
            walking cost.
        """
        if self.task.goal_reached(state):
            return 0

        # 2. Identify loose nuts needing tightening
        goal_nuts = {fact[1:-1].split()[1] for fact in self.task.goals if fact.startswith('(tightened ') and len(fact[1:-1].split()) == 2}
        tightened_nuts_in_state = {fact[1:-1].split()[1] for fact in state if fact.startswith('(tightened ') and len(fact[1:-1].split()) == 2}
        loose_nuts_to_tighten = goal_nuts - tightened_nuts_in_state
        N = len(loose_nuts_to_tighten)

        if N == 0:
            return 0 # Should be caught by goal_reached, but safe check

        # 4. Find man's location
        man_loc = None
        # Ensure man_name was successfully identified during init
        if self.man_name:
            for fact in state:
                if fact.startswith('(at '):
                    parts = fact[1:-1].split()
                    if len(parts) == 3 and parts[1] == self.man_name:
                        man_loc = parts[2]
                        break
        if man_loc is None:
             # Man location not found or man_name not identified.
             # Should not happen in valid states/problems.
             return float('inf')


        # 5. Find usable spanners carried
        spanners_carried = {fact[2] for fact in state if fact.startswith('(carrying ') and len(fact[1:-1].split()) == 3 and fact[1] == self.man_name}
        usable_spanners_in_state = {fact[1] for fact in state if fact.startswith('(usable ') and len(fact[1:-1].split()) == 2}
        usable_spanners_carried = spanners_carried.intersection(usable_spanners_in_state)

        # 6. Find usable spanners at locations
        usable_spanners_at_locs = {} # spanner -> location
        for fact in state:
            if fact.startswith('(at '):
                parts = fact[1:-1].split()
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    # Check if obj is a spanner (identified during init) and is usable and not carried
                    if obj in self.spanners_in_problem and obj in usable_spanners_in_state and obj not in spanners_carried:
                         usable_spanners_at_locs[obj] = loc

        # 7. Calculate spanners needed from locations
        spanners_needed_from_locs_count = max(0, N - len(usable_spanners_carried))

        # 8. Base action cost
        base_cost = N + spanners_needed_from_locs_count

        # 9. Estimate walking cost
        l_man = man_loc
        l_nut_goal = self.nut_goal_location

        # If nut_goal_location was not found during init (e.g., no goal nuts, or goal nut not in initial state)
        if l_nut_goal is None:
             # This case indicates a potentially malformed problem or an assumption violation.
             # For a solvable problem with tightened nuts as goal, the nut must exist and be placed.
             # Return infinity as a safe fallback.
             return float('inf')


        # Cost to reach the first nut location (assuming using carried spanners first)
        dist_to_first_nut = self.distances.get((l_man, l_nut_goal), float('inf'))
        if dist_to_first_nut == float('inf'):
             return float('inf') # Unreachable nut location

        walking_cost = dist_to_first_nut

        # Cost for subsequent pickups and returns
        if spanners_needed_from_locs_count > 0:
            usable_spanner_locations_list = list(usable_spanners_at_locs.values())

            # If not enough usable spanners at locations for pickup, problem is unsolvable from here
            if len(usable_spanner_locations_list) < spanners_needed_from_locs_count:
                 return float('inf') # Not enough spanners to pick up

            # Sort usable spanner locations by distance from the nut goal location
            # Use a list of (distance, location) tuples for sorting stability and easy slicing
            locations_with_dist = []
            for l in usable_spanner_locations_list:
                 dist = self.distances.get((l_nut_goal, l), float('inf'))
                 if dist == float('inf'):
                      # If any potential pickup location is unreachable, the required spanner might be there
                      # This branch of the heuristic path is infinite.
                      return float('inf')
                 locations_with_dist.append((dist, l))

            locations_with_dist.sort()

            # Take the locations of the K closest spanners
            L_pickup = [loc for dist, loc in locations_with_dist[:spanners_needed_from_locs_count]]

            # Calculate walking cost for picking up and returning
            pickup_return_walking_cost = 0
            for l_s in L_pickup:
                # We already checked reachability from l_nut_goal to l_s during sorting prep
                # Need to check reachability back from l_s to l_nut_goal
                dist_to_spanner = self.distances[(l_nut_goal, l_s)] # Already checked not inf
                dist_back_to_nut = self.distances.get((l_s, l_nut_goal), float('inf'))
                if dist_back_to_nut == float('inf'):
                     return float('inf') # Cannot return to nut location after pickup
                pickup_return_walking_cost += dist_to_spanner + dist_back_to_nut

            walking_cost += pickup_return_walking_cost

        # 10. Total heuristic
        return base_cost + walking_cost
