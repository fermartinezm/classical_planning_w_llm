from heuristics.heuristic_base import Heuristic
# No need to import task or operator as we only use state and static info from task

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the cost to reach the goal (all passengers served) by summing
        three components:
        1. The estimated number of move actions required to visit all floors
           where unserved passengers need to be picked up or dropped off.
        2. The number of board actions needed for unserved passengers who are
           currently waiting at their origin floors.
        3. The number of depart actions needed for all unserved passengers.

    Assumptions:
        - The PDDL domain is 'miconic' as provided.
        - The floor structure defined by 'above' predicates forms a single,
          linear, ordered sequence of floors (a chain). There is typically a
          unique lowest floor and a unique highest floor. The heuristic attempts
          to map floors to numerical levels based on these predicates.
        - Passenger destinations are defined in the static facts using the
          'destin' predicate.
        - Valid states will always contain exactly one 'lift-at' fact.
        - Valid states will represent passengers as either 'origin', 'boarded',
          or 'served', and their destinations are consistent with static facts.
        - If the floor structure is not a simple linear chain or is disconnected,
          or if required floors/current floor are not in the mapped levels,
          the heuristic may return infinity, indicating a potential issue
          or unsolvable state.

    Heuristic Initialization:
        In the constructor (__init__), the heuristic pre-processes the static
        facts from the task description:
        - It builds a mapping from floor names (e.g., 'f1', 'f2') to numerical
          levels (e.g., 0, 1, ...). This is done by parsing the 'above'
          predicates to determine the floor order and performing a BFS starting
          from the lowest floor (in-degree 0 in the 'above' graph).
        - It stores the destination floor for each passenger by parsing the
          'destin' predicates into a dictionary.
        - It identifies the set of passengers that need to be served according
          to the goal state.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state (in the __call__ method):
        1. Identify the set of unserved passengers by comparing the goal
           passengers (pre-calculated) with the passengers currently marked
           as 'served' in the state.
        2. If there are no unserved passengers, the goal is reached, and the
           heuristic value is 0.
        3. Determine the current floor of the lift from the 'lift-at' fact in
           the state and find its corresponding numerical level using the
           pre-calculated floor level mapping. If the floor is not in the map,
           return infinity.
        4. Identify the set of 'required floors' that the lift must visit to
           serve the unserved passengers:
           - For each unserved passenger currently at their origin floor
             ('(origin p f)' in state), add their origin floor 'f' to the set
             (pickup stop). Ensure the origin floor is in the level map, else
             return infinity.
           - For each unserved passenger currently boarded
             ('(boarded p)' in state), look up their destination floor from
             pre-calculated destinations. Add the destination floor to the set
             (dropoff stop). Ensure the destination floor is in the level map,
             else return infinity.
           - If an unserved passenger is neither waiting nor boarded, return
             infinity (unreachable goal for this passenger).
        5. If there are no required floors to visit (meaning all unserved
           passengers are boarded and their destinations are the current floor),
           the remaining actions are just the 'depart' actions for these
           passengers. The heuristic value is the number of such passengers.
        6. If there are required floors:
           - Find the minimum and maximum numerical levels among the required
             floors.
           - Estimate the minimum number of move actions ('up' or 'down')
             required to visit all required floors, starting from the current
             floor. This is calculated as the span of the required floor levels
             plus the minimum distance from the current floor level to either
             the minimum or maximum required level:
             `moves = (max_req_level - min_req_level) + min(abs(level_curr - min_req_level), abs(level_curr - max_req_level))`
           - Count the number of board actions needed: This is the number of
             unserved passengers who are currently waiting at their origin
             floors.
           - Count the number of depart actions needed: This is the total number
             of unserved passengers.
           - The total heuristic value is the sum of the estimated move cost,
             board cost, and depart cost.
    """

    def __init__(self, task):
        super().__init__()
        self.floor_levels = {}
        self.passenger_destinations = {}
        self.goal_passengers = set()

        # Pre-process static facts
        above_graph = {} # f_j -> f_i if (above f_i f_j)
        in_degree = {}
        all_floors = set()

        for fact in task.static:
            parts = fact.strip('()').split()
            if not parts: # Handle empty fact string if any
                continue
            predicate = parts[0]

            if predicate == 'above':
                if len(parts) == 3:
                    f_i = parts[1] # floor above
                    f_j = parts[2] # floor below
                    all_floors.add(f_i)
                    all_floors.add(f_j)
                    # Edge f_j -> f_i means f_i is immediately above f_j
                    above_graph.setdefault(f_j, []).append(f_i)
                    in_degree[f_i] = in_degree.get(f_i, 0) + 1
                    in_degree.setdefault(f_j, 0) # Ensure all floors are in in_degree
            elif predicate == 'destin':
                if len(parts) == 3:
                    p = parts[1]
                    d = parts[2]
                    self.passenger_destinations[p] = d

        # Find the lowest floor (in-degree 0)
        lowest_floor = None
        # Check if there are any floors found
        if all_floors:
            # Find candidate lowest floors (in-degree 0)
            candidate_lowest_floors = [floor for floor in all_floors if in_degree.get(floor, 0) == 0]

            # In a linear structure, there should be exactly one lowest floor.
            # If there are multiple or none, the structure is unexpected.
            # For simplicity, we pick one if multiple exist. If none, handle below.
            if candidate_lowest_floors:
                 lowest_floor = candidate_lowest_floors[0]
            elif len(all_floors) == 1:
                 # Handle case with a single floor not mentioned in 'above'
                 lowest_floor = list(all_floors)[0]
            # else: lowest_floor remains None, handled below

        # Use BFS to assign levels starting from the lowest floor
        if lowest_floor and lowest_floor in all_floors: # Ensure lowest_floor is actually one of the floors found
            queue = [(lowest_floor, 0)]
            visited = {lowest_floor}
            while queue:
                current_floor, level = queue.pop(0)
                self.floor_levels[current_floor] = level
                if current_floor in above_graph:
                    for floor_above in above_graph[current_floor]:
                        if floor_above not in visited:
                            visited.add(floor_above)
                            queue.append((floor_above, level + 1))
        # else: If lowest_floor is None, self.floor_levels remains empty.
        # This will cause KeyError later if lift-at floor is not in floor_levels,
        # which is handled by returning infinity.


        # Store goal passengers for quick lookup
        self.goal_passengers = {goal.strip('()').split()[1] for goal in task.goals if goal.startswith('(served ')}


    def __call__(self, node):
        """
        Computes the domain-dependent miconic heuristic for the given state.
        """
        state = node.state

        # 1. Identify unserved passengers
        served_passengers_in_state = {fact.strip('()').split()[1] for fact in state if fact.startswith('(served ')}
        unserved_passengers = self.goal_passengers - served_passengers_in_state

        if not unserved_passengers:
            return 0 # Goal reached for all relevant passengers

        # 2. Identify current lift floor
        current_floor = None
        for fact in state:
            if fact.startswith('(lift-at '):
                parts = fact.strip('()').split()
                if len(parts) == 2:
                    current_floor = parts[1]
                    break

        # If current_floor is not found or not in floor_levels, state is invalid/unreachable
        if current_floor is None or current_floor not in self.floor_levels:
             return float('inf')

        level_curr = self.floor_levels[current_floor]

        # Store origin floors for quick lookup in this state
        passenger_origins = {}
        for fact in state:
            if fact.startswith('(origin '):
                parts = fact.strip('()').split()
                if len(parts) == 3:
                    p = parts[1]
                    f = parts[2]
                    passenger_origins[p] = f

        # 3. Identify required floors and passenger status
        unserved_passengers_waiting = set()
        unserved_passengers_boarded = set()
        pickup_floors = set()
        dropoff_floors = set()

        for p in unserved_passengers:
            if '(boarded ' + p + ')' in state:
                unserved_passengers_boarded.add(p)
                # Find destination from pre-parsed static info
                if p in self.passenger_destinations:
                     dest_f = self.passenger_destinations[p]
                     # Ensure destination floor is in our mapped floors
                     if dest_f in self.floor_levels:
                         # Only add to dropoff_floors if destination is not the current floor
                         # If destination is current floor, no move action is required to reach it for dropoff
                         if dest_f != current_floor:
                             dropoff_floors.add(dest_f)
                     else:
                         # Destination floor not in floor map - indicates problem structure issue
                         return float('inf')
                else:
                    # Passenger destination not found in static facts - indicates problem structure issue
                    return float('inf')

            elif p in passenger_origins: # Must be waiting if not boarded and not served
                unserved_passengers_waiting.add(p)
                origin_f = passenger_origins[p]
                # Ensure origin floor is in our mapped floors
                if origin_f in self.floor_levels:
                    # Only add to pickup_floors if origin is not the current floor
                    # If origin is current floor, no move action is required to reach it for pickup
                    if origin_f != current_floor:
                        pickup_floors.add(origin_f)
                else:
                    # Origin floor not in floor map - indicates problem structure issue
                    return float('inf')

            # else: Passenger is unserved but neither waiting nor boarded?
            # This shouldn't happen in valid states for this domain.
            # If it happens, this passenger is stuck and goal is unreachable.
            # Return infinity.
            else:
                 return float('inf')


        required_floors = pickup_floors | dropoff_floors

        # 4. If no required floors
        if not required_floors:
            # This implies all unserved passengers are boarded and their destination
            # is the current floor (because if destination was a different floor,
            # it would have been added to dropoff_floors).
            # The remaining cost is the number of depart actions.
            return len(unserved_passengers) # Which is equal to len(unserved_passengers_boarded) here

        # 5. If required floors exist
        required_levels = {self.floor_levels[f] for f in required_floors}
        min_req_level = min(required_levels)
        max_req_level = max(required_levels)

        # Calculate move cost
        # Moves = (span of required floors) + min(distance from current to min_req, distance from current to max_req)
        # This estimates the travel needed to cover the range of required floors starting from current.
        move_cost = (max_req_level - min_req_level) + min(abs(level_curr - min_req_level), abs(level_curr - max_req_level))

        # Calculate board cost
        # Each waiting unserved passenger needs one board action.
        board_cost = len(unserved_passengers_waiting)

        # Calculate depart cost
        # Each unserved passenger needs one depart action eventually.
        depart_cost = len(unserved_passengers)

        heuristic_value = move_cost + board_cost + depart_cost

        return heuristic_value

