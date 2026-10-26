import re

class miconicHeuristic:
    """
    Domain-dependent heuristic for the miconic domain.

    Summary:
    Estimates the cost to reach the goal by summing the estimated costs
    for each unserved passenger. The estimated cost for a passenger depends
    on whether they are at their origin or boarded, and includes lift movement
    to their relevant floors (origin for pickup, destination for dropoff)
    plus the board/depart actions. Lift movements for different passengers
    are not shared in this calculation, leading to an overestimation which
    is suitable for greedy best-first search.

    Assumptions:
    - Floor names are strings starting with 'f' followed by one or more digits (e.g., 'f1', 'f10').
    - The 'above' predicates define a total order on floors such that `(above f_a f_b)` is true
      if and only if the number in `f_a` is smaller than the number in `f_b`.
    - Floor levels are assigned such that a smaller number in the floor name corresponds to a higher level.
      Specifically, if there are N floors f1, f2, ..., fN (sorted numerically), f1 is level N,
      f2 is level N-1, ..., fN is level 1.
    - The input state and static facts follow the PDDL structure and internal
      representation (frozenset of strings).
    - The task is solvable.

    Heuristic Initialization:
    The constructor processes the static facts to:
    1. Identify all floor objects by parsing facts in the initial state and static information.
    2. Assign unique integer levels to floors based on the assumption that floor 'fN' is at level `TotalFloors - N + 1`,
       where `TotalFloors` is the total number of distinct floors and N is the number parsed from the floor name.
       This mapping ensures that `(above f_a f_b)` implies `level(f_a) > level(f_b)` if `a < b`.
    3. Store the destination floor for each passenger from the 'destin' static facts.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the current floor of the lift by finding the fact '(lift-at ?f)' in the state.
    2. Identify all passengers who are not yet 'served'. These are the unserved passengers.
    3. For each unserved passenger:
       a. Determine if the passenger is at their origin floor (fact '(origin ?p ?f)' in state)
          or if they are boarded (fact '(boarded ?p)' in state).
       b. Get the passenger's destination floor from the pre-calculated destinations map.
       c. Calculate the estimated cost for this passenger:
          - If the passenger is at their origin floor 'f_origin':
            Cost = (moves from current lift floor to f_origin) + 1 (board) +
                   (moves from f_origin to f_destin) + 1 (depart).
            Moves between floors are calculated as the absolute difference in their levels.
            Cost = abs(level(current_lift_floor) - level(f_origin)) + 1 + \
                   abs(level(origin_floor) - level(destin_floor)) + 1.
          - If the passenger is boarded:
            Cost = (moves from current lift floor to f_destin) + 1 (depart).
            Moves between floors are calculated as the absolute difference in their levels.
            Cost = abs(level(current_lift_floor) - level(destin_floor)) + 1.
    4. The total heuristic value is the sum of the estimated costs for all unserved passengers.
    5. If all passengers are served, the set of unserved passengers is empty, the sum is 0, which is correct for a goal state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        @param task: The planning task object.
        """
        self.task = task
        self.floor_levels = {}
        self.passenger_destinations = {}
        self._process_static_info()

    def _process_static_info(self):
        """
        Extracts floor levels and passenger destinations from static facts.
        Assigns levels based on numerical order of floor names.
        """
        floor_names = set()
        passengers = set()

        # Collect all potential floor and passenger objects from static and initial state
        all_facts = self.task.static | self.task.initial_state
        for fact_str in all_facts:
            # Remove parentheses and split
            parts = fact_str[1:-1].split()
            predicate = parts[0]
            args = parts[1:]

            if predicate in ['origin', 'destin']:
                if len(args) > 1: # Ensure there are arguments
                    passengers.add(args[0])
                    floor_names.add(args[1])
            elif predicate == 'above':
                 if len(args) > 1:
                    floor_names.add(args[0])
                    floor_names.add(args[1])
            elif predicate == 'lift-at':
                 if len(args) > 0:
                    floor_names.add(args[0])
            elif predicate in ['boarded', 'served']:
                 if len(args) > 0:
                    passengers.add(args[0])

        # Sort floor names numerically to assign levels
        # Assumes floor names are 'f' followed by digits
        # Assign levels in reverse numerical order of floor name
        # f1 gets highest level, fN gets lowest level
        sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))
        total_floors = len(sorted_floor_names)
        # Map f_i to level (total_floors - i + 1) where i is the 0-based index after numerical sort
        # Example: f1, f2, f3. N=3.
        # f1 is at index 0 -> level 3 - 0 = 3
        # f2 is at index 1 -> level 3 - 1 = 2
        # f3 is at index 2 -> level 3 - 2 = 1
        self.floor_levels = {f: total_floors - i for i, f in enumerate(sorted_floor_names)}


        # Extract passenger destinations from static facts
        self.passenger_destinations = {}
        for fact_str in self.task.static:
            if fact_str.startswith('(destin '):
                parts = fact_str[1:-1].split()
                if len(parts) > 2:
                    p = parts[1]
                    f = parts[2]
                    self.passenger_destinations[p] = f

    def h(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated number of actions to reach the goal.
        """
        heuristic_value = 0
        current_lift_floor = None

        # Find current lift floor
        for fact_str in state:
            if fact_str.startswith('(lift-at '):
                parts = fact_str[1:-1].split()
                if len(parts) > 1:
                    current_lift_floor = parts[1]
                    break

        if current_lift_floor is None:
             # Should not happen in a valid miconic state reachable from initial
             # where lift-at is always true. Return infinity or a large value.
             # For solvable states, this won't be reached.
             return float('inf')

        current_level = self.floor_levels.get(current_lift_floor)
        if current_level is None:
             # Should not happen if floor names are parsed correctly
             return float('inf') # Or handle as error


        # Identify unserved passengers
        all_passengers = set(self.passenger_destinations.keys())
        served_passengers = {fact_str[1:-1].split()[1] for fact_str in state if fact_str.startswith('(served ')}
        unserved_passengers = all_passengers - served_passengers

        if not unserved_passengers:
            # Goal state
            return 0

        # Calculate cost for each unserved passenger
        for p in unserved_passengers:
            destin_floor = self.passenger_destinations.get(p)
            if destin_floor is None:
                 # Should not happen for passengers defined in static facts
                 continue # Or handle as error

            destin_level = self.floor_levels.get(destin_floor)
            if destin_level is None:
                 # Should not happen if floor names are parsed correctly
                 continue # Or handle as error


            # Check if passenger is at origin
            is_at_origin = False
            origin_floor = None
            for fact_str in state:
                if fact_str.startswith(f'(origin {p} '):
                    parts = fact_str[1:-1].split()
                    if len(parts) > 2:
                        origin_floor = parts[2]
                        is_at_origin = True
                        break

            # Check if passenger is boarded
            is_boarded = f'(boarded {p})' in state

            if is_at_origin:
                origin_level = self.floor_levels.get(origin_floor)
                if origin_level is None:
                     continue # Or handle as error

                # Cost: move to origin + board + move from origin to destin + depart
                # abs(current_level - origin_level) is moves to origin
                # 1 is board action
                # abs(origin_level - destin_level) is moves from origin to destin
                # 1 is depart action
                cost_p = abs(current_level - origin_level) + 1 + abs(origin_level - destin_level) + 1
                heuristic_value += cost_p
            elif is_boarded:
                # Cost: move to destin + depart
                # abs(current_level - destin_level) is moves to destin
                # 1 is depart action
                cost_p = abs(current_level - destin_level) + 1
                heuristic_value += cost_p
            # else: passenger is unserved but neither at origin nor boarded.
            # This case implies the passenger is at their destination floor but not served,
            # and they are neither at origin (meaning origin != destination or origin fact removed)
            # nor boarded. This state is likely unreachable in a valid plan for a solvable problem.
            # We assume solvable states and the two cases above cover all unserved passengers.

        return heuristic_value
