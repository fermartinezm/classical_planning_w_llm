class miconicHeuristic:
    """
    Domain-dependent heuristic for the miconic domain.

    Summary:
    Estimates the cost to reach the goal by summing the number of board actions,
    the number of depart actions, and an estimate of the minimum movement actions
    required. The movement estimate is the total vertical distance spanning the
    current lift floor, all origin floors of waiting passengers, and all
    destination floors of boarded passengers.

    Assumptions:
    - Floor names are consistently structured (e.g., 'f1', 'f2', 'f10') allowing
      parsing of an integer floor number.
    - The `(above f_lower f_higher)` facts define a linear, contiguous order
      of floors, where `f_higher` is immediately above `f_lower`.
    - The goal is to serve all passengers, i.e., achieve `(served p)` for every
      passenger `p` in the problem.

    Heuristic Initialization:
    The constructor `__init__` parses the static facts from the task.
    It builds a mapping from floor names to integer floor numbers by
    identifying the lowest floor and traversing upwards using `(above ...)`
    facts. It also builds a mapping from passenger names to their destination
    floor names using `(destin ...)` facts. It also identifies the set of all
    passengers in the problem. This pre-computation is done once when the
    heuristic object is created.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the state is a goal state. A state is a goal state if all
       passengers identified during initialization are marked as `served` in
       the current state. If it is a goal state, the heuristic value is 0.
    2. Extract the current floor of the lift from the state fact `(lift-at ...)`.
       Convert the floor name to its integer number using the pre-computed map.
    3. Iterate through the state facts to identify:
       a. Waiting passengers: Those with `(origin p f)` facts. Count them (`num_board_needed`)
          and collect their origin floor numbers.
       b. Boarded passengers: Those with `(boarded p)` facts. Count them (`num_depart_needed`).
          For each boarded passenger, find their destination floor name using the
          pre-computed destination map and add the corresponding floor number to
          the set of required floors.
    4. Collect the set of *required floor numbers*. This set includes the integer
       floor numbers of all origin floors of waiting passengers and the
       destination floors of all boarded passengers.
    5. Calculate the movement actions needed:
       a. Create a set `all_relevant_nums` containing the integer number of the
          current lift floor and all numbers in `required_floor_nums`.
       b. If `all_relevant_nums` contains 0 or 1 unique floor number, the movement
          cost is 0.
       c. Otherwise, find the minimum (`min_all`) and maximum (`max_all`) floor
          numbers in `all_relevant_nums`. The estimated movement cost is
          `max_all - min_all`. This represents the vertical span the lift must
          cover.
    6. The total heuristic value is the sum of the estimated movement cost,
       the number of board actions needed (`num_board_needed`), and the number
       of depart actions needed (`num_depart_needed`). This sum correctly handles
       the case where no movement is needed (when `required_floor_nums` is empty
       and thus `all_relevant_nums` only contains the current floor).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static task information.

        @param task: The planning task object.
        """
        self.task = task
        self.floor_map = {} # Map floor name to integer
        self.destinations = {} # Map passenger name to destination floor name
        self._parse_static(task.static)
        self._all_passengers = self._get_all_passengers(task.initial_state, task.static)

    def _parse_static(self, static_facts):
        """
        Parses static facts to build floor map and destination map.
        """
        above_map = {} # Map f_lower to f_higher
        floor_names = set()
        for fact in static_facts:
            if fact.startswith('(above '):
                parts = fact[1:-1].split()
                f_lower = parts[1]
                f_higher = parts[2]
                above_map[f_lower] = f_higher
                floor_names.add(f_lower)
                floor_names.add(f_higher)
            elif fact.startswith('(destin '):
                parts = fact[1:-1].split()
                p = parts[1]
                f = parts[2]
                self.destinations[p] = f

        # Build floor order and map names to numbers
        # Find the lowest floor (a floor f such that no (above ?x f) exists)
        higher_floors_in_above = set(above_map.values())
        lower_floors_in_above = set(above_map.keys())

        lowest_floor = None
        # Find the floor that is a lower floor but never a higher floor
        potential_lowest = lower_floors_in_above - higher_floors_in_above
        if potential_lowest:
             lowest_floor = potential_lowest.pop()
        elif floor_names and not above_map and len(floor_names) == 1:
             # Case with a single floor and no above facts
             lowest_floor = list(floor_names)[0]
        # else: Problem structure is not a simple chain or single floor.
        # Assuming valid miconic structure where lowest_floor is found.

        if lowest_floor:
            current_floor = lowest_floor
            current_floor_num = 1
            self.floor_map[current_floor] = current_floor_num

            # Traverse upwards
            while current_floor in above_map:
                next_floor = above_map[current_floor]
                current_floor = next_floor
                current_floor_num += 1
                self.floor_map[current_floor] = current_floor_num


    def _get_all_passengers(self, initial_state, static_facts):
        """
        Identifies all passengers in the problem.
        Assumes passengers appear in initial state (origin/boarded) or static (destin).
        """
        passengers = set()
        for fact in initial_state:
            if fact.startswith('(origin '):
                parts = fact[1:-1].split()
                passengers.add(parts[1])
            elif fact.startswith('(boarded '):
                 parts = fact[1:-1].split()
                 passengers.add(parts[1])
        for fact in static_facts:
             if fact.startswith('(destin '):
                 parts = fact[1:-1].split()
                 passengers.add(parts[1])
        return frozenset(passengers)


    def __call__(self, state, task):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @param task: The planning task object (used for goal info, though we infer served from state).
        @return: The estimated number of actions to reach the goal.
        """
        # Check for goal state
        served_passengers = {p for fact in state if fact.startswith('(served ') for p in [fact[1:-1].split()[1]]}
        if served_passengers == self._all_passengers:
             return 0

        # Extract current lift floor
        current_floor_name = None
        for fact in state:
            if fact.startswith('(lift-at '):
                current_floor_name = fact[1:-1].split()[1]
                break
        # Assuming valid states always have lift-at.
        current_floor_num = self.floor_map[current_floor_name]


        # Identify required floors to visit and count board/depart actions needed
        required_floor_nums = set()
        num_board_needed = 0
        num_depart_needed = 0

        boarded_passengers = set()

        for fact in state:
            if fact.startswith('(origin '):
                parts = fact[1:-1].split()
                # p = parts[1] # passenger name not needed here
                f_origin = parts[2]
                required_floor_nums.add(self.floor_map[f_origin])
                num_board_needed += 1
            elif fact.startswith('(boarded '):
                parts = fact[1:-1].split()
                p = parts[1]
                boarded_passengers.add(p)
                num_depart_needed += 1 # Each boarded passenger needs a depart action

        # Add destination floors of boarded passengers to required floors
        for p in boarded_passengers:
             f_destin = self.destinations.get(p)
             if f_destin: # Should always exist if passenger is in the problem
                 required_floor_nums.add(self.floor_map[f_destin])

        # Calculate movement cost
        all_relevant_nums = required_floor_nums | {current_floor_num}

        movement_cost = 0
        if len(all_relevant_nums) > 1:
            min_all = min(all_relevant_nums)
            max_all = max(all_relevant_nums)
            movement_cost = max_all - min_all

        # Total heuristic
        # The sum works for all cases, including when movement_cost is 0.
        return movement_cost + num_board_needed + num_depart_needed
