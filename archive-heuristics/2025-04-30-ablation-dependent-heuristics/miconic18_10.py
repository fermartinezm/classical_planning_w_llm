import re

def parse_fact(fact_str):
    """Parses a PDDL fact string into a tuple."""
    # Remove surrounding brackets and split by spaces
    parts = fact_str[1:-1].split()
    return tuple(parts)

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the cost to reach the goal by summing two components:
    1. An action cost: 2 actions (board + depart) for each unserved passenger.
    2. A movement cost: The estimated distance the lift needs to travel to visit
       all floors where unboarded passengers need to be picked up and boarded
       passengers need to be dropped off. The movement cost is calculated
       as the sum of distances between consecutive floors in the sorted list
       of required stops, starting from the current lift floor.

    Assumptions:
    - Floor names are in the format 'fN' where N is an integer, allowing numerical sorting.
    - The 'above' facts define a linear ordering of floors, consistent with 'fN' naming.
    - The input state and static information are in the expected frozenset format
      with PDDL facts as strings.
    - Every passenger mentioned in the problem has a 'destin' fact in the static information.
    - Every unboarded, unserved passenger has an 'origin' fact in the current state.
    - The 'lift-at' fact is always present in the state.

    Heuristic Initialization:
    - Parses static facts and initial state facts to identify all floor objects.
    - Sorts floor objects numerically based on their names (e.g., f1, f2, f10).
    - Creates mappings between floor names and their numerical indices for distance calculation.
    - Parses static facts to create a mapping from passenger names to their
      destination floor names.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state using the task's goal_reached method. If yes, return 0.
    2. Identify the current floor of the lift by finding the '(lift-at ?f)' fact in the state.
    3. Initialize the heuristic value `h` to 0.
    4. Initialize an empty set `required_floors_set` to store floors the lift must visit.
    5. Create sets of served and boarded passengers by iterating through the state facts.
    6. Create a map from passenger to origin floor for unboarded passengers by iterating through state facts.
    7. Iterate through all known passengers (from the passenger-to-destination map initialized from static facts).
    8. For each passenger:
       a. Check if the passenger is in the set of served passengers. If yes, skip this passenger.
       b. If not served, add 2 to `h` (representing the board and depart actions needed for this passenger).
       c. Check if the passenger is in the set of boarded passengers.
       d. If boarded, add the passenger's destination floor (from the pre-computed map) to `required_floors_set`.
       e. If not boarded, find the passenger's origin floor from the origin map created in step 6. Add the origin floor to `required_floors_set`. Add the passenger's destination floor to `required_floors_set`.
    9. If `required_floors_set` is empty (meaning all passengers are served), the movement cost is 0.
    10. If `required_floors_set` is not empty:
       a. Get the numerical indices for all floors in `required_floors_set` using the `floor_to_index` map.
       b. Sort these indices.
       c. Get the numerical index of the current lift floor.
       d. Calculate the movement cost: Start with the absolute difference between the current lift floor index and the index of the first required floor in the sorted list. Then, add the absolute difference between each consecutive pair of required floor indices in the sorted list.
       e. Add the calculated movement cost to `h`.
    11. Return the final heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static domain information.

        Args:
            task: The planning task object (instance of the Task class).
        """
        self.task = task
        self.floor_to_index = {}
        self.index_to_floor = {}
        self.passenger_to_destin = {}

        # 1. Parse floors and determine order
        floor_names = set()
        # Collect all unique floor names from relevant predicates in static and initial state
        for fact_str in task.static | task.initial_state:
            predicate, *args = parse_fact(fact_str)
            if predicate in ('above', 'origin', 'destin', 'lift-at'):
                for arg in args:
                    # Simple check if it looks like a floor name 'fN'
                    if arg.startswith('f') and arg[1:].isdigit():
                         floor_names.add(arg)

        # Sort floors numerically (assuming f1, f2, f10, f20 etc.)
        # This assumes a standard naming convention and linear floor structure.
        sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))
        for i, floor_name in enumerate(sorted_floor_names):
            self.floor_to_index[floor_name] = i
            self.index_to_floor[i] = floor_name

        # 2. Parse passenger destinations from static facts
        for fact_str in task.static:
            predicate, *args = parse_fact(fact_str)
            if predicate == 'destin':
                p, f = args
                self.passenger_to_destin[p] = f

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            The estimated number of actions to reach the goal.
        """
        # 1. Check for goal state
        if self.task.goal_reached(state):
            return 0

        # 2. Find current lift floor
        current_lift_floor = None
        for fact_str in state:
            if fact_str.startswith('(lift-at '):
                current_lift_floor = parse_fact(fact_str)[1]
                break

        # Defensive check (should not happen in valid states)
        if current_lift_floor is None:
             # If lift location is unknown, this state is likely unreachable or invalid
             # Returning a very high value discourages the search from exploring here.
             # In a real planner, you might raise an error or return float('inf').
             # For this context, let's assume valid states always have lift-at.
             pass


        # 3. Initialize heuristic and required stops
        h = 0
        required_floors_set = set()

        # Pre-process state to quickly check passenger status and origins
        served_passengers = set()
        boarded_passengers = set()
        origin_map = {} # Map passenger -> origin floor for unboarded passengers

        for fact_str in state:
            if fact_str.startswith('(served '):
                served_passengers.add(parse_fact(fact_str)[1])
            elif fact_str.startswith('(boarded '):
                boarded_passengers.add(parse_fact(fact_str)[1])
            elif fact_str.startswith('(origin '):
                 p, f = parse_fact(fact_str)[1:]
                 origin_map[p] = f # Store origin for quick lookup

        # 7. Calculate action cost and identify required floors
        # Iterate through all passengers known from the problem definition (via destin facts)
        for p in self.passenger_to_destin.keys():
            if p in served_passengers:
                # This passenger is already served, no cost contribution
                continue

            # This passenger is unserved
            # Each unserved passenger needs board + depart = 2 actions
            h += 2

            if p in boarded_passengers:
                # Boarded passenger needs to be dropped off at destination
                required_floors_set.add(self.passenger_to_destin[p])
            else:
                # Unboarded passenger needs to be picked up at origin and dropped off at destination
                # Need to find origin floor from the origin_map built from state
                f_origin = origin_map.get(p)
                # Defensive check: An unboarded, unserved passenger should have an origin
                if f_origin:
                    required_floors_set.add(f_origin)
                    required_floors_set.add(self.passenger_to_destin[p])
                # else: This passenger's origin is unknown in the state.
                # This might indicate an invalid state or a passenger that cannot be picked up.
                # For this heuristic, we assume valid states where unboarded passengers have origins.


        # 9. Calculate movement cost
        movement_cost = 0
        if required_floors_set:
            # Get indices for required floors and sort them
            required_indices = sorted([self.floor_to_index[f] for f in required_floors_set])

            # Get index of current lift floor
            current_lift_index = self.floor_to_index[current_lift_floor]

            # Calculate distance from current lift floor to the first required stop
            movement_cost += abs(required_indices[0] - current_lift_index)

            # Calculate distance between consecutive required stops in the sorted list
            for i in range(len(required_indices) - 1):
                movement_cost += abs(required_indices[i+1] - required_indices[i])

        # 10. Add movement cost to heuristic
        h += movement_cost

        # 11. Return the final heuristic value
        return h
