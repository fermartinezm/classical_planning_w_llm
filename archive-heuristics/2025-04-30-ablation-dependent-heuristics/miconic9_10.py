# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """Helper to parse a PDDL fact string into predicate and arguments."""
    # Remove surrounding brackets and split by space
    parts = fact_str[1:-1].split()
    return parts[0], parts[1:]

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    This heuristic estimates the remaining cost to solve a Miconic planning
    problem instance. It combines an estimate of the required passenger
    actions (boarding and departing) with an estimate of the required lift
    movement. It is designed for greedy best-first search and is not admissible.

    Assumptions:
    - Floor names follow a consistent pattern (e.g., 'f1', 'f2', 'f10') that
      allows them to be sorted numerically to determine floor levels.
    - The `(above f1 f2)` predicates in the static information define a linear
      ordering of floors consistent with the numerical sorting.
    - Passenger destinations are provided as static facts `(destin ?p ?f)`.
    - The goal state is defined as a conjunction of `(served ?p)` predicates
      for a specific set of passengers.
    - The state representation includes exactly one `(lift-at ?f)` fact.

    Heuristic Initialization:
    The constructor `__init__` processes the static information and the goal
    description from the planning task.
    - It identifies all floor objects mentioned in `(above)` facts and creates
      a mapping from floor names to numerical levels by sorting the names.
    - It stores the destination floor for each passenger from `(destin)` facts.
    - It identifies the set of passengers that must be served to satisfy the goal
      from `(served)` facts in the goal state.

    Step-By-Step Thinking for Computing Heuristic:
    The `__call__` method computes the heuristic value for a given state:
    1. It first checks if the current state is a goal state using `self.task.goal_reached(state)`.
       If it is, the heuristic value is 0.
    2. It identifies the set of passengers that are included in the goal but
       are not yet marked as `served` in the current state. These are the
       unserved goal passengers.
    3. It then examines the current state to determine which of these unserved
       passengers are waiting at their origin floor (`origin` predicate) and
       which are currently inside the lift (`boarded` predicate).
    4. An 'action cost' component is calculated as the total number of `board`
       and `depart` actions required for the unserved passengers. This is simply
       the count of waiting passengers (each needs a `board`) plus the count
       of boarded passengers (each needs a `depart`).
    5. It identifies the set of floors that the lift needs to visit to service
       these unserved passengers. This includes the origin floor for each
       waiting passenger and the destination floor for each boarded passenger.
    6. A 'movement cost' component is calculated based on the current lift
       position and the required stop floors.
       - If there are no required stop floors (meaning all unserved passengers
         are boarded and the lift is already at their destination floors), the
         movement cost is 0.
       - Otherwise, it finds the minimum and maximum floor levels among the
         required stop floors. The movement cost is estimated as the vertical
         distance spanning the required floors plus the distance from the
         current lift floor to the nearest end of this span. This encourages
         the lift to move towards the range of floors needing service and
         then traverse that range.
    7. The total heuristic value for the state is the sum of the calculated
       action cost and the movement cost.
    """

    def __init__(self, task):
        """
        Heuristic Initialization.

        Parses static information from the task, including passenger
        destinations, floor ordering, and the set of passengers that
        need to be served to reach the goal.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.destinations = {}
        self.floor_levels = {}
        self.goal_passengers = set()

        floor_names = set()

        # Parse static facts to get destinations and floor names
        for fact_str in task.static:
            pred, args = parse_fact(fact_str)
            if pred == 'destin':
                p, f = args
                self.destinations[p] = f
            elif pred == 'above':
                f1, f2 = args
                floor_names.add(f1)
                floor_names.add(f2)

        # Parse goal facts to get the set of passengers to be served
        for fact_str in task.goals:
            pred, args = parse_fact(fact_str)
            if pred == 'served':
                p = args[0]
                self.goal_passengers.add(p)

        # Determine floor levels by sorting floor names numerically.
        # This assumes floor names follow a pattern like f1, f2, f10, etc.
        def sort_key(floor_name):
            # Extracts the numerical part of the floor name
            # Assumes floor_name starts with 'f' followed by digits
            return int(floor_name[1:])

        sorted_floor_names = sorted(list(floor_names), key=sort_key)
        for i, floor in enumerate(sorted_floor_names):
            self.floor_levels[floor] = i + 1

        # Note: If floor_names is empty, floor_levels will be empty.
        # This is handled implicitly in __call__ if F_stops is empty.


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            An integer estimate of the remaining cost to reach the goal.
        """
        # 1. Check if the current state is a goal state.
        if self.task.goal_reached(state):
            return 0

        f_current = None
        served_passengers_in_state = set()

        # Extract current lift position and served passengers
        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'lift-at':
                f_current = args[0]
            elif pred == 'served':
                 served_passengers_in_state.add(args[0])

        # 2. Identify unserved goal passengers.
        unserved_goal_passengers = self.goal_passengers - served_passengers_in_state

        # If this set is empty, the goal is reached, which should have been
        # caught by the initial goal_reached check.

        # 3. Separate unserved passengers into waiting and boarded based on state.
        current_waiting = set()
        current_boarded = set()
        origin_floors_of_waiting = {} # Store origin floor for easy lookup

        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'origin':
                p, f = args
                if p in unserved_goal_passengers:
                    current_waiting.add(p)
                    origin_floors_of_waiting[p] = f
            elif pred == 'boarded':
                p = args[0]
                if p in unserved_goal_passengers:
                    current_boarded.add(p)

        # 4. Calculate action cost.
        # Each waiting passenger needs a 'board' action.
        # Each boarded passenger needs a 'depart' action.
        h_action = len(current_waiting) + len(current_boarded)

        # 5. Identify required stop floors.
        F_stops = set()
        for p in current_waiting:
            # Add origin floor for waiting passengers
            # We rely on origin_floors_of_waiting having the floor
            F_stops.add(origin_floors_of_waiting[p])
        for p in current_boarded:
            # Add destination floor for boarded passengers
            # Destination is in static info
            # Assuming all goal passengers have a destination in static facts
            F_stops.add(self.destinations[p])

        # 6. & 7. Calculate movement cost.
        h_move = 0
        # Only calculate movement if there are floors that need servicing
        if F_stops:
            # Ensure current lift floor was found (should be if not goal)
            if f_current is not None and self.floor_levels: # Also check floor_levels is not empty
                current_l = self.floor_levels[f_current]
                min_stop_l = min(self.floor_levels[f] for f in F_stops)
                max_stop_l = max(self.floor_levels[f] for f in F_stops)

                # Movement heuristic: vertical distance spanning required floors
                # plus distance from current floor to the nearest end of the span.
                # This estimates moves to get to one end of the required range
                # and then traverse the range.
                h_move = (max_stop_l - min_stop_l) + min(abs(current_l - min_stop_l), abs(current_l - max_stop_l))

            # else: This case implies an issue with state representation (no lift-at)
            # or no floors defined, which shouldn't happen in valid problems.
            # h_move remains 0, which might be misleading but avoids crashing.


        # 8. Total heuristic.
        h = h_action + h_move

        return h
