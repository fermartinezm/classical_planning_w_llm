from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    Example: '(origin p1 f6)' -> ['origin', 'p1', 'f6']
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Summary
    -------
    This heuristic estimates the number of required actions to serve all passengers,
    guiding the greedy best-first search in the Miconic elevator domain.

    Assumptions
    -----------
    - Each passenger must be boarded at their origin floor and departed at their destination floor.
    - The elevator moves between floors, boards passengers currently on the elevator's floor,
      and departs passengers when the elevator is at their destination floor.
    - The predicates `boarded` and `served` indicate if a passenger is on the elevator or has been served.
    - The elevator starts at some floor identified by `(lift-at ?floor)`.
    - The spatial relation `(above f1 f2)` means f2 is directly above f1 (helping to compute floor order).
    - The heuristic need not be admissible but should be efficiently computable.

    Heuristic Initialization
    ------------------------
    - Extract from static facts:
      * the floors and their partial ordering via `above` predicates.
      * for each passenger, their origin and destination floors.
    - Build a sorted list of floors (bottom to top) using the `above` relations.
    - This floor order enables the heuristic to compute shortest move distances between floors.
    - Store goal condition: all passengers should be `served`.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all passengers are served (goal), heuristic is 0.
    2. Identify the current floor of the elevator (`lift-at`).
    3. For each passenger:
       - If served already, no cost.
       - If boarded, estimate: distance from elevator floor to passenger's destination floor + 1 depart action.
       - If not boarded, estimate: 
         (distance from elevator floor to passenger's origin floor) + 1 board action +
         (distance from origin floor to destination floor) + 1 depart action.
    4. Sum over all passengers the estimated minimal actions to serve them.
    5. To avoid gross overestimation, instead of summing all moves individually,
       consider the maximal elevator moves required to visit all relevant floors in an optimal order:
       - Compute all floors where actions must occur (boarding and/or departing).
       - Compute minimal elevator travel cost to cover these floors starting from current floor:
         * This is approximated by the span between lowest and highest required floor relative to current floor.
       - Count 1 action per passenger to board plus 1 to depart.
    6. Return sum of:
       - total boarding and departing actions = 2 * (number of unserved passengers not served yet)
       - plus minimal elevator moves to cover all floors that require action.
    This heuristic balances accuracy and efficiency, accounting for elevator travel cost and passenger actions.

    Note: The heuristic never returns infinite and is zero iff the goal is reached.
    """

    def __init__(self, task):
        # Extract floor order from static facts using 'above' relations
        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.floors = set()
        self.above = {}  # floor -> set of floors above it

        # Parse static facts
        for fact in task.static:
            pred, *args = get_parts(fact)
            if pred == "above":
                floor1, floor2 = args
                self.floors.add(floor1)
                self.floors.add(floor2)
                self.above.setdefault(floor1, set()).add(floor2)
            elif pred == "destin":
                passenger, floor = args
                self.passenger_destinations[passenger] = floor
                self.floors.add(floor)
            elif pred == "origin":
                passenger, floor = args
                self.passenger_origins[passenger] = floor
                self.floors.add(floor)

        # Build total floor order from 'above' transitive closure
        # We find a topological order consistent with above relations:
        # floor f1 < f2 if (above f1 f2) i.e. f2 is above f1 (higher floor).
        self.floor_indices = self._compute_floor_indices()

        # Store passengers list
        self.passengers = set(self.passenger_origins.keys())

        # Save goal facts for fast check
        self.goal_served_passengers = set()
        for goal in task.goals:
            pred, *args = get_parts(goal)
            if pred == "served":
                self.goal_served_passengers.add(args[0])

    def _compute_floor_indices(self):
        """
        Compute an ordering index mapping for floors consistent with the above relations.
        Lower index means lower floor.
        The 'above' relation means: above(f1,f2) => floor2 is above floor1,
        so index(f1) < index(f2).
        """
        floors = list(self.floors)
        above = self.above

        # Build predecessors count for topological sorting
        preds_count = {f: 0 for f in floors}
        for f1 in floors:
            for f2 in above.get(f1, []):
                preds_count[f2] += 1

        order = []
        zero_preds = [f for f in floors if preds_count[f] == 0]

        while zero_preds:
            f = zero_preds.pop()
            order.append(f)
            for successor in above.get(f, []):
                preds_count[successor] -= 1
                if preds_count[successor] == 0:
                    zero_preds.append(successor)

        if len(order) != len(floors):
            # Cycle detected or incomplete order, fallback to sorted order by name
            order = sorted(floors)

        # Map each floor to its index in order (lower floors have smaller index)
        return {floor: idx for idx, floor in enumerate(order)}

    def _floor_distance(self, f1, f2):
        """
        Returns the absolute distance between two floors using their indices.
        Used as elevator moves count between floors.
        """
        return abs(self.floor_indices[f1] - self.floor_indices[f2])

    def __call__(self, node):
        state = node.state

        # Goal check: if all passengers that must be served are served, heuristic=0
        served_passengers = {get_parts(fact)[1] for fact in state if fact.startswith("(served ")}
        if self.goal_served_passengers <= served_passengers:
            return 0

        # Identify elevator current floor
        lift_floor = None
        for fact in state:
            if fact.startswith("(lift-at "):
                lift_floor = get_parts(fact)[1]
                break
        if lift_floor is None:
            # Should not happen in valid states, but fallback
            lift_floor = next(iter(self.floor_indices))

        # Track passengers who are boarded and served
        boarded_passengers = {get_parts(fact)[1] for fact in state if fact.startswith("(boarded ")}

        # Determine floors to visit and count passenger actions
        floors_to_visit = set()
        boarding_count = 0
        departing_count = 0

        # For every passenger that is not served yet
        for p in self.passengers:
            if p in served_passengers:
                continue  # no cost for served passengers

            origin = self.passenger_origins[p]
            destination = self.passenger_destinations[p]

            if p in boarded_passengers:
                # Passenger onboard, need only to go to destination floor and depart
                floors_to_visit.add(destination)
                departing_count += 1
            else:
                # Passenger not onboard, need to go to origin to board then to destination to depart
                floors_to_visit.add(origin)
                floors_to_visit.add(destination)
                boarding_count += 1
                departing_count += 1

        # If no floors to visit (all passengers served or no passengers), heuristic is 0
        if not floors_to_visit:
            return 0

        # Compute min cost to cover all required floors starting from lift_floor.
        # We approximate by calculating the distance needed to cover the range between
        # the min and max floor indices that include the current lift floor.
        floor_indices = [self.floor_indices[f] for f in floors_to_visit]

        min_floor_idx = min(floor_indices)
        max_floor_idx = max(floor_indices)
        current_floor_idx = self.floor_indices[lift_floor]

        # Minimal elevator moves to cover all floors with passengers boarding/departing:
        # It covers the minimal travel from current floor to cover all floors between min and max.
        # Cases:
        # 1. Current floor inside [min, max]: travel distance = max - min
        # 2. Current floor below min: travel distance = max - current_floor
        # 3. Current floor above max: travel distance = current_floor - min
        if current_floor_idx < min_floor_idx:
            elevator_moves = max_floor_idx - current_floor_idx
        elif current_floor_idx > max_floor_idx:
            elevator_moves = current_floor_idx - min_floor_idx
        else:
            elevator_moves = max_floor_idx - min_floor_idx

        # Each elevator move corresponds to one action (up or down).
        # Boarding and departing each count as one action per passenger.
        total_cost = elevator_moves + boarding_count + departing_count

        return total_cost
