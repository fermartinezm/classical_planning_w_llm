from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern (with wildcards).

    Args:
        fact (str): Fact string like "(predicate arg1 arg2)".
        args: Pattern components, e.g. "boarded", "*".

    Returns:
        bool: True if fact matches the pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    --------
    This heuristic estimates the number of required actions (lift moves, boardings, departures)
    to serve all remaining passengers starting from the current state. It provides a non-admissible,
    informed estimate designed to minimize node expansions in greedy best-first search.

    Assumptions:
    ------------
    - Each passenger must be boarded at their origin floor and departed at their destination floor.
    - Lift moves cost 1 action each.
    - Boarding or departing a passenger costs 1 action each.
    - The lift moves one floor per action, moving either up or down the building.
    - The "above" relations define a linear order of floors (can be used to compute floor distances).
    - Multiple passengers can be boarded before starting trips downward or upward.
    - Passengers already "served" require no further action.
    - We ignore the order of serving passengers to approximate cost efficiently.

    Heuristic Initialization:
    -------------------------
    - Extract floor ordering from static facts by interpreting the "above" predicate,
      building a mapping from each floor to a numeric floor level.
    - Extract destination floors for all passengers.
    - Extract origin floors for all passengers.
    - Store goal facts (which are all "served" passengers).

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify passengers not yet served.
    2. For each unserved passenger:
       - If not boarded: they must be boarded at their origin floor.
         This requires the lift to be at the origin floor, or move there.
       - If boarded: they must be departed at their destination floor.
         This requires the lift to move to the destination floor if not already there.
    3. Count the number of board actions and depart actions remaining.
    4. Estimate lift moves as the minimal distance needed to visit all required floors:
       - All origin floors for boarding passengers,
       - All destination floors for departing passengers,
       - Including the lift's current floor.
       Because exact minimal traveling route is costly, we use a heuristic bound:
         - Compute minimal floor, maximal floor among relevant floors,
         - Estimated lift moves is max_floor - min_floor (floor numbers) plus 
           minimal distance from current lift floor to either end.
    5. Sum boarding actions, departing actions, and estimated lift moves for total heuristic.
    6. Return 0 if all passengers are served (goal state).

    This heuristic balances computation efficiency and informed estimate by leveraging the 
    numeric floor layout and action counts for boarding and departing.
    """

    def __init__(self, task):
        """
        Initialize heuristic by reading static information:
        - Construct a map from floor name to numeric floor level.
        - Map each passenger to their origin and destination floors.

        Args:
            task (Task): Planning task object with static facts and goals.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract floors from static facts "above fX fY" meaning floorX < floorY
        # Use these to build floor levels (integers) for each floor for distance calculations.
        # Floors will be assigned numeric levels so that if (above f1 f2) then level[f1] < level[f2].
        # We'll do a topological sort-like assignment to determine floor levels.

        # Collect floors from static facts.
        floors = set()
        above_pairs = []
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                f1 = parts[1]
                f2 = parts[2]
                floors.update([f1, f2])
                above_pairs.append((f1, f2))
        floors = list(floors)

        # Build adjacency graph from above pairs: f1 -> f2 means f2 above f1
        # We'll assign level of f1 < level of f2
        adj = {floor: [] for floor in floors}
        indegree = {floor: 0 for floor in floors}
        for f1, f2 in above_pairs:
            adj[f1].append(f2)
            indegree[f2] += 1

        # Assign floor levels with a topological sorting. The lowest floors have level 0.
        floor_levels = {}
        queue = [f for f in floors if indegree[f] == 0]
        current_level = 0
        while queue:
            next_queue = []
            for f in queue:
                floor_levels[f] = current_level
            for f in queue:
                for neigh in adj[f]:
                    indegree[neigh] -= 1
                    if indegree[neigh] == 0:
                        next_queue.append(neigh)
            queue = next_queue
            current_level += 1

        # Save floor_levels mapping.
        self.floor_levels = floor_levels

        # Extract passenger origin and destination floors
        self.passenger_origins = {}
        self.passenger_destins = {}

        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger, dest_floor = parts[1], parts[2]
                self.passenger_destins[passenger] = dest_floor

        for fact in task.initial_state:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                passenger, origin_floor = parts[1], parts[2]
                self.passenger_origins[passenger] = origin_floor

    def __call__(self, node):
        """
        Estimate the number of required actions to serve all remaining passengers.

        Args:
            node (Node): Search node containing the current state.

        Returns:
            int: Estimated cost (number of actions) to reach goal.
        """
        state = node.state

        # Quick goal check: heuristic=0 if all served
        if self.goals <= state:
            return 0

        # Identify lift current floor
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break
        if lift_floor is None:
            # Should never happen if state is valid, but fallback:
            lift_floor = next(iter(self.floor_levels.keys()))

        lift_level = self.floor_levels[lift_floor]

        # Track passengers states
        boarded_passengers = set()
        served_passengers = set()
        waiting_passengers = set()  # Not boarded and not served

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "boarded":
                boarded_passengers.add(parts[1])
            elif predicate == "served":
                served_passengers.add(parts[1])
            elif predicate == "origin":
                # passenger not yet boarded
                waiting_passengers.add(parts[1])

        # Remaining passengers to board and to depart
        # Board: passengers not boarded and not served
        passengers_to_board = waiting_passengers - served_passengers
        # Depart: passengers who are boarded but not served
        passengers_to_depart = boarded_passengers - served_passengers

        # Number of board and depart actions remaining
        board_actions = len(passengers_to_board)
        depart_actions = len(passengers_to_depart)

        # Floors where lift needs to stop to board (origins) and to depart (destinations)
        origin_floors_to_visit = set(
            self.passenger_origins[p] for p in passengers_to_board if p in self.passenger_origins
        )
        dest_floors_to_visit = set(
            self.passenger_destins[p] for p in passengers_to_depart if p in self.passenger_destins
        )

        # Combine all relevant floors
        floors_to_visit = origin_floors_to_visit | dest_floors_to_visit

        # If no floors to visit (no passengers to board or depart), cost is just zero for actions
        if not floors_to_visit:
            # lift moves 0, actions = board_actions + depart_actions == 0
            return 0

        # Convert floors to levels for distance calculation
        floor_levels_to_visit = [self.floor_levels[f] for f in floors_to_visit]

        min_floor = min(floor_levels_to_visit)
        max_floor = max(floor_levels_to_visit)

        # Estimated minimal lift moves needed to cover all floors:
        # The building floors considered form an interval [min_floor, max_floor].
        # The lift must at least move from its current position to somewhere in [min_floor, max_floor]
        # plus cover distance max_floor - min_floor (to cover all floors in between)
        distance_to_min = abs(lift_level - min_floor)
        distance_to_max = abs(lift_level - max_floor)
        # Minimal lift moves to cover all floors:
        lift_moves = (max_floor - min_floor) + min(distance_to_min, distance_to_max)

        # Sum heuristic cost: lift moves + board actions + depart actions
        heuristic_cost = lift_moves + board_actions + depart_actions

        # Return integer heuristic
        return heuristic_cost
