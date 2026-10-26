from heuristics.heuristic_base import Heuristic
import math
from collections import deque

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the miconic domain.

    Summary:
    This heuristic estimates the remaining cost to reach the goal by summing
    the estimated costs for each unserved passenger independently. For an
    unboarded passenger, the cost includes travel from the current lift
    location to their origin, the board action, travel from their origin
    to their destination, and the depart action. For a boarded passenger,
    the cost includes travel from the current lift location to their
    destination and the depart action. Travel cost between floors is
    estimated as the absolute difference in floor levels.

    Assumptions:
    - The PDDL instance is valid:
        - All floors mentioned in the initial state or static facts exist.
        - The 'above' predicate defines a directed acyclic graph (DAG) on floors,
          allowing assignment of floor levels.
        - Every passenger mentioned in the state has a corresponding 'destin'
          fact in the static information.
        - The 'lift-at' predicate is present in the state.
        - Unserved passengers are either at their origin (if unboarded) or boarded.
          (i.e., a passenger is not unserved, unboarded, and not at their origin).

    Heuristic Initialization:
    In the constructor (`__init__`), the heuristic pre-processes the static
    information from the task:
    1. It identifies all floors mentioned in the static facts ('above', 'destin')
       and the initial state ('lift-at', 'origin').
    2. It parses the 'above' facts to build a graph representing the floor
       hierarchy, where an edge exists from a lower floor to a floor
       immediately above it.
    3. It performs a Breadth-First Search (BFS) starting from the lowest
       floor(s) (those with nothing below them according to 'above' facts)
       to assign a numerical 'level' to each floor. The level of a floor is
       1 + the maximum level of any floor immediately below it.
    4. It parses the 'destin' facts to create a mapping from each passenger
       to their destination floor.
    This pre-processing is done once and the results (`floor_levels`,
    `passenger_destinations`) are stored for efficient lookup during heuristic
    computation.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the current floor of the lift by finding the `(lift-at ?f)` fact.
       If the lift floor is not found or its level is unknown (e.g., due to
       disconnected floor graph), return infinity.
    2. Initialize the total heuristic value to 0.
    3. Identify all passengers that are not yet served by checking for the
       `(served ?p)` fact.
    4. For each unserved passenger `p`:
       a. Get their destination floor `d` from the precomputed `passenger_destinations`.
          If the destination is unknown or its level is unknown, return infinity
          (indicating an unsolvable state).
       b. Check if the passenger `p` is currently `(boarded ?p)`.
          - If yes: The passenger needs to be transported from the current lift
            location to their destination `d` and then depart. The estimated
            cost for this passenger is the absolute difference between the
            current lift floor level and the destination floor level (for movement)
            plus 1 (for the depart action). Add this cost to the total heuristic.
       c. Check if the passenger `p` is currently at their origin `o` by finding
          the `(origin ?p ?o)` fact.
          - If yes: The passenger needs to be picked up at `o`, transported to
            `d`, and then depart. The estimated cost for this passenger is the
            absolute difference between the current lift floor level and the
            origin floor level (moves to origin) plus 1 (board action) plus
            the absolute difference between the origin floor level and the
            destination floor level (moves from origin to destination) plus 1
            (depart action). Add this cost to the total heuristic. If the origin
            floor level is unknown, return infinity.
       d. If the passenger is unserved but neither boarded nor at their origin,
          this state is considered unreachable from a valid initial state via
          valid actions. Return infinity.
    5. The total heuristic value is the sum of the estimated costs for all
       unserved passengers. This value is 0 if and only if all passengers are
       served (goal state).
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.floor_levels = {}
        self.passenger_destinations = {}
        self._process_static_info()

    def _parse_fact(self, fact_string):
        """Helper to parse fact string '(pred arg1 arg2 ...)'"""
        # Remove surrounding brackets and split by space
        cleaned_string = fact_string.strip().replace('(', '').replace(')', '')
        parts = cleaned_string.split()
        if not parts:
            return None, [] # Handle empty string case
        return parts[0], parts[1:]

    def _process_static_info(self):
        """Build floor levels and passenger destinations from static facts."""
        all_floors = set()
        above_pairs = [] # List of (f_above, f_below)

        # Collect information from static facts
        for fact_string in self.task.static:
            pred, args = self._parse_fact(fact_string)
            if pred == 'above':
                if len(args) == 2:
                    f_above, f_below = args
                    all_floors.add(f_above)
                    all_floors.add(f_below)
                    above_pairs.append((f_above, f_below))
            elif pred == 'destin':
                if len(args) == 2:
                    p, f = args
                    self.passenger_destinations[p] = f
                    all_floors.add(f)

        # Add floors from initial state if not in static (e.g., lift-at, origin)
        for fact_string in self.task.initial_state:
             pred, args = self._parse_fact(fact_string)
             if pred in ['lift-at', 'origin']:
                 if args: # Ensure args is not empty
                     all_floors.add(args[-1]) # The last arg is the floor

        # If no floors found at all, nothing to do
        if not all_floors:
             return

        # Build graph where edge f_below -> f_above if (above f_above f_below)
        # This graph represents the 'is immediately below' relation.
        # We want levels such that level(f_above) = level(f_below) + 1.
        # We need the reverse graph: f_above -> f_below if (above f_above f_below)
        # In-degree in this reverse graph gives number of floors immediately below.
        # Lowest floors have in-degree 0 in the reverse graph.

        above_to_below_graph = {f: set() for f in all_floors}
        in_degree_reverse = {f: 0 for f in all_floors} # In-degree in the reverse graph

        for f_above, f_below in above_pairs:
            above_to_below_graph[f_above].add(f_below)
            in_degree_reverse[f_below] += 1

        # Find nodes with in-degree 0 in the reverse graph (lowest floors)
        q = deque()
        self.floor_levels = {f: float('inf') for f in all_floors}

        lowest_floors = [f for f in all_floors if in_degree_reverse[f] == 0]

        # If there are floors but no identified lowest floors (e.g., cycle or disconnected)
        # leave levels as inf. Heuristic will return inf if these floors are relevant.
        if not lowest_floors and all_floors:
             # This indicates an invalid 'above' structure (e.g., cycle)
             # Leaving levels as inf is a safe way to indicate unsolvability via heuristic.
             return

        # Initialize queue with lowest floors (level 0)
        for f_low in lowest_floors:
            self.floor_levels[f_low] = 0
            q.append(f_low)

        # Build the graph for BFS traversal: f_below -> f_above
        below_to_above_graph = {f: set() for f in all_floors}
        for f_above, f_below in above_pairs:
             below_to_above_graph[f_below].add(f_above)

        # BFS to propagate levels upwards
        # We use visited set to handle potential graph irregularities safely,
        # although PDDL 'above' should define a DAG.
        visited = set()

        while q:
            f_below = q.popleft()

            if f_below in visited:
                 continue
            visited.add(f_below)

            current_level = self.floor_levels[f_below]

            # Find floors directly above f_below
            for f_above in below_to_above_graph.get(f_below, set()):
                # Update level if we found a path giving a higher level (correct for max level)
                # This handles branching correctly (level = 1 + max level below)
                if current_level + 1 > self.floor_levels[f_above]:
                     self.floor_levels[f_above] = current_level + 1
                     # Re-queue if level increased (needed for max propagation in non-tree DAGs)
                     q.append(f_above)


    def __call__(self, node):
        state = node.state
        h_value = 0
        current_lift_floor = None

        # Find current lift floor
        for fact_string in state:
            pred, args = self._parse_fact(fact_string)
            if pred == 'lift-at' and len(args) == 1:
                current_lift_floor = args[0]
                break

        # If lift-at is not found, or its floor is not in our map, state is likely invalid/unsolvable
        current_lift_level = self.floor_levels.get(current_lift_floor, float('inf'))
        if current_lift_level == float('inf'):
             return float('inf')

        served_passengers = set()
        boarded_passengers = set()
        origin_passengers = {} # p -> floor

        # Collect passenger states
        for fact_string in state:
            pred, args = self._parse_fact(fact_string)
            if pred == 'served' and len(args) == 1:
                served_passengers.add(args[0])
            elif pred == 'boarded' and len(args) == 1:
                boarded_passengers.add(args[0])
            elif pred == 'origin' and len(args) == 2:
                p, f = args
                origin_passengers[p] = f

        # Iterate through all passengers known from static info (those with destinations)
        all_passengers_with_dest = set(self.passenger_destinations.keys())

        for p in all_passengers_with_dest:
            if p in served_passengers:
                continue # Passenger is served, cost is 0

            # Passenger is unserved. Calculate their contribution to the heuristic.

            dest_floor = self.passenger_destinations.get(p)
            # If passenger has no destination in static info, instance is malformed.
            # Or if destination floor level is unknown (disconnected graph).
            dest_level = self.floor_levels.get(dest_floor, float('inf'))
            if dest_level == float('inf'):
                 return float('inf') # Unsolvable from this state

            if p in boarded_passengers:
                # Passenger is boarded, needs to depart at destination
                # Cost = moves from current lift to destination + 1 (depart)
                h_value += abs(current_lift_level - dest_level) + 1
            elif p in origin_passengers.keys():
                # Passenger is unboarded at origin, needs board and depart
                origin_floor = origin_passengers[p]
                origin_level = self.floor_levels.get(origin_floor, float('inf'))

                # If origin floor level is unknown, state is likely invalid/unsolvable
                if origin_level == float('inf'):
                    return float('inf')

                # Cost = moves from current lift to origin + 1 (board) + moves from origin to destination + 1 (depart)
                h_value += abs(current_lift_level - origin_level) + 1 + abs(origin_level - dest_level) + 1
            else:
                 # Passenger is unserved but neither boarded nor at their origin.
                 # This state is likely a dead end or invalid according to domain rules.
                 return float('inf') # Unsolvable from this state

        # The heuristic is 0 if and only if all passengers in self.passenger_destinations
        # are in served_passengers. This is the goal condition.

        return h_value
