# Need to import the base class
# from heuristics.heuristic_base import Heuristic # Assuming this path is correct in the target environment

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is like "(predicate arg1 arg2 ...)"
    # Remove outer parentheses and split by space
    return fact[1:-1].split()

# Assuming Heuristic base class is defined elsewhere and imported
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

# Inherit from Heuristic if available in the target environment
# class miconicHeuristic(Heuristic):
class miconicHeuristic:
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers by summing the estimated minimum actions for each unserved
    passenger independently. The estimate for a single passenger includes
    the movement of the lift to their origin (if not boarded), the board action,
    the movement to their destination, and the depart action.

    # Assumptions
    - Floors are arranged in a linear order defined by the 'above' predicate,
      where (above f_higher f_lower) means f_higher is immediately above f_lower.
    - All actions (move, board, depart) have a cost of 1.
    - The movement cost between two floors is the absolute difference in their
      floor indices.
    - The heuristic sums the costs for each passenger independently,
      overestimating movement costs when multiple passengers share lift travel.
      This is acceptable for a greedy best-first search heuristic aiming to
      minimize expanded nodes.

    # Heuristic Initialization
    - Parses the static facts to build a mapping from floor names to numerical
      indices based on the 'above' relationships.
    - Parses the static facts to store the origin and destination floors for
      each passenger.
    - Identifies the set of all passengers.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. Initialize the total heuristic cost to 0.
    3. For each passenger known from the problem definition:
       a. Check if the passenger is already 'served' in the current state. If yes,
          this passenger contributes 0 to the heuristic.
       b. If the passenger is not served:
          i. Get the passenger's origin and destination floors using the maps
             built during initialization.
          ii. Check if the passenger is currently 'boarded'.
          iii. If the passenger is 'boarded':
              - Estimate the cost to serve them:
                - Movement cost: Absolute difference between the current lift
                  floor index and the passenger's destination floor index.
                - Action cost: 1 (for the 'depart' action).
              - Add this estimated cost to the total heuristic.
          iv. If the passenger is not 'boarded' (meaning they are at their origin):
              - Estimate the cost to serve them:
                - Movement cost 1: Absolute difference between the current lift
                  floor index and the passenger's origin floor index.
                - Action cost 1: 1 (for the 'board' action).
                - Movement cost 2: Absolute difference between the passenger's
                  origin floor index and their destination floor index.
                - Action cost 2: 1 (for the 'depart' action).
              - Add the sum of Movement cost 1 + Action cost 1 + Movement cost 2 +
                Action cost 2 to the total heuristic.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger
        origin/destination information from the task's static facts.
        """
        # Assuming task object has 'goals' and 'static' attributes
        self.goals = task.goals  # Goal conditions (used to check if served)
        static_facts = task.static  # Facts that are not affected by actions.

        self.origin_map = {}  # passenger -> origin_floor
        self.destin_map = {}  # passenger -> destin_floor
        above_pairs = []      # list of (f_higher, f_lower) tuples
        all_floors = set()    # set of all floor names

        # Parse static facts to populate maps and collect floor info
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'origin':
                # Fact is (origin passenger floor)
                self.origin_map[parts[1]] = parts[2]
            elif parts[0] == 'destin':
                # Fact is (destin passenger floor)
                self.destin_map[parts[1]] = parts[2]
            elif parts[0] == 'above':
                # Fact is (above f_higher f_lower)
                above_pairs.append((parts[1], parts[2]))
                all_floors.add(parts[1])
                all_floors.add(parts[2])

        # Build floor index map based on 'above' relationships
        self.floor_indices = {}
        # Map f_lower to f_higher (the floor immediately above it)
        above_map = {f_lower: f_higher for f_higher, f_lower in above_pairs}

        # Find the lowest floor (the one that is never the 'higher' floor in any 'above' pair)
        set_of_higher_floors = {f_higher for f_higher, f_lower in above_pairs}
        lowest_floor = None
        for f in all_floors:
            if f not in set_of_higher_floors:
                lowest_floor = f
                break

        # Build the ordered floor list and index map starting from the lowest floor
        current = lowest_floor
        index = 0
        while current is not None:
            self.floor_indices[current] = index
            index += 1
            # Get the floor immediately above current using the map
            current = above_map.get(current)

        # Basic check: Ensure all floors were indexed
        if len(self.floor_indices) != len(all_floors):
             # This case should ideally not happen in valid miconic problems
             # with a linear floor structure defined by 'above'.
             # print(f"Warning: Could not build linear floor index for all floors. Indexed {len(self.floor_indices)} out of {len(all_floors)}")
             pass # Proceed assuming the indexed floors are sufficient for relevant objects

        # Store the set of all passengers for easy iteration in __call__
        self.all_passengers = set(self.origin_map.keys()) | set(self.destin_map.keys())


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state (all passengers served).
        """
        state = node.state  # Current world state (frozenset of facts)

        total_heuristic = 0  # Initialize action cost counter.

        # Find the current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        # If lift location is unknown (should not happen in valid states), return infinity or a large value
        if current_lift_floor is None:
             # This state is likely invalid or represents a failure
             return float('inf') # Or a large number if infinity is not desired/handled

        current_lift_idx = self.floor_indices[current_lift_floor]

        # Iterate through all passengers defined in the problem
        for passenger in self.all_passengers:
            # Check if this passenger is already served
            if f"(served {passenger})" in state:
                continue # Passenger is served, no more cost for them

            # Passenger is unserved. Calculate their contribution to the heuristic.
            # Get origin and destination floors. These must exist for unserved passengers.
            origin_floor = self.origin_map[passenger]
            destin_floor = self.destin_map[passenger]
            origin_idx = self.floor_indices[origin_floor]
            destin_idx = self.floor_indices[destin_floor]

            # Check if the passenger is currently boarded
            is_boarded = f"(boarded {passenger})" in state

            if is_boarded:
                # Passenger is boarded, needs to be dropped off at destination
                # Cost = movement to destination + depart action
                movement_cost = abs(current_lift_idx - destin_idx)
                action_cost = 1 # depart
                total_heuristic += movement_cost + action_cost
            else:
                # Passenger is not boarded, needs to be picked up at origin and dropped off at destination
                # Cost = movement to origin + board action + movement to destination + depart action
                movement_cost = abs(current_lift_idx - origin_idx) + abs(origin_idx - destin_idx)
                action_cost = 2 # board + depart
                total_heuristic += movement_cost + action_cost

        return total_heuristic
