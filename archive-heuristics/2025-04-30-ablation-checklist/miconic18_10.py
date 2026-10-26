from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check is just to zip and compare, fnmatch handles length differences implicitly
    # but explicitly checking length can prevent unexpected behavior if args is shorter than parts
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the necessary board and depart actions for unserved passengers
    and adds an estimate for the lift movement actions needed to reach the
    relevant floors.

    # Assumptions
    - Each unserved passenger needs one 'board' action and one 'depart' action.
    - Lift movement cost is estimated based on the number of distinct floors
      that need to be visited for pickups or dropoffs.

    # Heuristic Initialization
    - Extracts passenger destinations from static facts.
    - Determines the floor ordering from 'above' facts and creates a floor-to-index mapping.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify the current floor of the lift.
    2. Identify passengers who are waiting at their origin ('origin' predicate). These need a 'board' action.
    3. Identify passengers who are boarded ('boarded' predicate). These need a 'depart' action at their destination.
    4. Count the number of passengers needing pickup (waiting at origin). This is the number of 'board' actions needed.
    5. Count the number of passengers needing dropoff (currently boarded). This is the number of 'depart' actions needed.
    6. Determine the set of unique floors that need to be visited:
       - Origin floors for passengers needing pickup.
       - Destination floors for passengers needing dropoff.
    7. Estimate the number of lift movement actions ('up' or 'down'):
       - This is roughly the number of distinct floors that need visiting.
       - If the lift is already at one of these floors, one 'visit' is free, so subtract 1 from the count of distinct floors.
       - Ensure the move count is non-negative.
    8. The total heuristic value is the sum of the board actions, depart actions, and estimated movement actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal conditions (implicitly, all passengers served).
        - Static facts ('destin' and 'above' relationships).
        """
        self.goals = task.goals  # Goal conditions are used to identify passengers

        # Store goal destinations for each passenger from static facts
        self.destinations = {}
        # Store floor indices based on 'above' relationships
        self.floor_indices = {}

        above_pairs = []
        all_floors = set()

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "destin":
                # Fact is like (destin passenger floor)
                passenger, floor = parts[1:]
                self.destinations[passenger] = floor
            elif parts[0] == "above":
                # Fact is like (above floor_higher floor_lower)
                f_higher, f_lower = parts[1:]
                above_pairs.append((f_higher, f_lower))
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        # Determine floor order from 'above' facts
        # Build a map from lower floor to the floor immediately above it
        higher_than = {}
        # Track which floors appear as the 'higher' floor in an 'above' relation
        is_higher_floor = {f: False for f in all_floors}

        for f_higher, f_lower in above_pairs:
            higher_than[f_lower] = f_higher
            is_higher_floor[f_higher] = True

        # Find the lowest floor (the one that is never the 'higher' floor)
        lowest_floor = None
        for floor in all_floors:
            if not is_higher_floor[floor]:
                lowest_floor = floor
                break

        # Build the ordered list of floors from lowest to highest
        ordered_floors = []
        current_floor = lowest_floor
        while current_floor is not None:
            ordered_floors.append(current_floor)
            current_floor = higher_than.get(current_floor)

        # Create the floor name to index mapping
        self.floor_indices = {floor: i for i, floor in enumerate(ordered_floors)}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is already reached
        if self.goals <= state:
            return 0

        current_floor = None
        passengers_needing_pickup = 0
        passengers_needing_dropoff = 0
        pickup_floors = set()
        dropoff_floors = set()

        # Iterate through the state facts to find relevant information
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "lift-at":
                current_floor = parts[1]
            elif predicate == "origin":
                # (origin passenger floor)
                passenger, floor = parts[1:]
                # This passenger needs to be picked up
                passengers_needing_pickup += 1
                pickup_floors.add(floor)
            elif predicate == "boarded":
                # (boarded passenger)
                passenger = parts[1]
                # This passenger needs to be dropped off at their destination
                passengers_needing_dropoff += 1
                # Look up the destination floor using the pre-computed map
                if passenger in self.destinations:
                     dropoff_floors.add(self.destinations[passenger])
                # Note: If a passenger is boarded but not in self.destinations,
                # it implies they are not one of the passengers defined in the problem
                # with a specific destination goal, which shouldn't happen in valid PDDL.
                # We proceed assuming valid states/tasks.

        # Combine floors that need service (pickup or dropoff)
        floors_to_visit = pickup_floors | dropoff_floors

        # Estimate movement cost
        num_moves = 0
        if floors_to_visit:
            # The lift needs to visit each distinct floor in floors_to_visit.
            # A simple estimate is the number of such floors.
            num_moves = len(floors_to_visit)
            # If the lift is already at one of the required floors, the first 'stop' is free.
            if current_floor in floors_to_visit:
                num_moves -= 1
            # Ensure move count is not negative (can happen if only 1 floor to visit and lift is there)
            num_moves = max(0, num_moves)

            # Alternative move estimate based on floor range (potentially more informative but complex):
            # if self.floor_indices: # Ensure floor indices were successfully built
            #     service_indices = {self.floor_indices[f] for f in floors_to_visit if f in self.floor_indices}
            #     if service_indices:
            #         min_idx = min(service_indices)
            #         max_idx = max(service_indices)
            #         current_floor_idx = self.floor_indices.get(current_floor, -1) # Use -1 or handle if current_floor not found

            #         if current_floor_idx != -1: # Only calculate if current floor index is valid
            #             # Estimate moves to cover the range [min_idx, max_idx] starting from current_floor_idx
            #             # This is the distance to the closer end of the range plus the range size
            #             dist_to_min = abs(current_floor_idx - min_idx)
            #             dist_to_max = abs(current_floor_idx - max_idx)
            #             range_size = max_idx - min_idx
            #             num_moves = min(dist_to_min, dist_to_max) + range_size
            #         else: # Fallback if current floor index is invalid
            #              num_moves = len(floors_to_visit) # Simple count as fallback


        # Total heuristic estimate: board actions + depart actions + movement actions
        total_cost = passengers_needing_pickup + passengers_needing_dropoff + num_moves

        return total_cost

