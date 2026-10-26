# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
# In a real planning environment, this would be provided.
class Heuristic:
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing parentheses
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Return empty list for non-string or invalid format, consistent with match function's expectation
    return []

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to transport all
    passengers to their destinations. It counts the required board and depart
    actions for unserved passengers and adds an estimate of the lift movement
    cost based on the range of floors that need to be visited.

    # Assumptions
    - Each unserved passenger requires one board and one depart action.
    - The lift can carry all passengers simultaneously (no capacity limit).
    - The movement cost is estimated based on the distance to the lowest and
      highest required floors relative to the current lift position. This
      non-admissible estimate encourages the lift to move towards the floors
      at the extremes of the required stops.
    - Floors are arranged in a single vertical tower defined by 'above' predicates.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.
    - Parses the 'above' relations to determine the floor order from highest
      to lowest and create a mapping from floor names to numerical indices
      (highest floor is index 0).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Get the current state of the world.
    2. Check if the goal state (all passengers served) is reached. If yes, the heuristic is 0.
    3. Identify the current floor of the lift by finding the fact `(lift-at ?f)`.
    4. Determine the set of all passengers defined in the problem.
    5. Identify which passengers are currently served by checking for `(served ?p)` facts.
    6. Identify which passengers are currently boarded in the lift by checking for `(boarded ?p)` facts.
    7. Initialize a counter for unserved passengers and sets to store the floor indices that require a lift stop.
    8. Iterate through all passengers:
       - If a passenger is served, ignore them.
       - If a passenger is not served, increment the unserved passenger counter.
       - If the unserved passenger is boarded, add their destination floor's index to the set of required stops.
       - If the unserved passenger is not boarded, find their origin floor by checking for `(origin ?p ?f)` facts and add that origin floor's index to the set of required stops.
    9. Calculate the base cost: Multiply the number of unserved passengers by 2 (representing the minimum one board and one depart action needed per passenger).
    10. Calculate the movement cost:
        - If the set of required stop indices is empty, the movement cost is 0.
        - Otherwise, find the minimum and maximum index among the required stops.
        - Get the index of the current lift floor using the pre-calculated floor-to-index map.
        - The movement cost is estimated as the sum of the absolute difference between the current lift floor index and the minimum required floor index, and the absolute difference between the current lift floor index and the maximum required floor index. This simple sum encourages the lift to move towards the floors that represent the extremes of the remaining tasks.
    11. The total heuristic value is the sum of the base cost and the movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and
        the floor order from the static facts of the task.
        """
        super().__init__(task)

        # Extract passenger destinations from static facts
        self.destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor

        # Parse 'above' facts to determine floor order and indices
        above_relations = []
        all_floors = set()
        # Floors that are the first element in an 'above' pair (i.e., are above something)
        floors_above_something = set()
        # Floors that are the second element in an 'above' pair (i.e., something is above them)
        something_is_above_floors = set()

        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, f1, f2 = get_parts(fact)
                above_relations.append((f1, f2))
                all_floors.add(f1)
                all_floors.add(f2)
                floors_above_something.add(f1)
                something_is_above_floors.add(f2)

        # Find the highest floor (a floor that is not the second element in any 'above' pair)
        # Assumes a single tower structure where exactly one floor has nothing above it.
        highest_floor_candidates = list(all_floors - something_is_above_floors)
        if len(highest_floor_candidates) != 1:
             # This indicates an unexpected floor structure (e.g., multiple towers, isolated floors)
             # For this domain, we expect exactly one highest floor.
             # In a real scenario, add logging or error handling.
             # For this problem, we assume valid miconic structure and take the first candidate.
             pass
        highest_floor = highest_floor_candidates[0]


        # Build the ordered list of floors from highest to lowest
        self.ordered_floors = [highest_floor]
        current_floor = highest_floor
        while len(self.ordered_floors) < len(all_floors):
            next_floor = None
            for f1, f2 in above_relations:
                if f1 == current_floor:
                    next_floor = f2
                    break # Found the floor directly below current_floor
            if next_floor:
                self.ordered_floors.append(next_floor)
                current_floor = next_floor
            else:
                 # This implies a gap or issue in the 'above' chain before all floors are added.
                 # Should not happen in a well-formed miconic domain with a single tower.
                 # In a real scenario, add logging or error handling.
                 break


        # Create floor name to index mapping (highest floor is index 0)
        self.floor_to_idx = {floor: i for i, floor in enumerate(self.ordered_floors)}

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of facts)

        # Check if goal is reached
        # The goal is typically (and (served p1) (served p2) ...).
        # Checking if task.goals is a subset of state is the standard way.
        if self.goals <= state:
             return 0

        current_lift_floor = None
        num_unserved = 0
        waiting_origins_indices = set()
        boarded_destins_indices = set()

        # Find current lift location
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break # Assuming only one lift

        # Iterate through all passengers to find their status and required stops
        all_passengers = set(self.destinations.keys())
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}

        for passenger in all_passengers:
            if passenger in served_passengers:
                continue # This passenger is served

            num_unserved += 1

            if passenger in boarded_passengers:
                # Passenger is boarded, needs to be dropped off at destination
                dest_floor = self.destinations.get(passenger) # Use .get for safety
                if dest_floor and dest_floor in self.floor_to_idx:
                    boarded_destins_indices.add(self.floor_to_idx[dest_floor])
                # else: Passenger boarded but no destination or invalid destination floor?
                # This implies an inconsistent state. Assuming valid states for problem instances.
            else:
                # Passenger is waiting at origin, needs to be picked up
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        _, _, origin_floor = get_parts(fact)
                        break # Found the origin floor
                if origin_floor and origin_floor in self.floor_to_idx:
                     waiting_origins_indices.add(self.floor_to_idx[origin_floor])
                # else: Passenger not served, not boarded, but no origin? Inconsistent state.
                # Assuming valid states where unserved, unboarded passengers have an origin.


        # Calculate base cost (board + depart for each unserved passenger)
        # This is a lower bound on actions directly involving passengers.
        base_cost = num_unserved * 2

        # Calculate movement cost
        movement_cost = 0
        required_stop_indices = waiting_origins_indices | boarded_destins_indices

        if required_stop_indices:
            min_idx = min(required_stop_indices)
            max_idx = max(required_stop_indices)

            # Ensure current_lift_floor was found and is a valid floor
            if current_lift_floor and current_lift_floor in self.floor_to_idx:
                current_lift_floor_idx = self.floor_to_idx[current_lift_floor]

                # Estimate movement cost as distance to lowest + distance to highest required floor
                # This is a non-admissible estimate that encourages spanning the required floors.
                movement_cost = abs(current_lift_floor_idx - min_idx) + abs(current_lift_floor_idx - max_idx)
            # else: Lift location unknown or invalid floor? Inconsistent state.
            # Assuming valid states for problem instances.

        # Total heuristic is the sum of action costs and estimated movement costs
        total_cost = base_cost + movement_cost

        return total_cost
