# Need to import Heuristic base class. Assuming it's available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If running this code standalone for testing, you might need a dummy Heuristic class:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch
import math # Although abs() is built-in, importing math is common practice.

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, though in a planner context, facts should be strings like '(predicate arg1 arg2)'
        # print(f"Warning: Unexpected fact format: {fact}") # Debugging
        return [] # Return empty list for safety
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The fact must have at least as many parts as the pattern has arguments.
    if len(parts) < len(args):
        return False
    # Check if each part matches the corresponding argument pattern.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def floor_name_to_level(floor_name):
    """Converts floor name like 'fN' to integer level N."""
    # This function relies on the domain-specific naming convention
    # where floors are named 'f' followed by an integer (e.g., f1, f2, f3).
    # This is a common convention in Miconic benchmarks.
    # A more robust approach would parse the 'above' facts to build the floor order,
    # but this simple approach is efficient and sufficient for typical instances.
    try:
        # Extract the numerical part of the floor name (after 'f') and convert to integer.
        # Assumes floor names are like 'f1', 'f10', etc.
        level_str = floor_name[1:]
        return int(level_str)
    except (ValueError, IndexError):
        # If the floor name format is unexpected (e.g., not starting with 'f' or not followed by a number),
        # this indicates an issue with the problem definition or state representation.
        # Returning a very large number will make states involving such floors have a high heuristic cost,
        # effectively discouraging the search from exploring them.
        # print(f"Warning: Unexpected floor name format encountered: {floor_name}") # Debugging
        return 1_000_000 # Return a large value to indicate an issue


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    This heuristic estimates the remaining number of actions required to reach
    the goal state (all passengers served). It is designed for use with a
    greedy best-first search and does not need to be admissible.

    The heuristic sums an estimated cost for each passenger that still needs
    to be served according to the goal conditions.

    For a passenger currently waiting at their origin floor:
    Estimated cost = 2 (for 'board' and 'depart' actions)
                   + movement cost from the lift's current floor to the passenger's origin floor
                   + movement cost from the passenger's origin floor to their destination floor.

    For a passenger currently boarded in the lift:
    Estimated cost = 1 (for the 'depart' action)
                   + movement cost from the lift's current floor to the passenger's destination floor.

    Movement cost between floors is estimated as the absolute difference in their floor levels,
    derived from the floor names (assuming 'fN' convention).

    This heuristic is non-admissible because it sums movement costs independently for each
    passenger, whereas in reality, lift movements can serve multiple passengers simultaneously.
    However, it provides a reasonable estimate that guides the search towards states where
    passengers are closer to being served.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information from the task.

        Specifically, it extracts the destination floor for each passenger
        from the static 'destin' facts.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are facts that are not affected by actions.
        self.static_facts = task.static

        # Store passenger destinations: {passenger_name: destination_floor_name}
        # This information is static and needed to calculate movement costs.
        self.passenger_destinations = {}
        for fact in self.static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact)
                self.passenger_destinations[passenger] = destination_floor

        # Note: Floor levels are determined dynamically in __call__ based on naming convention.
        # If the naming convention ('fN') is not guaranteed across all instances,
        # a more robust approach would involve parsing the 'above' facts in __init__
        # to build a complete floor ordering and level map. For typical Miconic benchmarks,
        # the 'fN' convention is standard.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.

        This function is called by the search algorithm for each state (node).
        It calculates the heuristic value for the given state.
        """
        state = node.state # Current world state (represented as a frozenset of fact strings).

        # If the current state satisfies all goal conditions, the heuristic cost is 0.
        if self.goals <= state:
            return 0

        # Find the current floor of the lift. There should be exactly one 'lift-at' fact.
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        # If the lift's location is not found in the state facts (which indicates an invalid state
        # according to the domain model), return a very large value to represent an effectively
        # infinite cost or unsolvable state from this point.
        if current_lift_floor is None:
             # print("Warning: No 'lift-at' fact found in state. Returning infinity.") # Debugging
             return float('inf') # Represents a very high cost/unsolvable state

        # Convert the current lift floor name to its numerical level.
        current_lift_level = floor_name_to_level(current_lift_floor)

        # Build a map of each passenger's current status and location (if waiting).
        # Status can be 'served', 'boarded', or 'waiting'.
        # We only care about passengers mentioned in the state facts.
        passenger_status = {} # {passenger: 'served' or 'boarded' or 'waiting'}
        passenger_origin_map = {} # {passenger: origin_floor} # Only for 'waiting' passengers

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'served' and len(parts) == 2:
                passenger_status[parts[1]] = 'served'
            elif predicate == 'boarded' and len(parts) == 2:
                passenger_status[parts[1]] = 'boarded'
            elif predicate == 'origin' and len(parts) == 3:
                passenger = parts[1]
                origin_floor = parts[2]
                passenger_status[passenger] = 'waiting'
                passenger_origin_map[passenger] = origin_floor
            # Facts like 'lift-at', 'destin', 'above' are not passenger status/location facts.

        total_cost = 0

        # Identify all passengers that need to be served according to the goal state.
        passengers_to_serve = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 _, passenger = get_parts(goal)
                 passengers_to_serve.add(passenger)

        # Calculate the estimated cost for each passenger who is required to be served
        # but is not yet in the 'served' state.
        for passenger in passengers_to_serve:
            status = passenger_status.get(passenger) # Get the passenger's status from the current state

            # If the passenger is already served, they contribute 0 to the heuristic cost.
            if status == 'served':
                continue

            # If the passenger is not served, they must be either waiting or boarded
            # in any valid reachable state from a standard Miconic initial state.

            # Get the passenger's destination floor from the pre-calculated map.
            destin_floor = self.passenger_destinations.get(passenger)
            # If a passenger needs serving but has no destination defined (indicates an invalid problem instance),
            # we cannot calculate their cost. Skip this passenger.
            if destin_floor is None:
                 # print(f"Warning: Passenger {passenger} needs serving but has no destination defined in static facts. Skipping.") # Debugging
                 continue

            # Convert the destination floor name to its numerical level.
            destin_level = floor_name_to_level(destin_floor)

            if status == 'boarded':
                # This passenger is currently inside the lift.
                # They need to be transported to their destination and then departed.
                # Estimated cost:
                # 1 (for the 'depart' action)
                # + movement cost from the lift's current floor to the destination floor.
                cost_for_passenger = 1 # Cost of the 'depart' action
                cost_for_passenger += abs(current_lift_level - destin_level) # Estimated movement cost
                total_cost += cost_for_passenger

            elif status == 'waiting':
                # This passenger is waiting at their origin floor.
                # They need to be boarded, transported to their destination, and then departed.
                origin_floor = passenger_origin_map.get(passenger)
                # If a passenger's status is 'waiting' but there is no 'origin' fact for them
                # in the state (indicates an inconsistent state), skip this passenger.
                if origin_floor is None:
                    # print(f"Warning: Passenger {passenger} status is 'waiting' but no 'origin' fact found in state. Skipping.") # Debugging
                    continue

                # Convert the origin floor name to its numerical level.
                origin_level = floor_name_to_level(origin_floor)

                # Estimated cost:
                # 2 actions (1 for 'board' + 1 for 'depart')
                # + movement cost from the lift's current floor to the origin floor
                # + movement cost from the origin floor to the destination floor.
                cost_for_passenger = 2 # Cost of 'board' + 'depart' actions
                cost_for_passenger += abs(current_lift_level - origin_level) # Estimated movement cost to origin
                cost_for_passenger += abs(origin_level - destin_level) # Estimated movement cost from origin to destin
                total_cost += cost_for_passenger

            # If status is None, it means the passenger is not in the state as served, boarded, or waiting.
            # In a valid reachable state from a standard Miconic initial state, any unserved
            # passenger must be either waiting or boarded. This case should ideally not be reached.
            # If it is reached, it might indicate an unsolvable state or a malformed problem/state.
            # We ignore such passengers, relying on the search to handle dead ends.
            pass

        # The total_cost calculated here will be 0 only if the loop over passengers_to_serve
        # finds that all of them have status 'served', which is exactly the goal condition.
        # If the goal is not met, total_cost will be > 0 (assuming valid state and problem).

        return total_cost

