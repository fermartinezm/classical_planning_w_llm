from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    # Use zip to handle cases where parts might be longer than args (e.g., wildcard *)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the number of actions (board, depart, move) required to serve
    all passengers.

    Heuristic components:
    1. Number of passengers waiting at their origin floor (need boarding).
    2. Number of passengers currently boarded (need departing).
    3. Estimated movement cost: The vertical distance the lift must travel
       to cover all floors where actions (board/depart) are needed for
       unserved passengers, starting from the current floor.

    This heuristic is not admissible as it might overestimate the movement
    cost or count actions multiple times if a single stop serves multiple
    passengers. It aims to guide the search efficiently by prioritizing
    states where more required actions can potentially be achieved soon
    or where the required service points are clustered.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Floor ordering mapping (floor name -> index).
        - Passenger destinations mapping (passenger name -> destination floor).
        - Identify all passengers involved in the goal.
        """
        self.goals = task.goals  # Goal conditions, used to identify all passengers and served status.
        static_facts = task.static # Static facts like above and destin.

        # 1. Build floor ordering mapping (floor name -> index)
        self.floor_to_index = {}
        above_map_correct = {} # Maps floor2 -> floor1 if floor1 is immediately above floor2
        all_floors = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f1, f2 = get_parts(fact)[1:3]
                above_map_correct[f2] = f1 # f1 is above f2
                all_floors.add(f1)
                all_floors.add(f2)

        if not all_floors:
             # Handle case with no floors (shouldn't happen in valid problems)
             self.floor_to_index = {}
        else:
            # Find the lowest floor (a floor that is not a value in above_map_correct)
            floors_above = set(above_map_correct.values())
            lowest_floor = None
            for floor in all_floors:
                if floor not in floors_above:
                    lowest_floor = floor
                    break

            # Traverse upwards to build the index map
            if lowest_floor is not None:
                current_floor = lowest_floor
                index = 0
                while current_floor is not None:
                    self.floor_to_index[current_floor] = index
                    index += 1
                    # Get the floor immediately above current_floor
                    current_floor = above_map_correct.get(current_floor)

        # 2. Store passenger destinations
        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)[1:3]
                self.passenger_destinations[p] = f

        # 3. Identify all passengers from goals (assuming goals are only (served ?p))
        self.all_passengers = {p for (pred, p) in [get_parts(g) for g in self.goals] if pred == 'served'}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        The heuristic value is the sum of:
        - The number of unserved passengers waiting at their origin (needs board).
        - The number of unserved passengers currently boarded (needs depart).
        - An estimate of the movement cost.
        """
        state = node.state  # Current world state.
        task = node.task # Access the task to check goal_reached

        # If the goal is reached, the heuristic is 0.
        if task.goal_reached(state):
            return 0

        # Find the current floor of the lift.
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown, return infinity (should not happen in valid states)
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             return float('inf')

        # Identify unserved passengers.
        unserved_passengers = {
            p for p in self.all_passengers
            if f"(served {p})" not in state
        }

        # If no unserved passengers, but goal not reached, something is wrong with problem definition
        # or state representation. Based on domain, this implies goal is reached.
        if not unserved_passengers:
             return 0

        n_origin = 0 # Count of unserved passengers waiting at origin
        n_boarded = 0 # Count of unserved passengers currently boarded
        needed_floors = set() # Floors that need a board or depart action for unserved passengers

        for passenger in unserved_passengers:
            # Check if passenger is waiting at origin
            origin_fact = next((f for f in state if match(f, "origin", passenger, "*")), None)
            if origin_fact:
                n_origin += 1
                origin_floor = get_parts(origin_fact)[2]
                needed_floors.add(origin_floor)
            else:
                # Passenger must be boarded if not at origin and unserved
                boarded_fact = next((f for f in state if match(f, "boarded", passenger)), None)
                if boarded_fact:
                    n_boarded += 1
                    # Get destination floor from static info
                    destin_floor = self.passenger_destinations.get(passenger)
                    if destin_floor: # Should always exist for a valid problem
                        needed_floors.add(destin_floor)
                    # else: Invalid state - passenger boarded but no destination?

        # Calculate movement cost
        movement_cost = 0
        if needed_floors:
            # Map floors to indices
            current_idx = self.floor_to_index[current_lift_floor] # We checked existence above

            needed_indices = set()
            for f in needed_floors:
                if f in self.floor_to_index:
                    needed_indices.add(self.floor_to_index[f])
                else:
                    # Invalid state - needed floor has no index
                    return float('inf')

            if needed_indices: # Ensure there are valid needed floors with indices
                min_needed_idx = min(needed_indices)
                max_needed_idx = max(needed_indices)

                # Movement cost is the distance to cover the range of needed floors
                # starting from the current floor.
                # This is max(current_idx, max_needed_idx) - min(current_idx, min_needed_idx)
                movement_cost = max(current_idx, max_needed_idx) - min(current_idx, min_needed_idx)

        # Total heuristic estimate
        # Sum of actions at floors + movement cost
        return n_origin + n_boarded + movement_cost

