from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It sums the estimated cost for each unserved passenger independently,
    including the actions (board, depart) and the estimated movement cost
    for the lift to pick them up and drop them off.

    # Assumptions
    - Passengers are either waiting at their origin, boarded, or served.
    - The 'above' facts define a linear order of floors.
    - The cost of moving one floor is 1.
    - The cost of boarding is 1.
    - The cost of departing is 1.

    # Heuristic Initialization
    - Build a mapping from floor names to numerical indices based on the 'above' facts.
    - Store the destination floor for each passenger from the static facts.
    - Identify the set of passengers that need to be served from the goal conditions.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Find the current floor of the lift.
    2. For each passenger who is not yet served (based on the goal conditions):
       a. Find their destination floor (precomputed).
       b. Check if the passenger is waiting at their origin floor or is boarded.
       c. If waiting at origin_f:
          - Add 1 for the 'board' action.
          - Add 1 for the 'depart' action.
          - Estimate movement cost: Distance from current lift floor to origin_f
            plus distance from origin_f to destination_f.
       d. If boarded:
          - Add 1 for the 'depart' action.
          - Estimate movement cost: Distance from current lift floor to destination_f.
    3. Sum the costs calculated for each unserved passenger.

    This heuristic is non-admissible because it sums movement costs independently
    for each passenger, potentially counting the same lift movement multiple times.
    However, it provides a greedy estimate that correlates with the remaining work.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and the set of passengers to be served.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build floor-to-index mapping from 'above' facts
        # (above f_i f_j) means f_i is immediately above f_j
        below_to_above = {}
        all_floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                f_above, f_below = parts[1], parts[2]
                below_to_above[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Find the lowest floor (a floor that is below something, but nothing is below it)
        all_below_args = set(below_to_above.keys())
        all_above_args = set(below_to_above.values())
        # The lowest floor is in all_below_args but not in all_above_args
        lowest_floor = (all_below_args - all_above_args).pop()

        self.floor_to_index = {}
        current_floor = lowest_floor
        current_index = 1
        # Build the map by following the chain up
        while current_floor in below_to_above or current_floor == lowest_floor:
             self.floor_to_index[current_floor] = current_index
             if current_floor in below_to_above:
                 current_floor = below_to_above[current_floor]
                 current_index += 1
             else: # Reached the highest floor
                 break

        # Store passenger destinations from static facts
        self.passenger_destin = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin":
                passenger, destination_floor = parts[1], parts[2]
                self.passenger_destin[passenger] = destination_floor

        # Get the set of all passengers who need to be served (from goals)
        self.passengers_to_serve = {get_parts(goal)[1] for goal in task.goals if get_parts(goal)[0] == "served"}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find the current floor of the lift
        current_f = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                current_f = parts[1]
                break
        current_f_idx = self.floor_to_index[current_f]

        total_cost = 0

        # Iterate through all passengers that need to be served
        for p in self.passengers_to_serve:
            # If the passenger is already served, they don't contribute to the heuristic cost
            if f"(served {p})" in state:
                continue

            # Find the passenger's destination floor index
            destin_f = self.passenger_destin[p]
            destin_idx = self.floor_to_index[destin_f]

            # Check if the passenger is waiting at their origin floor
            is_waiting = False
            origin_f = None
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "origin" and parts[1] == p:
                    is_waiting = True
                    origin_f = parts[2]
                    break

            if is_waiting:
                # Passenger is waiting: needs board, depart, and movement
                origin_idx = self.floor_to_index[origin_f]
                total_cost += 1 # Cost for 'board' action
                total_cost += 1 # Cost for 'depart' action
                # Movement cost: current -> origin -> destin
                total_cost += abs(current_f_idx - origin_idx) # Move to origin
                total_cost += abs(origin_idx - destin_idx) # Move from origin to destin (while boarded)
            else:
                # Passenger must be boarded (if not served and not waiting)
                # Needs depart and movement
                total_cost += 1 # Cost for 'depart' action
                # Movement cost: current -> destin
                total_cost += abs(current_f_idx - destin_idx) # Move to destin (while boarded)

        return total_cost

