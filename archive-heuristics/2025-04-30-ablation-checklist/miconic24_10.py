from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import sys

# Define a very large number to represent infinity for distance calculations
INF = sys.maxsize

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It counts the number of necessary board and depart actions
    and adds an estimate of the lift movement cost.

    # Assumptions
    - The goal is to serve all passengers.
    - Each unboarded passenger needs one 'board' action.
    - Each boarded passenger needs one 'depart' action.
    - The lift movement cost is estimated based on the range of floors
      that need to be visited (origin floors for unboarded passengers,
      destination floors for boarded passengers).

    # Heuristic Initialization
    - Builds a mapping from floor names to numerical indices based on the
      'above' predicates to calculate floor distances.
    - Builds a mapping from passenger names to their destination floors.
    - Identifies all passengers from the goal state.

    # Step-by-Step Thinking for Computing the Heuristic Value
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Current Lift Location:** Find the floor where the lift is currently located.
    2.  **Identify Unserved Passengers:** Determine which passengers have not yet reached their destination (`(served ?p)` is not true).
    3.  **Count Required Board/Depart Actions:**
        *   For each unserved passenger:
            *   If they are at their origin floor (`(origin ?p ?f)` is true), they need a 'board' action. Count these.
            *   If they are boarded (`(boarded ?p)` is true), they need a 'depart' action at their destination. Count these.
    4.  **Identify Required Floors:** Collect the set of floors that the lift *must* visit:
        *   The origin floor for every unserved, unboarded passenger.
        *   The destination floor for every unserved, boarded passenger.
    5.  **Estimate Lift Movement Cost:**
        *   If there are no required floors (all relevant passengers are served or the goal is empty), the movement cost is 0.
        *   Otherwise, find the minimum and maximum floor indices among the required floors.
        *   Calculate the distance between the current lift floor index and the minimum required floor index (`dist_to_min`).
        *   Calculate the distance between the current lift floor index and the maximum required floor index (`dist_to_max`).
        *   Calculate the range distance between the minimum and maximum required floor indices (`range_dist = max_idx - min_idx`).
        *   Estimate movement cost as the minimum of:
            *   Going to the minimum required floor and sweeping up to the maximum: `dist_to_min + range_dist`.
            *   Going to the maximum required floor and sweeping down to the minimum: `dist_to_max + range_dist`.
        *   This estimates the cost of reaching one end of the necessary floor range and traversing the entire range.
    6.  **Sum Costs:** The total heuristic value is the sum of the required board actions, required depart actions, and the estimated lift movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and the list of all passengers to be served.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Build floor mapping from names to numerical indices
        self.floor_to_index = {}
        self._build_floor_mapping(task.static)

        # Build passenger destination mapping
        self.passenger_to_destin = {}
        self._build_passenger_destin_mapping(task.static)

        # Get list of all passengers from goals (assuming goal is always (served p) for all p)
        self.all_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def _build_floor_mapping(self, static_facts):
        """
        Builds a mapping from floor names (e.g., 'f1', 'f2') to numerical indices
        (e.g., 0, 1, 2...) based on the 'above' predicates.
        Assumes 'above' facts define a linear order of floors.
        """
        above_map = {} # Maps floor_below -> floor_above
        all_floors = set()
        floors_above_set = set() # Floors that are the first arg of 'above'
        floors_below_set = set() # Floors that are the second arg of 'above'

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1:]
                above_map[f_below] = f_above
                floors_above_set.add(f_above)
                floors_below_set.add(f_below)
                all_floors.add(f_above)
                all_floors.add(f_below)

        if not all_floors:
             # Handle case with no floors or no above facts - shouldn't happen in valid problems
             return

        # The lowest floor is the one that is in floors_below_set but not floors_above_set
        lowest_floor = (floors_below_set - floors_above_set).pop() # Assumes exactly one lowest floor

        # Build the mapping by following the 'above' chain starting from the lowest floor
        current_floor = lowest_floor
        index = 0
        while current_floor is not None:
            self.floor_to_index[current_floor] = index
            index += 1
            # Get the floor directly above the current_floor
            current_floor = above_map.get(current_floor)

        # Optional: Verify all floors were mapped
        # if len(self.floor_to_index) != len(all_floors):
        #     print("Warning: Not all floors were mapped correctly!", file=sys.stderr)


    def _build_passenger_destin_mapping(self, static_facts):
         """Builds a mapping from passenger names to their destination floors."""
         for fact in static_facts:
             if match(fact, "destin", "*", "*"):
                 p, f = get_parts(fact)[1:]
                 self.passenger_to_destin[p] = f


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        Estimates cost as (needed board actions) + (needed depart actions) + (estimated lift movement).
        """
        state = node.state

        # 1. Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        # In a valid miconic state, the lift location should always be defined.
        if current_lift_floor is None:
             # This indicates an invalid state representation or initial state.
             # Return infinity or a very high cost.
             return INF # Or handle as an error

        current_lift_idx = self.floor_to_index.get(current_lift_floor, INF)
        if current_lift_idx == INF:
             # Lift at an unknown floor - indicates an invalid state.
             return INF


        # 2. Identify required floors and count board/depart actions
        required_floors = set()
        num_boards_needed = 0
        num_departs_needed = 0

        # Create sets for quick lookup of origins and boarded status
        origins_in_state = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}
        boarded_in_state = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        served_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}


        for passenger in self.all_passengers:
            served = passenger in served_in_state

            if not served:
                boarded = passenger in boarded_in_state

                if not boarded:
                    # Passenger is at origin, needs boarding
                    num_boards_needed += 1
                    origin_floor = origins_in_state.get(passenger)
                    if origin_floor: # Should always find origin if not boarded/served
                        required_floors.add(origin_floor)
                    # else: # Indicates an invalid state - passenger not served, not boarded, but no origin?
                        # return INF # Or handle as an error
                else:
                    # Passenger is boarded, needs departing at destination
                    num_departs_needed += 1
                    destin_floor = self.passenger_to_destin.get(passenger)
                    if destin_floor: # Should always find destination for a passenger
                         required_floors.add(destin_floor)
                    # else: # Indicates an invalid state - boarded passenger with no destination?
                        # return INF # Or handle as an error


        # 3. Calculate movement cost
        movement_cost = 0
        if required_floors:
            required_indices = {self.floor_to_index.get(f, INF) for f in required_floors}
            # Filter out any floors not found in the mapping (invalid state)
            required_indices = {idx for idx in required_indices if idx != INF}

            if not required_indices:
                 # All required floors were invalid - indicates an invalid state
                 return INF

            min_idx = min(required_indices)
            max_idx = max(required_indices)

            # Estimate movement: distance to closest required floor + distance to cover the range
            dist_to_min = abs(current_lift_idx - min_idx)
            dist_to_max = abs(current_lift_idx - max_idx)
            range_dist = max_idx - min_idx

            # The lift must travel from its current position to at least one end
            # of the required floor range (min_idx or max_idx) and then traverse
            # the entire range (min_idx to max_idx).
            # Cost = (distance to closest end) + (distance to traverse range)
            movement_cost = min(dist_to_min, dist_to_max) + range_dist

            # Alternative calculation (equivalent):
            # Option 1: Go to min, then sweep up to max
            # cost1 = dist_to_min + range_dist
            # Option 2: Go to max, then sweep down to min
            # cost2 = dist_to_max + range_dist
            # movement_cost = min(cost1, cost2)


        # 4. Total heuristic = board actions + depart actions + movement actions
        total_cost = num_boards_needed + num_departs_needed + movement_cost

        return total_cost

