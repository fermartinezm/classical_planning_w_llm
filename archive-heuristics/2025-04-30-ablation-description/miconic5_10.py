from fnmatch import fnmatch
# Assuming heuristic_base is available in a 'heuristics' directory
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running in the planner environment
# This allows the code to be checked for syntax without the actual planner framework
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # print("Warning: heuristics.heuristic_base not found. Using dummy base class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the total cost to serve all passengers.
    For each unserved passenger, it estimates the cost based on their current state:
    - If waiting at an origin floor: Estimate is travel from current lift floor to origin,
      plus boarding, plus travel from origin to destination, plus departing.
    - If boarded: Estimate is travel from current lift floor to destination, plus departing.
    The total heuristic is the sum of these individual passenger estimates.

    # Assumptions
    - Floors are linearly ordered, and the `above` predicate defines this order.
    - Travel cost between floors is the absolute difference in their levels (Manhattan distance).
    - Boarding and departing each cost 1 action.
    - The heuristic sums costs for passengers independently, ignoring potential
      optimizations from serving multiple passengers on a single trip or floor visit.
      This makes it non-admissible but potentially useful for greedy search.

    # Heuristic Initialization
    - Parse static facts (`task.static`) to determine the linear order of floors
      and create a mapping from floor names to integer levels.
    - Parse relevant static facts (`destin`) to determine the destination floor
      for each passenger that needs to be served according to the goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Determine the current floor of the lift from the state.
    3. Identify the current origin floor for each waiting passenger from the state.
    4. Identify the set of passengers currently boarded in the lift from the state.
    5. Iterate through all passengers whose destination is known (identified during initialization).
    6. For each passenger:
       a. Check if the passenger is already served (present in the state's `served` facts). If yes, contribute 0 to the cost for this passenger.
       b. If not served, check if the passenger is boarded.
       c. If boarded:
          - Get the passenger's destination floor (from initialization data).
          - Calculate the estimated cost for this passenger:
            `abs(level(current_lift_floor) - level(destination_floor)) + 1` (for depart action).
          - Add this cost to the total heuristic.
       d. If not boarded (meaning they are waiting at their origin):
          - Find the passenger's origin floor in the current state.
          - Get the passenger's destination floor (from initialization data).
          - Calculate the estimated cost for this passenger:
            `abs(level(current_lift_floor) - level(origin_floor)) + 1` (for board action) +
            `abs(level(origin_floor) - level(destination_floor)) + 1` (for depart action).
          - Add this cost to the total heuristic.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Build floor level mapping from static 'above' facts.
        # (above f_high f_low) means f_high is immediately above f_low
        floor_above_map = {} # Maps floor_low -> floor_high
        all_floors = set()
        floors_above_others = set()
        floors_below_others = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, f_high, f_low = parts
                    floor_above_map[f_low] = f_high
                    all_floors.add(f_high)
                    all_floors.add(f_low)
                    floors_above_others.add(f_high)
                    floors_below_others.add(f_low)

        # Find the lowest floor (appears only as f_low, not f_high)
        # Assuming there is exactly one lowest floor in a connected component
        lowest_floor_candidates = floors_below_others - floors_above_others
        self.floor_levels = {}

        if not lowest_floor_candidates:
             # Handle case with no 'above' facts or circular 'above' or single floor
             if len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0]
                 self.floor_levels[lowest_floor] = 1
             else:
                 # Cannot determine floor order, heuristic is invalid
                 # print("Error: Cannot determine unique lowest floor from 'above' facts.")
                 pass # self.floor_levels remains empty
        else:
             # Assuming the first found lowest floor is sufficient for a connected chain
             lowest_floor = lowest_floor_candidates.pop()
             current_floor = lowest_floor
             level = 1
             # Traverse upwards from the lowest floor
             while current_floor is not None and current_floor in all_floors:
                  self.floor_levels[current_floor] = level
                  level += 1
                  # Find the floor immediately above the current one using the map
                  current_floor = floor_above_map.get(current_floor)


        # 2. Store passenger destinations from static facts.
        # Destinations are static, so they appear in the initial state or static facts.
        # We collect destinations for all passengers mentioned in static destin facts.
        self.passenger_destinations = {}
        for fact in static_facts:
             if match(fact, "destin", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     _, passenger, destination = parts
                     self.passenger_destinations[passenger] = destination

        # Note: Passengers mentioned in goals but not having a destin fact in static
        # would be missed here. Assuming valid problems have destin facts for all
        # passengers that appear in the goal's served predicates.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        and travel distance to serve all passengers.
        """
        state = node.state  # Current world state.

        # If floor levels weren't successfully built, return infinity
        if not self.floor_levels:
             return float('inf')

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    _, current_lift_floor = parts
                    break

        if current_lift_floor is None or current_lift_floor not in self.floor_levels:
             # This should not happen in a valid miconic state with leveled floors
             return float('inf') # Cannot compute heuristic

        current_lift_level = self.floor_levels[current_lift_floor]

        # Identify waiting, boarded, and served passengers in the current state
        waiting_passengers = {} # passenger -> origin_floor
        boarded_passengers = set()
        served_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "origin":
                if len(parts) == 3:
                    _, passenger, origin_floor = parts
                    waiting_passengers[passenger] = origin_floor
            elif predicate == "boarded":
                if len(parts) == 2:
                    _, passenger = parts
                    boarded_passengers.add(passenger)
            elif predicate == "served":
                if len(parts) == 2:
                    _, passenger = parts
                    served_passengers.add(passenger)

        total_cost = 0  # Initialize action cost counter.

        # Iterate through all passengers whose destination is known (i.e., relevant passengers)
        for passenger, f_destin in self.passenger_destinations.items():
            # If already served, no cost contribution
            if passenger in served_passengers:
                continue

            destin_level = self.floor_levels.get(f_destin)
            if destin_level is None:
                 # Destination floor not found in floor levels - problem setup error?
                 # Assign high cost or skip this passenger
                 total_cost += float('inf') # Indicate problem
                 continue

            if passenger in boarded_passengers:
                # Passenger is boarded, needs to go to destination and depart
                cost_travel = abs(current_lift_level - destin_level)
                cost_actions = 1 # depart
                total_cost += cost_travel + cost_actions
            elif passenger in waiting_passengers:
                # Passenger is waiting at origin, needs pickup, travel, dropoff
                f_origin = waiting_passengers[passenger]
                origin_level = self.floor_levels.get(f_origin)
                if origin_level is None:
                     # Origin floor not found in floor levels - problem setup error?
                     total_cost += float('inf') # Indicate problem
                     continue

                cost_travel_to_origin = abs(current_lift_level - origin_level)
                cost_action_board = 1
                cost_travel_to_destin = abs(origin_level - destin_level)
                cost_action_depart = 1
                total_cost += cost_travel_to_origin + cost_action_board + cost_travel_to_destin + cost_action_depart
            # else: passenger is not served, not boarded, not waiting?
            # This case implies the passenger's state is inconsistent with the domain model
            # (e.g., a passenger just disappeared). We ignore such passengers or add infinite cost.
            # Given standard PDDL, they should be in one of the three states (served, boarded, origin).
            # If a passenger is not served, not boarded, and not in waiting_passengers,
            # it means the 'origin' fact for them is missing from the state.
            # This is an invalid state according to the domain's flow.
            # We could add infinite cost or ignore, ignoring seems safer for heuristic robustness.


        return total_cost
