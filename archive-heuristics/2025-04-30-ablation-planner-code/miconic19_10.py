# from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
import re

# Assume Heuristic base class is provided by the environment
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#         # self.objects = task.objects # Assuming task object has objects

#     def __call__(self, node):
#         raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential errors or unexpected fact formats
        # Assuming string representation based on example state/static.
        return [] # Or raise an error
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
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the number of board actions needed, the number of depart actions needed,
    and an estimate of the minimum lift travel distance to visit all necessary floors
    (origins for waiting passengers, destinations for boarded passengers).

    # Assumptions
    - The floor structure is a linear sequence defined by `above` predicates.
    - The lift can carry multiple passengers simultaneously.
    - The cost of each action (board, depart, up, down) is 1.

    # Heuristic Initialization
    - Extracts the linear order of floors and creates a mapping from floor name to index.
    - Extracts the destination floor for each passenger from `destin` facts.
    - Identifies all passengers who need to be served from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift and its index.
    2. Identify all passengers who have not yet been served (based on goal conditions).
    3. For each unserved passenger:
       - If the passenger is waiting at their origin floor, count this as needing a 'board' action
         and add their origin floor to the set of required floors for pickup.
       - If the passenger is boarded, count this as needing a 'depart' action
         and add their destination floor (looked up from static facts) to the set of required floors for dropoff.
    4. Sum the counts of needed 'board' and 'depart' actions.
    5. Determine the set of all required floors (union of pickup and dropoff floors).
    6. If there are no required floors, the heuristic is 0 (goal state).
    7. If there are required floors, find the minimum and maximum floor indices among them.
    8. Calculate the estimated travel cost: This is the minimum distance from the current lift floor
       to either the minimum or maximum required floor index, plus the distance spanning the range
       of required floors (max_idx - min_idx). This estimates the cost to reach the range and traverse it.
    9. The total heuristic value is the sum of the board actions needed, the depart actions needed,
       and the estimated travel cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Floor order and index mapping.
        - Passenger destinations.
        - Set of all passengers from goal.
        """
        # Assuming task object has attributes goals, static, and objects
        self.goals = task.goals
        self.static = task.static
        # self.objects = task.objects # Not strictly needed for this heuristic logic

        # 1. Build floor order and index mapping
        all_floors = set()
        below_map = {} # Maps floor_above -> floor_below
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_above, f_below = parts[1], parts[2]
                all_floors.add(f_above)
                all_floors.add(f_below)
                below_map[f_above] = f_below

        # Find the lowest floor (a floor that is not a value in below_map)
        lowest_floor = None
        floors_that_are_below = set(below_map.values())
        for floor in all_floors:
            if floor not in floors_that_are_below:
                lowest_floor = floor
                break

        self.floor_to_index = {}
        if lowest_floor:
            current_floor = lowest_floor
            index = 1
            # Build index map by traversing upwards
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                index += 1
                # Find the floor directly above current_floor
                next_floor = None
                for f_above, f_below in below_map.items():
                    if f_below == current_floor:
                        next_floor = f_above
                        break
                current_floor = next_floor
        # Note: If lowest_floor is None or all_floors is empty, self.floor_to_index remains empty.
        # This should be handled gracefully in __call__ if needed, but assumes valid PDDL.


        # 2. Extract passenger destinations
        self.destinations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                passenger, destination_floor = parts[1], parts[2]
                self.destinations[passenger] = destination_floor

        # 3. Identify all passengers from goal conditions
        self.all_passengers = set()
        # Goal is typically (and (served p1) (served p2) ... ) or just (served p1)
        for goal_str in self.goals: # self.goals is a frozenset of strings
             # Use regex to find all served passengers within the goal string
             served_matches = re.findall(r'\(served (\S+)\)', goal_str)
             for passenger in served_matches:
                 self.all_passengers.add(passenger)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        served_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        # Goal is reached if all passengers identified in __init__ are served
        if self.all_passengers.issubset(served_passengers_in_state):
             return 0 # Goal state

        # 1. Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # This indicates an invalid state or problem definition
             return float('inf') # Cannot proceed

        current_floor_idx = self.floor_to_index[current_lift_floor]

        # 2. Identify unserved passengers
        unserved_passengers = self.all_passengers - served_passengers_in_state

        # 3. Count needed actions and required floors
        n_waiting = 0
        n_boarded = 0
        required_floors = set()

        # Get current status and location of unserved passengers
        passenger_info = {} # passenger -> {'status': 'waiting'/'boarded', 'floor': floor}
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == "origin" and parts[1] in unserved_passengers:
                  p, f = parts[1], parts[2]
                  passenger_info[p] = {'status': 'waiting', 'floor': f}
             elif parts and parts[0] == "boarded" and parts[1] in unserved_passengers:
                  p = parts[1]
                  passenger_info[p] = {'status': 'boarded'}

        for p in unserved_passengers:
             info = passenger_info.get(p)
             if info and info['status'] == 'waiting':
                  n_waiting += 1
                  origin_floor = info['floor']
                  if origin_floor in self.floor_to_index:
                      required_floors.add(origin_floor)
                  else:
                      # Invalid origin floor
                      return float('inf')
             elif info and info['status'] == 'boarded':
                  n_boarded += 1
                  destination_floor = self.destinations.get(p)
                  if destination_floor and destination_floor in self.floor_to_index:
                      required_floors.add(destination_floor)
                  else:
                      # Invalid destination floor or destination not found
                      return float('inf')
             # Note: If a passenger is unserved but neither waiting nor boarded,
             # it suggests a state inconsistent with the domain dynamics.
             # The heuristic might underestimate or return inf depending on how it's handled.
             # Assuming valid states where unserved passengers are either waiting or boarded.
             elif not info:
                 # Unserved passenger with no origin or boarded status - indicates invalid state
                 return float('inf')


        # 4. Sum board/depart costs
        action_cost = n_waiting + n_boarded

        # 5. Calculate travel cost
        if not required_floors:
            # This case should be covered by the initial goal check, but as a fallback
            travel_cost = 0
        else:
            required_indices = {self.floor_to_index[f] for f in required_floors}
            min_idx = min(required_indices)
            max_idx = max(required_indices)

            # Travel to the closest required floor
            travel_to_range = min(abs(current_floor_idx - min_idx), abs(current_floor_idx - max_idx))

            # Travel across the range of required floors
            travel_across_range = max_idx - min_idx

            travel_cost = travel_to_range + travel_across_range

        # 9. Total heuristic value
        total_cost = action_cost + travel_cost

        return total_cost
