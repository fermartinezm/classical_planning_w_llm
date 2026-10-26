from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for malformed facts, although PDDL facts should be well-formed.
        return []
    return fact[1:-1].split()

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

# Helper function for numerical sorting of floor names like 'f1', 'f10', 'f2'
def numerical_sort_key(floor_name):
    """Extract the number from floor names like 'fN' for sorting."""
    match = re.match(r'f(\d+)', floor_name)
    if match:
        return int(match.group(1))
    # Handle cases that don't match the pattern by placing them last
    return float('inf')

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the number of board actions needed, the number of depart actions needed,
    and an estimate of the minimum lift movement required to visit all relevant floors.

    # Assumptions
    - All passengers need to be served (reach their destination).
    - A passenger is either waiting at their origin floor, boarded in the lift, or served.
    - The lift moves one floor at a time (up or down).
    - Floor names follow the pattern 'fN' where N is a number, allowing numerical sorting.
    - The 'above' predicates define a linear ordering of floors consistent with 'fN' naming.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from static facts.
    - Builds a mapping from floor names (e.g., 'f5') to numerical floor levels (e.g., 5)
      by sorting the floor names numerically based on the 'fN' pattern found in static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift from the state.
    2. Check if the goal state is reached (all passengers served). If yes, heuristic is 0.
    3. Identify all passengers who are not yet served by checking against the 'served' facts in the state.
    4. For each unserved passenger:
       - If the passenger is waiting at their origin floor (check for '(origin p f)' in state):
         - Increment the count of 'board' actions needed (one per passenger waiting at origin).
         - Add the passenger's origin floor to the set of floors the lift must visit.
       - Increment the count of 'depart' actions needed (one per unserved passenger).
       - Add the passenger's destination floor (looked up from initialization data) to the set of floors the lift must visit.
    5. If there are unserved passengers:
       - Calculate the estimated lift movement cost:
         - Get the numerical floor levels for all floors in the set of required service floors.
         - Find the minimum and maximum floor numbers among these required floors.
         - The movement cost is estimated as the distance from the current lift floor number
           to the closer of the minimum or maximum required floor numbers, plus the distance
           between the minimum and maximum required floor numbers. This estimates the cost
           to reach the range of required floors and then traverse that range.
       - The total heuristic value is the sum of:
         - The total count of 'board' actions needed.
         - The total count of 'depart' actions needed.
         - The estimated lift movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ordering and passenger destinations.
        """
        self.goals = task.goals
        self.static = task.static

        # Build floor mapping: floor name -> floor number
        # Collect all unique floor names from static facts (e.g., from 'above' or 'destin')
        floor_names = set()
        for fact in self.static:
            parts = get_parts(fact)
            # Collect floors from 'above' facts
            if len(parts) == 3 and parts[0] == 'above':
                 floor_names.add(parts[1])
                 floor_names.add(parts[2])
            # Collect floors from 'destin' facts
            elif len(parts) == 3 and parts[0] == 'destin':
                 floor_names.add(parts[2])

        # Sort floor names numerically (e.g., f1, f2, f10)
        sorted_floor_names = sorted(list(floor_names), key=numerical_sort_key)

        # Create floor name to number mapping (1-based index)
        self.floor_map = {name: i + 1 for i, name in enumerate(sorted_floor_names)}

        # Build passenger destination mapping: passenger name -> destination floor name
        self.destin_map = {}
        for fact in self.static:
            parts = get_parts(fact)
            if match(fact, "destin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                self.destin_map[passenger] = floor

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
             return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "lift-at", "*"):
                current_lift_floor = parts[1]
                break

        # Should always find lift-at in a valid state for this domain.
        # If not found, it's an invalid state representation for a solvable problem.
        # Return infinity to prune this branch in search.
        if current_lift_floor is None:
             return float('inf')

        current_f_num = self.floor_map[current_lift_floor]

        num_board_needed = 0
        num_depart_needed = 0
        service_floors = set() # Floors the lift must visit

        # Identify unserved passengers and required service floors
        # Iterate through all passengers defined in the problem (via destin_map)
        # and check their status in the current state.
        for passenger, destin_floor in self.destin_map.items():
             # Check if the passenger is served in the current state
             if f'(served {passenger})' not in state:
                  # This passenger is unserved
                  num_depart_needed += 1 # Each unserved passenger needs a depart action
                  service_floors.add(destin_floor) # The lift must visit their destination floor

                  # Check if passenger is waiting at their origin floor in the current state
                  is_at_origin = False
                  for fact in state:
                       parts = get_parts(fact)
                       if match(fact, "origin", passenger, "*"):
                            origin_floor = parts[2]
                            num_board_needed += 1 # Needs a board action
                            service_floors.add(origin_floor) # The lift must visit their origin floor
                            is_at_origin = True
                            break # Found the origin fact for this passenger

                  # If a passenger is unserved and not at their origin, they must be boarded.
                  # No additional board action is needed from the current state if already boarded.


        # Calculate estimated lift movement cost
        if not service_floors:
             # This case should only be reached if all passengers are served,
             # which is handled by the initial goal check.
             movement_cost = 0
        else:
            # Get numerical floor levels for all required service floors
            service_floor_nums = [self.floor_map[f] for f in service_floors]
            min_f_num = min(service_floor_nums)
            max_f_num = max(service_floor_nums)

            # Estimate movement: distance to reach the range [min_f, max_f] + distance to sweep the range
            dist_to_min = abs(current_f_num - min_f_num)
            dist_to_max = abs(current_f_num - max_f_num)
            dist_range = abs(max_f_num - min_f_num)

            # The lift must travel from current_f_num to somewhere in [min_f, max_f],
            # and then traverse the distance max_f - min_f.
            # Minimum distance to reach the range is min(dist_to_min, dist_to_max).
            movement_cost = min(dist_to_min, dist_to_max) + dist_range


        # Total heuristic is sum of actions and estimated movement
        total_cost = num_board_needed + num_depart_needed + movement_cost

        return total_cost
