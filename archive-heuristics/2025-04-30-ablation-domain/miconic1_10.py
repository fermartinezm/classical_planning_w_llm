from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like '(predicate)' or '(predicate arg1 arg2)'
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        return [] # Invalid fact format
    content = fact[1:-1].strip()
    if not content:
        return [] # Empty fact like '()'
    return content.split()

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
    This heuristic estimates the number of actions needed to serve all passengers.
    It sums the number of required board actions, required depart actions, and
    an estimate of the lift movement cost to visit all necessary floors.

    # Assumptions
    - Floor names are in the format 'fN' where N is an integer representing the floor level.
    - The 'above' predicate correctly defines the floor order consistent with 'fN' naming.
    - The goal is to have all passengers served.

    # Heuristic Initialization
    - Extracts all passenger names from the goal state.
    - Extracts destination floors for all passengers from static facts.
    - Builds a mapping from floor names ('fN') to integer floor levels (N).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of all passengers defined in the problem goals.
    2. Identify the set of passengers who are already served in the current state by checking for the `(served ?p)` predicate.
    3. The set of unserved passengers is the difference between all passengers and served passengers.
    4. If there are no unserved passengers, the state is a goal state, and the heuristic is 0.
    5. Identify passengers currently waiting at their origin floors (`P_waiting`) by checking for the `(origin ?p ?f)` predicate. The number of required board actions is `|P_waiting|`.
    6. Identify passengers currently boarded in the lift (`P_boarded`) by checking for the `(boarded ?p)` predicate. The number of required depart actions is `|P_boarded|`.
    7. Determine the set of critical floors the lift must visit:
       - Origin floors for waiting passengers (`F_pickup`). These are the floors `?f` from `(origin ?p ?f)` facts.
       - Destination floors for boarded passengers (`F_dropoff`). These are the destination floors for passengers in `P_boarded`, looked up using the pre-computed destination map.
       The set of critical floors is `F_critical = F_pickup union F_dropoff`.
    8. If `F_critical` is empty (which should only happen if there are no unserved passengers, already handled), the movement cost is 0.
    9. If `F_critical` is not empty, map these floor names to integer levels using the pre-computed floor map. Find the minimum (`min_crit_int`) and maximum (`max_crit_int`) integer levels among critical floors.
    10. Find the current floor of the lift by checking for the `(lift-at ?f)` predicate and map it to an integer level (`current_floor_int`).
    11. Estimate the lift movement cost: The lift must travel from `current_floor_int` to reach the range `[min_crit_int, max_crit_int]` and traverse that range. A simple estimate is the distance to the nearest end of the range plus the width of the range: `min(abs(current_floor_int - min_crit_int), abs(current_floor_int - max_crit_int)) + (max_crit_int - min_crit_int)`. This estimates the total floor units the lift must traverse.
    12. The total heuristic value is the sum of required board actions, required depart actions, and the estimated movement cost: `|P_waiting| + |P_boarded| + movement_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Include initial state to find all floors

        # 1. Get all passenger names from the goal state
        self.all_passengers = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, 'served', '*')
        }

        # 2. Extract destination floors for all passengers
        self.destin_map = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in static_facts
            if match(fact, 'destin', '*', '*')
        }

        # 3. Build floor map: fN -> N
        # Extract all floor names from static facts and initial state
        floor_names = set()
        # Look for floors in 'above' predicates (static)
        for fact in static_facts:
             parts = get_parts(fact)
             if match(fact, 'above', '*', '*'):
                 if len(parts) > 1: floor_names.add(parts[1]) # f_higher
                 if len(parts) > 2: floor_names.add(parts[2]) # f_lower
             # Look for floors in 'destin' predicates (static)
             elif match(fact, 'destin', '*', '*'):
                 if len(parts) > 2: floor_names.add(parts[2]) # f_destin

        # Look for floors in 'lift-at', 'origin' predicates in initial state
        for fact in initial_state:
             parts = get_parts(fact)
             if match(fact, 'lift-at', '*'):
                 if len(parts) > 1: floor_names.add(parts[1])
             elif match(fact, 'origin', '*', '*'):
                 if len(parts) > 2: floor_names.add(parts[2])

        # Use a robust way to extract the number from floor names like 'f1', 'f10'
        def get_floor_number(f_name):
            match = re.match(r'f(\d+)', f_name)
            if match:
                return int(match.group(1))
            # Return a value that places unparseable names at the end if any exist
            return float('inf')

        # Sort floor names numerically based on the number part
        sorted_floor_names = sorted(list(floor_names), key=get_floor_number)

        # Create the mapping fN -> N (using the extracted number as the level)
        self.floor_map = {f_name: get_floor_number(f_name) for f_name in sorted_floor_names}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1-4. Identify unserved passengers. If none, return 0.
        served_passengers = {
            p for p in self.all_passengers
            if '(served {})'.format(p) in state
        }
        unserved_passengers = self.all_passengers - served_passengers

        if not unserved_passengers:
            return 0 # Goal state reached

        # 5. Identify waiting passengers and count board actions
        waiting_passengers = set()
        pickup_floors = set()
        for fact in state:
            parts = get_parts(fact)
            if match(fact, 'origin', '*', '*'):
                 if len(parts) > 2:
                     p, f = parts[1], parts[2]
                     # We only care about unserved passengers for the heuristic calculation
                     if p in unserved_passengers:
                         waiting_passengers.add(p)
                         pickup_floors.add(f)

        num_board_actions = len(waiting_passengers)

        # 6. Identify boarded passengers and count depart actions
        boarded_passengers = set()
        for fact in state:
             parts = get_parts(fact)
             if match(fact, 'boarded', '*'):
                 if len(parts) > 1:
                     p = parts[1]
                     # We only care about unserved passengers
                     if p in unserved_passengers:
                         boarded_passengers.add(p)

        num_depart_actions = len(boarded_passengers)

        # 7. Determine critical floors
        dropoff_floors = set()
        for p in boarded_passengers:
             if p in self.destin_map:
                 dropoff_floors.add(self.destin_map[p])
             # else: This shouldn't happen in a valid miconic state - boarded passenger without destination?

        critical_floors = pickup_floors.union(dropoff_floors)

        # 8. Handle empty critical floors
        # If unserved > 0, then either waiting_passengers > 0 or boarded_passengers > 0.
        # If waiting > 0, pickup_floors > 0. If boarded > 0, dropoff_floors > 0 (assuming valid destin_map).
        # So critical_floors must be non-empty if unserved > 0.
        # The fallback returns a base cost proportional to remaining passengers.
        if not critical_floors:
             return len(unserved_passengers) # Fallback / Base cost

        # 9. Map critical floors to integers and find min/max
        critical_floor_ints = {self.floor_map[f] for f in critical_floors if f in self.floor_map}
        # Ensure critical_floor_ints is not empty after filtering (should be covered by the check above)
        if not critical_floor_ints:
             return len(unserved_passengers) # Double fallback

        min_crit_int = min(critical_floor_ints)
        max_crit_int = max(critical_floor_ints)

        # 10. Find current lift floor and map to integer
        current_lift_floor_str = None
        for fact in state:
             parts = get_parts(fact)
             if match(fact, 'lift-at', '*'):
                 if len(parts) > 1:
                     current_lift_floor_str = parts[1]
                     break # Found the lift location

        if current_lift_floor_str is None or current_lift_floor_str not in self.floor_map:
             # This indicates an invalid state representation or floor not in map
             # Return a large value to discourage this state
             return float('inf')

        current_floor_int = self.floor_map[current_lift_floor_str]

        # 11. Estimate lift movement cost
        # Distance to nearest critical floor + distance to cover the range
        movement_cost = min(
            abs(current_floor_int - min_crit_int),
            abs(current_floor_int - max_crit_int)
        ) + (max_crit_int - min_crit_int)

        # 12. Total heuristic value
        h = num_board_actions + num_depart_actions + movement_cost

        return h
