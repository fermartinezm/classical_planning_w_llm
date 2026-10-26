from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the number of remaining board actions, the number of remaining depart actions,
    and an estimate of the floor movements needed to visit all relevant floors.

    # Assumptions
    - Floors are named 'f1', 'f2', ..., 'fn', and 'fi' corresponds to floor number i.
    - The 'above' predicate implies a linear ordering consistent with floor numbers.
    - Each board action costs 1.
    - Each depart action costs 1.
    - Each floor movement (up or down one level) costs 1.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the initial state.
    - Stores the set of all passengers involved in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify the lift's current floor and its corresponding floor number.
    3. Initialize a set `required_stop_nums` to store the floor numbers the lift *must* visit (origin floors for waiting passengers, destination floors for boarded passengers).
    4. Iterate through all passengers identified during initialization:
       - Check if the passenger is currently in the state as '(served ?p)'. If yes, they are served and require no further actions from this state.
       - If the passenger is not served:
          - Check if the passenger is currently in the state as '(origin ?p ?f)'. If yes, the passenger is waiting. Add 1 to the total cost (for the board action) and add the origin floor number (parsed from the floor name) to the `required_stop_nums` set.
          - Check if the passenger is currently in the state as '(boarded ?p)'. If yes, the passenger is boarded. Add 1 to the total cost (for the depart action).
          - Find the passenger's destination floor using the pre-calculated destination mapping (`self.destinations`) and add its floor number (parsed from the floor name) to the `required_stop_nums` set.
    5. If the `required_stop_nums` set is empty, it means no passenger is waiting or boarded. Assuming the problem goal requires all passengers to be served, this state must be the goal state. Return 0.
    6. If the `required_stop_nums` set is not empty:
       - Find the minimum (`min_required_floor_num`) and maximum (`max_required_floor_num`) floor numbers within the set.
       - Calculate the estimated floor moves: This is the total span of floors the lift must cover, starting from its current floor, to reach the range defined by the minimum and maximum required stops. It's calculated as `max(current_floor_number, max_required_floor_num) - min(current_floor_number, min_required_floor_num)`.
       - Add the estimated moves to the total cost.
    7. Return the total cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and identifying all passengers.
        Assumes floor names are 'f' followed by a number (e.g., 'f1', 'f10').
        """
        self.goals = task.goals # Keep goals reference if needed, though not used in current logic
        self.passengers = set()
        self.destinations = {} # Map passenger name to destination floor name

        # Extract passenger destinations and identify all passengers from initial state
        for fact in task.initial_state:
            parts = get_parts(fact)
            if match(fact, "destin", "*", "*"):
                p, f = parts[1], parts[2]
                self.destinations[p] = f
                self.passengers.add(p)
            elif match(fact, "origin", "*", "*"):
                 p, f = parts[1], parts[2]
                 self.passengers.add(p)
            elif match(fact, "boarded", "*"):
                 p = parts[1]
                 self.passengers.add(p)
            # Note: Passengers might also appear in 'served' in initial state
            elif match(fact, "served", "*"):
                 p = parts[1]
                 self.passengers.add(p)

        # Also ensure all passengers mentioned in the goal are included,
        # as they are the ones that ultimately need to be served.
        for goal in self.goals:
            if match(goal, "served", "*"):
                p = get_parts(goal)[1]
                self.passengers.add(p)


    def get_floor_number(self, floor_name):
         """Helper to parse floor number from floor name like 'f1', 'f10'."""
         try:
             # Assumes floor names are always 'f' followed by a number
             return int(floor_name[1:])
         except (ValueError, IndexError):
             # This indicates an unexpected floor naming convention.
             # Returning infinity makes states with unparseable floors seem bad.
             print(f"Error: Could not parse floor number from '{floor_name}'.")
             return float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        total_cost = 0
        required_stop_nums = set()

        # Find the lift's current floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
            # Should not happen in a valid state, but handle defensively
            return float('inf') # Cannot proceed if lift location is unknown

        current_floor_num = self.get_floor_number(current_lift_floor)
        if current_floor_num == float('inf'):
            return float('inf') # Propagate error from floor parsing

        # Track which passengers are currently waiting or boarded
        waiting_passengers = {} # {p: origin_floor}
        boarded_passengers = set() # {p}
        served_passengers = set() # {p}

        for fact in state:
            if match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)[1], get_parts(fact)[2]
                waiting_passengers[p] = f
            elif match(fact, "boarded", "*"):
                p = get_parts(fact)[1]
                boarded_passengers.add(p)
            elif match(fact, "served", "*"):
                p = get_parts(fact)[1]
                served_passengers.add(p)

        # Iterate through all passengers involved in the problem
        for p in self.passengers:
            if p in served_passengers:
                continue # This passenger is already served

            # Passenger needs service
            if p in waiting_passengers:
                # Needs board + depart + moves
                total_cost += 1 # Cost for board action
                origin_floor = waiting_passengers[p]
                origin_floor_num = self.get_floor_number(origin_floor)
                if origin_floor_num == float('inf'): return float('inf')
                required_stop_nums.add(origin_floor_num)

                # Destination floor is also a required stop
                if p in self.destinations:
                    destin_floor = self.destinations[p]
                    destin_floor_num = self.get_floor_number(destin_floor)
                    if destin_floor_num == float('inf'): return float('inf')
                    required_stop_nums.add(destin_floor_num)
                else:
                     # This indicates a problem definition issue if a passenger needs service but has no destination
                     print(f"Error: Destination not found for unserved passenger {p}")
                     return float('inf')

            elif p in boarded_passengers:
                # Needs depart + moves
                total_cost += 1 # Cost for depart action
                # Destination floor is a required stop
                if p in self.destinations:
                    destin_floor = self.destinations[p]
                    destin_floor_num = self.get_floor_number(destin_floor)
                    if destin_floor_num == float('inf'): return float('inf')
                    required_stop_nums.add(destin_floor_num)
                else:
                     print(f"Error: Destination not found for boarded passenger {p}")
                     return float('inf')

            # If a passenger is unserved but neither waiting nor boarded, this is an invalid state
            # (e.g., passenger disappeared). Assume valid states.

        # If no stops are required, all relevant passengers are served.
        if not required_stop_nums:
            # This implies all passengers in self.passengers are in served_passengers.
            # In Miconic, the only way to become served is via depart action after being boarded.
            # So if no one is waiting or boarded, everyone must be served to be a goal state.
            # If required_stop_nums is empty, the heuristic is 0. This is correct for goal states.
            return 0

        # Calculate estimated moves
        min_required_floor_num = min(required_stop_nums)
        max_required_floor_num = max(required_stop_nums)

        # Moves = total span covered starting from current floor
        # This is the distance from the current floor to the closest end of the required range,
        # plus the span of the required range itself.
        moves = max(current_floor_num, max_required_floor_num) - min(current_floor_num, min_required_floor_num)

        total_cost += moves

        return total_cost
