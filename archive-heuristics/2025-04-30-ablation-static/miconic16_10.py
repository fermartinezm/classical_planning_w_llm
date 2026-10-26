import re
from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided elsewhere
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
            pass
        def __call__(self, node):
            pass

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Or raise an error, depending on expected input robustness
        return []
    # Split by whitespace, ignoring whitespace inside potential quoted strings (not applicable in STRIPS)
    # and handling multiple spaces between parts.
    parts = fact[1:-1].split()
    return parts

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

# Helper to extract number from floor name like 'f12'
def get_floor_number_from_name(floor_name):
    match = re.search(r'\d+', floor_name)
    if match:
        return int(match.group(0))
    # Return a value that ensures names without numbers are sorted consistently,
    # e.g., after numbered floors. Or raise an error if names must have numbers.
    # Assuming valid miconic floor names like f1, f2, ...
    return float('inf')


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions (board, depart, and
    lift movements) required to serve all passengers. It calculates the cost
    for each unserved passenger independently and sums these costs.

    # Assumptions
    - Floors are ordered numerically based on the digit part of their names
      (e.g., f1 < f2 < f10). This order is used to calculate floor distances.
    - Passengers are only boarded once from their initial origin floor.
    - The cost of moving the lift one floor is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.

    # Heuristic Initialization
    - Extracts the initial origin and destination floor for each passenger
      from the initial state.
    - Collects all floor names and creates a mapping from floor names to
      numerical indices by sorting the names based on their numeric part.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the lift and its corresponding floor number.
    2. Initialize the total heuristic cost to 0.
    3. For each passenger identified in the initial state:
       a. Check if the passenger is already served in the current state. If yes, skip.
       b. Get the passenger's destination floor and its number (from initialization).
       c. Check if the passenger is waiting at their origin floor (using the
          `origin` predicate in the current state).
          - If yes: Get their current origin floor and its number. The estimated
            cost for this passenger is:
            - Distance from current lift floor to their origin floor.
            - Cost of boarding (1).
            - Distance from their origin floor to their destination floor.
            - Cost of departing (1).
            Add this total to the heuristic cost.
       d. Check if the passenger is boarded (using the `boarded` predicate
          in the current state).
          - If yes: Get their destination floor and its number. The estimated
            cost for this passenger is:
            - Distance from the current lift floor to their destination floor.
            - Cost of departing (1).
            Add this total to the heuristic cost.
       e. If a passenger is unserved but neither waiting nor boarded, this
          indicates an unexpected state. The heuristic might return infinity
          or a large value, but assuming valid states, this case is skipped.
    4. The final sum is the heuristic value. If the state is the goal state,
       this sum will be 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger origins/destinations
        and floor ordering based on name numbers.
        """
        self.task = task
        self.goals = task.goals

        self.initial_origins = {}
        self.destinations = {}
        all_floor_names = set()

        # Parse initial state for origins, destinations, and floor names
        for fact in task.initial_state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "origin":
                p, f = parts[1], parts[2]
                self.initial_origins[p] = f
                all_floor_names.add(f)
            elif parts[0] == "destin":
                p, f = parts[1], parts[2]
                self.destinations[p] = f
                all_floor_names.add(f)
            elif parts[0] == "lift-at":
                 f = parts[1]
                 all_floor_names.add(f)

        # Collect floor names from static facts as well (e.g., from 'above')
        for fact in task.static:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == "above":
                 if len(parts) > 1: all_floor_names.add(parts[1])
                 if len(parts) > 2: all_floor_names.add(parts[2])
             # Add other predicates if they involve floors and are static

        # Sort floor names numerically based on the digit part and create mapping
        # Assumes floor names are like 'f1', 'f2', 'f10'
        sorted_floor_names = sorted(list(all_floor_names), key=get_floor_number_from_name)
        self.floor_to_number = {name: i for i, name in enumerate(sorted_floor_names)}
        self.number_to_floor = {i: name for name, i in self.floor_to_number.items()}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.task.goal_reached(state):
             return 0

        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # Should always find the lift location in a valid state
        if current_lift_floor is None:
             return float('inf') # Should not happen in solvable problems

        current_lift_floor_number = self.floor_to_number.get(current_lift_floor, float('inf'))
        if current_lift_floor_number == float('inf'):
             return float('inf') # Unknown floor

        total_cost = 0

        # Iterate through all passengers identified in the initial state
        all_passengers = set(self.initial_origins.keys()).union(self.destinations.keys())

        for p in all_passengers:
            # Check if passenger is already served
            if f"(served {p})" in state:
                continue # This passenger is done

            # Passenger is not served. Find their status.
            is_waiting = False
            waiting_origin = None
            for fact in state:
                 if match(fact, "origin", p, "*"):
                      is_waiting = True
                      waiting_origin = get_parts(fact)[2]
                      break

            is_boarded = f"(boarded {p})" in state

            # Get destination floor number
            dest_floor = self.destinations.get(p)
            if dest_floor is None:
                 # Passenger exists but has no destination? Should not happen in valid problems.
                 continue
            dest_number = self.floor_to_number.get(dest_floor, float('inf'))
            if dest_number == float('inf'): return float('inf') # Unknown floor

            if is_waiting:
                # Passenger is waiting at waiting_origin, needs board and depart
                origin_number = self.floor_to_number.get(waiting_origin, float('inf'))
                if origin_number == float('inf'): return float('inf') # Unknown floor

                # Cost = travel to origin + board + travel origin to destin + depart
                travel_to_origin_cost = abs(current_lift_floor_number - origin_number)
                board_cost = 1
                travel_origin_to_destin_cost = abs(dest_number - origin_number)
                depart_cost = 1

                total_cost += travel_to_origin_cost + board_cost + travel_origin_to_destin_cost + depart_cost

            elif is_boarded:
                # Passenger is boarded, needs depart.
                # Cost = travel from current lift floor to destin + depart.
                travel_current_to_destin_cost = abs(current_lift_floor_number - dest_number)
                depart_cost = 1
                total_cost += travel_current_to_destin_cost + depart_cost

            # else: Passenger is unserved but neither waiting nor boarded.
            # This state should not be reachable from a valid initial state
            # via valid actions. Ignore or return inf. Assuming valid states.


        return total_cost
