from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re # Import regex for parsing floor names

# Helper functions (from Logistics example, adapted)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential extra spaces or complex structures if necessary, but simple split is usually fine for PDDL facts
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards appropriately
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the total number of actions (move, board, depart) required
    to serve all passengers, calculated by summing the estimated cost
    for each unserved passenger independently.

    For each passenger not yet served:
    - If waiting at origin floor F_o, needing to go to F_d:
      Estimated cost = (moves from current lift floor to F_o) + 1 (board)
                     + (moves from F_o to F_d) + 1 (depart)
    - If boarded, needing to go to F_d:
      Estimated cost = (moves from current lift floor to F_d) + 1 (depart)

    The total heuristic value is the sum of these independent costs for all
    unserved passengers. This heuristic is non-admissible as it overcounts
    shared movement between passengers.

    Floor levels are determined by parsing 'above' predicates and assuming
    a linear floor structure (e.g., f1 is highest, fN is lowest if (above fi fj)
    implies fi is immediately above fj). Distance is absolute difference in levels.
    Assumes floor names are in the format 'f{index}'.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and
        building the floor level mapping.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Build floor level map
        # Assumes floor names are f1, f2, ..., fN and (above fi fj) means fi is immediately above fj
        # This implies the order is fN, f(N-1), ..., f2, f1, where fN is the lowest.
        floor_names = set()
        for fact in self.static_facts:
            # Collect all floor names mentioned in static facts
            if match(fact, "above", "*", "*"):
                _, f1, f2 = get_parts(fact)
                floor_names.add(f1)
                floor_names.add(f2)
            if match(fact, "origin", "*", "*"):
                 _, p, f = get_parts(fact)
                 floor_names.add(f)
            if match(fact, "destin", "*", "*"):
                 _, p, f = get_parts(fact)
                 floor_names.add(f)

        # Find the maximum floor index to determine N
        max_floor_index = 0
        for name in floor_names:
            match_f = re.match(r'f(\d+)', name)
            if match_f:
                try:
                    index = int(match_f.group(1))
                    max_floor_index = max(max_floor_index, index)
                except ValueError:
                    # Should not happen if regex matches \d+
                    pass

        N = max_floor_index
        # Assign levels: fN gets level 0, f(N-1) gets level 1, ..., f1 gets level N-1
        self.floor_levels = {f'f{i}': N - i for i in range(1, N + 1)}

        # Store passenger destinations
        self.passenger_destinations = {}
        for fact in self.static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_destinations[passenger] = floor

    def get_floor_level(self, floor_name):
        """Returns the numerical level for a given floor name."""
        # Use the precomputed map. If a floor name is unexpected, default to level 0 or raise error.
        # Assuming all relevant floors are f1..fN and are in the map.
        return self.floor_levels.get(floor_name) # get() returns None if key not found, will cause error later if not handled

    def dist(self, f1, f2):
        """Calculates the distance (number of floors to traverse) between two floors."""
        level1 = self.get_floor_level(f1)
        level2 = self.get_floor_level(f2)
        if level1 is None or level2 is None:
             # This indicates an issue with floor naming or parsing
             # In a real scenario, handle this error appropriately
             # For this problem, assuming valid floor names f1..fN
             return float('inf') # Or some large value to penalize unknown floors
        return abs(level1 - level2)

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.
        """
        state = node.state

        current_floor = None
        waiting_passengers_facts = [] # List of (p, f) from origin facts
        boarded_passengers = set()
        served_passengers = set()

        # Parse the current state to find relevant facts
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "lift-at":
                current_floor = parts[1]
            elif predicate == "origin":
                p, f = parts[1], parts[2]
                waiting_passengers_facts.append((p, f))
            elif predicate == "boarded":
                boarded_passengers.add(parts[1])
            elif predicate == "served":
                served_passengers.add(parts[1])

        # If current_floor is not found, something is wrong with the state representation
        if current_floor is None:
             # This state is likely invalid or represents a failure state
             return float('inf') # Penalize invalid states

        total_cost = 0

        # Cost for boarded but unserved passengers
        # These are passengers for whom (boarded p) is true and (served p) is false
        boarded_unserved = boarded_passengers - served_passengers
        for p in boarded_unserved:
            dest_floor = self.passenger_destinations.get(p)
            if dest_floor: # Ensure destination is known
                 # Needs move from current to dest + depart action
                 total_cost += self.dist(current_floor, dest_floor) + 1
            # else: Passenger boarded but no destination in static facts? Invalid problem.

        # Cost for waiting passengers
        # These are passengers for whom (origin p f) is true
        # Note: A passenger cannot be both (origin ...) and (boarded ...) or (served ...)
        # in a valid STRIPS state according to the action effects.
        for p, origin_floor in waiting_passengers_facts:
             dest_floor = self.passenger_destinations.get(p)
             if dest_floor and origin_floor: # Ensure floors are known
                 # Needs move current -> origin + board + move origin -> dest + depart
                 total_cost += self.dist(current_floor, origin_floor) + 1 + self.dist(origin_floor, dest_floor) + 1
             # else: Passenger waiting but no destination or origin floor? Invalid problem/state.

        return total_cost

