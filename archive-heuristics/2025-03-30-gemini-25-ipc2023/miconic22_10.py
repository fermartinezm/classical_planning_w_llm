import math
from fnmatch import fnmatch
# Assuming the Heuristic base class is available in this path
# from heuristics.heuristic_base import Heuristic

# Placeholder for the Heuristic base class if not available
# Remove this if the actual base class is in the expected path
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handles facts like '(predicate obj1 obj2)'
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern using fnmatch for wildcards.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    # Ensure parts and args have elements before zipping if needed, though split() handles empty strings.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic (elevator) domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the necessary board and depart actions and adds an estimate of the
    lift movement required based on the current locations of the lift and passengers,
    and their destinations. The heuristic is designed for greedy best-first search
    and is not necessarily admissible.

    # Assumptions
    - The building has a linear arrangement of floors (e.g., f1, f2, ..., fN).
    - The `(above f_i f_j)` predicate means floor `f_i` is strictly higher than floor `f_j`.
    - Based on examples, floors are named `f1, f2, ...` where `f1` is the highest.
    - The lift moves one floor per `up` or `down` action. Note: The PDDL action names
      might be counter-intuitive based on the `above` predicate interpretation. `up` seems
      to move down, and `down` seems to move up if `(above f_higher f_lower)` holds.
      The distance calculation assumes 1 action per floor difference.
    - Each passenger requires one `board` and one `depart` action to be served.

    # Heuristic Initialization
    - Extracts passenger destination information from static facts `(destin p f)`.
    - Determines the set and total number of passengers.
    - Identifies all unique floor names from static facts and initial state predicates
      like `(lift-at f)` and `(origin p f)`.
    - Calculates the level (height) of each floor based on the assumption that
      floors are named `f1, f2, ..., fN` with `f1` being the highest (level N-1)
      and `fN` the lowest (level 0). Issues warnings if naming convention is not met.
    - Stores these mappings (destinations, floor levels, passenger set) for
      efficient lookup during heuristic calculation.
    - Stores goal facts as a frozenset for efficient checking.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check Goal State:**
        - If the current state satisfies all goal conditions (`self.goals <= state`), return 0.

    2.  **Identify Current State Details:**
        - Find the current floor of the lift `lift_f`. Handle cases where lift location is missing or unknown.
        - Identify passengers waiting at their origin floor (`passenger_origins` map).
        - Identify passengers currently inside the lift (`passengers_boarded` set).
        - Identify passengers already served (`passengers_served` set).

    3.  **Calculate Action Counts:**
        - `num_waiting = len(passenger_origins)`. This is the number of `board` actions needed for waiting passengers.
        - `num_served = len(passengers_served)`.
        - `num_unserved = self.num_total_passengers - num_served`. This is the number of `depart` actions eventually needed for all passengers not yet served.

    4.  **Estimate Movement Cost:**
        - Determine the set of `Relevant_floors` the lift needs to potentially visit:
          Includes the current `lift_f`, origin floors of waiting passengers, and
          destination floors of all unserved passengers. Only include floors with known levels.
        - If `Relevant_floors` contains more than one floor:
            - Calculate the vertical `span` = `max_level - min_level` of relevant floors.
            - Calculate `nearest_dist` = distance from `lift_f` to the closest other relevant floor.
            - `movement_cost = nearest_dist + span`. This estimates the cost to reach the
              operating range of floors and then cover that range.
        - If only `lift_f` is relevant or no relevant floors are known, `movement_cost = 0`.

    5.  **Total Heuristic Value:**
        - `h = num_waiting + num_unserved + movement_cost`.
        - Ensure `h` is at least 1 if the state is not a goal state (to guide search away from non-goal states evaluated as 0). Return `max(1, h)` if not goal, `0` if goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = frozenset(task.goals) # Store goals as frozenset
        static_facts = task.static

        # Extract passenger destinations and identify all passengers
        self.passenger_destins = {}
        passengers = set()
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                dest_floor = parts[2]
                self.passenger_destins[passenger] = dest_floor
                passengers.add(passenger)
        self.num_total_passengers = len(passengers)
        self.all_passengers = passengers

        # Extract all unique floor names from static facts and initial state
        floors = set()
        for fact in static_facts:
            # Predicates involving floors: above, destin
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "above":
                floors.add(parts[1])
                floors.add(parts[2])
            elif predicate == "destin":
                floors.add(parts[2]) # Destination floor

        for fact in task.initial_state:
             # Predicates involving floors: lift-at, origin
             parts = get_parts(fact)
             predicate = parts[0]
             if predicate == "lift-at":
                 floors.add(parts[1]) # Lift start floor
             elif predicate == "origin":
                 floors.add(parts[2]) # Origin floor

        if not floors:
             print("Warning: Could not determine any floor names.")
             self.floors = []
             self.num_floors = 0
             self.floor_levels = {}
        else:
            # Sort floors assuming 'f1', 'f2', ... naming convention
            floor_list = list(floors)
            is_f_numeric = all(f.startswith('f') and f[1:].isdigit() for f in floor_list)

            if is_f_numeric:
                 # Sort numerically based on the number part
                 floor_list.sort(key=lambda f: int(f[1:]))
                 self.floors = floor_list
                 self.num_floors = len(self.floors)
                 # Assign levels: Assuming f1 (first in sorted list) is highest (level N-1), fN is lowest (level 0)
                 self.floor_levels = {
                     floor_name: self.num_floors - 1 - i
                     for i, floor_name in enumerate(self.floors)
                 }
            else:
                 # Fallback if naming convention is not met
                 print(f"Warning: Floor names {floor_list} do not follow 'f<number>' convention. Using arbitrary sort.")
                 self.floors = sorted(floor_list) # Arbitrary sort
                 self.num_floors = len(self.floors)
                 # Assign arbitrary levels based on sort order - This might be inaccurate!
                 # A better fallback would be to build a graph from 'above' facts and find levels.
                 # For simplicity here, we assign levels based on arbitrary sort order.
                 print("Warning: Assigning arbitrary floor levels based on sort order.")
                 self.floor_levels = {floor_name: i for i, floor_name in enumerate(self.floors)}


    def dist(self, f1, f2):
        """Calculate the vertical distance (number of moves) between two floors."""
        if f1 not in self.floor_levels or f2 not in self.floor_levels:
             # This might happen if floor parsing failed or state contains unexpected floors
             # print(f"Warning: Unknown floor encountered in dist(): {f1} or {f2}")
             return 0 # Return 0 to avoid inflating heuristic due to parsing issues
        return abs(self.floor_levels[f1] - self.floor_levels[f2])

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given node."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find lift location
        lift_f = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_f = get_parts(fact)[1]
                break
        if lift_f is None:
            print("Error: Lift location not found in state. Cannot compute heuristic.")
            return float('inf') # Indicate error / invalid state for search
        if lift_f not in self.floor_levels:
             # This can happen if lift starts at a floor not mentioned elsewhere or parsing failed
             print(f"Error: Lift floor {lift_f} has no level assigned. Cannot compute heuristic.")
             return float('inf') # Indicate error

        # Find waiting, boarded, and served passengers
        passenger_origins = {}
        passengers_boarded = set()
        passengers_served = set()
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "origin":
                passenger_origins[parts[1]] = parts[2]
            elif predicate == "boarded":
                passengers_boarded.add(parts[1])
            elif predicate == "served":
                passengers_served.add(parts[1])

        # Calculate counts
        num_waiting = len(passenger_origins)
        num_served = len(passengers_served)
        # Ensure calculation uses total passengers identified at init
        num_unserved = self.num_total_passengers - num_served

        # Cost for board actions
        cost_board = num_waiting

        # Cost for depart actions
        cost_depart = num_unserved

        # Estimate movement cost
        relevant_floors = set()
        # Add lift floor only if its level is known (already checked lift_f is known)
        relevant_floors.add(lift_f)

        # Identify unserved passengers to find their destinations
        unserved_passengers = self.all_passengers - passengers_served

        # Add origins of waiting passengers
        for p, origin_f in passenger_origins.items():
            if origin_f in self.floor_levels:
                relevant_floors.add(origin_f)

        # Add destinations of all unserved passengers (waiting or boarded)
        for p in unserved_passengers:
            destin_f = self.passenger_destins.get(p)
            if destin_f and destin_f in self.floor_levels:
                relevant_floors.add(destin_f)

        movement_cost = 0
        if len(relevant_floors) > 1:
            # We know lift_f is in self.floor_levels from checks above
            levels = [self.floor_levels[f] for f in relevant_floors]
            min_level = min(levels)
            max_level = max(levels)
            span = max_level - min_level

            # Find distance to nearest relevant floor (excluding current lift floor)
            nearest_dist = float('inf')
            found_other_floor = False
            for f in relevant_floors:
                if f != lift_f:
                    found_other_floor = True
                    distance = self.dist(lift_f, f)
                    nearest_dist = min(nearest_dist, distance)

            if not found_other_floor: # Only lift_f was relevant
                nearest_dist = 0
            elif nearest_dist == float('inf'): # Should not happen if found_other_floor is True
                 nearest_dist = 0 # Safety default

            movement_cost = nearest_dist + span

        # Total heuristic value
        h = cost_board + cost_depart + movement_cost

        # Ensure heuristic is at least 1 for non-goal states
        if h == 0 and not (self.goals <= state):
            # This case should ideally not happen if there are unserved passengers
            # But if it does, return 1 to ensure search progresses.
            return 1
        else:
            # Ensure heuristic is non-negative
            return max(0, h)

