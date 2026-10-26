from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
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
    # Ensure parts has at least as many elements as args for zip
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# The heuristic class
class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It sums the number of board actions, the number of depart actions, and an
    estimate of the lift movement cost.

    # Assumptions
    - The floors are ordered linearly by the 'above' predicate.
    - Each unserved passenger requires one board action and one depart action.
    - The lift movement cost is estimated based on the range of floors that
      must be visited to pick up waiting passengers and drop off boarded passengers.

    # Heuristic Initialization
    - Parse static facts to build a map from passenger to destination floor.
    - Parse static facts to build a map from floor name to a numerical index,
      representing the floor's position in the building (e.g., 0 for the lowest floor).

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state (all passengers served). If yes, return 0.
    2. Identify the current floor of the lift.
    3. Identify all unserved passengers (those currently waiting at an origin or boarded).
    4. Collect the set of all origin floors for waiting passengers and all destination
       floors for boarded passengers. These are the required stops.
    5. Count the number of board actions needed: This is the number of passengers
       currently waiting at their origin floors.
    6. Count the number of depart actions needed: This is the total number of
       unserved passengers identified in step 3.
    7. Estimate the lift movement cost:
       - Find the minimum and maximum floor indices among the required stops.
       - If there are no required stops, the movement cost is 0.
       - Otherwise, the movement cost is the distance between the minimum and
         maximum required floor indices, plus the minimum distance from the
         current lift floor to either the minimum or maximum required floor index.
         This estimates the cost of traversing the range of required floors,
         starting from the current position.
    8. The total heuristic value is the sum of the number of board actions,
       the number of depart actions, and the estimated lift movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build destination map: passenger -> destination_floor
        self.destin_map = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destin_map[passenger] = floor

        # Build floor map: floor_name -> index (0-based, lowest floor is 0)
        # Assuming facts like (above f_higher f_lower) define the order.
        # We need to find the highest floor and traverse downwards.
        above_pairs = []
        all_floors_from_above = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_higher, f_lower = get_parts(fact)
                above_pairs.append((f_higher, f_lower))
                all_floors_from_above.add(f_higher)
                all_floors_from_above.add(f_lower)

        # Build map: f_higher -> f_lower
        above_to_below = dict(above_pairs)

        # Find the highest floor: a floor that is never the 'f_lower' in an 'above' fact
        all_higher_floors = set(above_to_below.keys())
        all_lower_floors = set(above_to_below.values())

        highest_floor = None
        potential_highest = all_higher_floors - all_lower_floors

        self.floor_map = {} # Initialize floor_map

        if len(potential_highest) == 1:
             highest_floor = potential_highest.pop()
             # Traverse downwards from the highest floor
             current = highest_floor
             index = len(all_floors_from_above) - 1 # Assign highest index to highest floor
             mapped_count = 0
             while current is not None:
                 if current in self.floor_map: # Cycle detection
                     break # Stop traversal on cycle
                 self.floor_map[current] = index
                 index -= 1
                 mapped_count += 1
                 current = above_to_below.get(current)

             # If the traversal didn't map all floors from 'above' facts, something is wrong with the structure
             if mapped_count != len(all_floors_from_above):
                  self._fallback_floor_mapping(all_floors_from_above)

        elif len(all_floors_from_above) == 1: # Case with only one floor
             self.floor_map[list(all_floors_from_above)[0]] = 0
        elif not all_floors_from_above: # No floors defined by 'above'
             self.floor_map = {} # Empty map
        else:
             # Ambiguous structure or no 'above' facts linking floors
             self._fallback_floor_mapping(all_floors_from_above)

        # Ensure all floors mentioned in destin_map are also considered, even if not in 'above' facts
        all_relevant_floors = set(self.destin_map.values()) | all_floors_from_above
        if len(self.floor_map) != len(all_relevant_floors):
             # Re-map using all relevant floors if the initial mapping was incomplete
             self._fallback_floor_mapping(all_relevant_floors)


    def _fallback_floor_mapping(self, floors_to_map):
         """Attempts to map floors by parsing numbers from names, or lexicographically."""
         try:
             # Sort floors by the number part (e.g., f1, f2, ..., f10)
             # This assumes floor names are like 'f' followed by a number.
             # Filter out any floors that don't fit this pattern before sorting by number
             numeric_floors = [f for f in floors_to_map if len(f) > 1 and f[0] == 'f' and f[1:].isdigit()]
             non_numeric_floors = [f for f in floors_to_map if f not in numeric_floors]

             sorted_numeric_floors = sorted(numeric_floors, key=lambda f: int(f[1:]))
             sorted_non_numeric_floors = sorted(non_numeric_floors) # Lexicographical sort for others

             # Combine and assign indices
             sorted_all_floors = sorted_numeric_floors + sorted_non_numeric_floors
             self.floor_map = {f: i for i, f in enumerate(sorted_all_floors)}

         except Exception: # Catch any unexpected errors during fallback
             # Fallback to simple lexicographical sort if number parsing fails completely
             self.floor_map = {f: i for i, f in enumerate(sorted(list(floors_to_map)))}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        # A state is a goal state if all passengers are served.
        # This is true if there are no (origin ...) or (boarded ...) facts.
        has_unserved = False
        for fact in state:
            if match(fact, "origin", "*", "*") or match(fact, "boarded", "*"):
                 has_unserved = True
                 break

        if not has_unserved:
             # If no unserved passengers are found in the state, it must be the goal state.
             # (Assuming valid problem where goal is exactly all passengers served).
             return 0


        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        # If lift location is unknown or not mapped (shouldn't happen in valid states),
        # return a large value.
        if current_lift_floor is None or current_lift_floor not in self.floor_map:
             return 1000000 # Arbitrary large number


        # Collect required stops and count board/depart actions needed
        required_stops = set()
        num_board_needed = 0
        unserved_passengers_in_state = set() # Passengers currently in origin or boarded state

        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, passenger, f_origin = get_parts(fact)
                unserved_passengers_in_state.add(passenger)
                num_board_needed += 1
                if f_origin in self.floor_map:
                    required_stops.add(f_origin)
                f_destin = self.destin_map.get(passenger)
                if f_destin and f_destin in self.floor_map:
                    required_stops.add(f_destin)
            elif match(fact, "boarded", "*"):
                 _, passenger = get_parts(fact)
                 # Check if this passenger is already served (shouldn't be if boarded)
                 if f"(served {passenger})" not in state:
                     unserved_passengers_in_state.add(passenger)
                     f_destin = self.destin_map.get(passenger)
                     if f_destin and f_destin in self.floor_map:
                         required_stops.add(f_destin)

        # Each unserved passenger (currently in origin or boarded state) needs one depart action
        num_depart_needed = len(unserved_passengers_in_state)

        # Estimate lift movement cost
        moves_needed = 0
        # Ensure there are valid required stops and the current floor is mapped
        if required_stops:
            required_indices = {self.floor_map[f] for f in required_stops} # required_stops already filtered for mapped floors
            min_idx = min(required_indices)
            max_idx = max(required_indices)
            current_idx = self.floor_map[current_lift_floor]

            # Minimum moves to visit all floors in [min_idx, max_idx] starting from current_idx
            dist_to_min = abs(current_idx - min_idx)
            dist_to_max = abs(current_idx - max_idx)
            range_dist = max_idx - min_idx

            moves_needed = min(dist_to_min, dist_to_max) + range_dist

        # Total heuristic value
        total_cost = num_board_needed + num_depart_needed + moves_needed

        return total_cost
