# Assuming heuristic_base.py exists and contains a Heuristic base class
# Example:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         raise NotImplementedError

# from heuristics.heuristic_base import Heuristic # Uncomment this line in the actual file

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.task = task
    def __call__(self, node):
        raise NotImplementedError

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
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
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the minimum travel cost for the lift to visit all
    necessary floors (origins of waiting passengers and destinations of
    boarded passengers) and the number of board/depart actions required
    for each unserved passenger.

    # Assumptions
    - The goal is to serve all passengers.
    - Each waiting passenger requires a board and a depart action.
    - Each boarded passenger requires a depart action.
    - The lift must visit the origin floor of a waiting passenger to board them.
    - The lift must visit the destination floor of a boarded passenger to depart them.
    - The minimum travel cost to visit a set of floors is the distance to reach
      the range of those floors plus the distance to traverse that range.
    - Floors are ordered linearly based on `above` facts.

    # Heuristic Initialization
    - Extracts the floor ordering from `above` facts to create a floor-to-rank
      mapping for distance calculation.
    - Extracts the destination floor for each passenger from `destin` facts.
    - Identifies all passengers involved in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information from the State:
       - Find the current floor of the lift (`lift-at ?f`).
       - Identify passengers who are waiting at their origin (`origin ?p ?f`).
       - Identify passengers who are currently boarded (`boarded ?p`).
       - Identify passengers who have been served (`served ?p`).

    2. Identify Unserved Passengers:
       - An unserved passenger is any passenger involved in the problem who is not yet `served`.

    3. Determine Necessary Stop Floors:
       - For each unserved passenger who is waiting at their origin, their origin floor is a necessary stop floor (for pickup).
       - For each unserved passenger who is currently boarded, their destination floor is a necessary stop floor (for dropoff).
       - Collect all these unique floors into a set of `stop_floors`.

    4. Calculate Minimum Travel Cost:
       - If `stop_floors` is empty (all unserved passengers are already at their destination or there are no unserved passengers), the travel cost is 0.
       - If `stop_floors` is not empty:
         - Get the ranks of all floors in `stop_floors`. Filter out any floors not found in the pre-calculated floor ranks (shouldn't happen in valid PDDL).
         - If there are valid stop floors with ranks:
           - Find the minimum and maximum ranks among these stop floors.
           - Get the actual floor names corresponding to the min and max ranks.
           - Calculate the distance between the min and max stop floors (`range_dist`).
           - Calculate the distance from the current lift floor to the min stop floor (`dist_to_min`).
           - Calculate the distance from the current lift floor to the max stop floor (`dist_to_max`).
           - The minimum travel cost is the cost to reach the closer end of the required floor range plus the cost to traverse the entire range: `min(dist_to_min, dist_to_max) + range_dist`.
         - If there are no valid stop floors with ranks (e.g., stop floors were identified but none are known floors), travel cost remains 0.

    5. Calculate Action Cost:
       - Each unserved passenger who is waiting at their origin needs 2 actions: `board` and `depart`.
       - Each unserved passenger who is currently boarded needs 1 action: `depart`.
       - The total action cost is `(|waiting_unserved| * 2) + (|boarded_unserved| * 1)`.

    6. Compute Total Heuristic Value:
       - The heuristic value is the sum of the minimum travel cost and the total action cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ranks and passenger destinations.
        """
        super().__init__(task) # Call base class constructor

        static_facts = task.static

        # Build floor ranking from 'above' facts
        # (above f_higher f_lower) means f_higher is immediately above f_lower
        is_immediately_above = {}
        all_floors_mentioned = set()
        floors_that_are_above = set() # Floors that appear as f_higher

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'above':
                if len(parts) == 3: # Ensure correct number of arguments
                    f_higher, f_lower = parts[1], parts[2]
                    is_immediately_above[f_lower] = f_higher
                    all_floors_mentioned.add(f_higher)
                    all_floors_mentioned.add(f_lower)
                    floors_that_are_above.add(f_higher)
                # else: ignore malformed 'above' fact

        self.floor_rank = {}
        self.rank_floor = {}

        if all_floors_mentioned:
            # Find the bottom floor: a floor that appears as f_lower but never as f_higher
            bottom_floor = None
            floors_that_are_lower = set(is_immediately_above.keys())
            potential_bottom_floors = floors_that_are_lower - floors_that_are_above

            # If there's exactly one such floor, that's our bottom
            if len(potential_bottom_floors) == 1:
                 bottom_floor = potential_bottom_floors.pop()
            # If there are multiple or zero, or if there are floors not in 'above' facts,
            # try to find a floor that is a lower floor but not a higher floor among *all* floors mentioned.
            # This handles cases like a single floor problem or disconnected floors (though invalid PDDL).
            elif not potential_bottom_floors:
                 # Could be a single floor, or all floors are part of a chain/cycle
                 # If there's only one floor total, it's the bottom (and top)
                 if len(all_floors_mentioned) == 1:
                     bottom_floor = list(all_floors_mentioned)[0]
                 # Otherwise, fallback to sorting if we can't find a clear bottom
                 # This might happen with complex or malformed 'above' structures
                 else:
                     # print("Warning: Could not determine unique bottom floor from 'above' facts. Falling back to sorting.")
                     # Attempt numeric sort if floors are like f1, f2, ...
                     try:
                         sorted_floors = sorted(list(all_floors_mentioned), key=lambda f: int(f[1:]) if f.startswith('f') and f[1:].isdigit() else f)
                     except ValueError:
                         # Fallback to alphabetical sort if numeric fails
                         # print("Warning: Numeric floor sorting failed. Falling back to alphabetical sort.")
                         sorted_floors = sorted(list(all_floors_mentioned))

                     self.floor_rank = {f: i + 1 for i, f in enumerate(sorted_floors)}
                     self.rank_floor = {i + 1: f for i, f in enumerate(sorted_floors)}
                     bottom_floor = None # Indicate fallback was used


            if bottom_floor is not None:
                # Build ranks starting from the bottom floor
                current_floor = bottom_floor
                rank = 1
                # Use a visited set to detect cycles and stop
                visited_floors = set()
                while current_floor is not None and current_floor not in visited_floors:
                    visited_floors.add(current_floor)
                    self.floor_rank[current_floor] = rank
                    self.rank_floor[rank] = current_floor
                    rank += 1
                    current_floor = is_immediately_above.get(current_floor) # Get floor above current

                # Add any floors not part of the main chain (e.g., disconnected floors) - unlikely in valid PDDL
                remaining_floors = all_floors_mentioned - set(self.floor_rank.keys())
                if remaining_floors:
                     # print(f"Warning: Found disconnected floors not in main chain: {remaining_floors}. Assigning arbitrary ranks.")
                     # Assign ranks after the main chain, sorted
                     try:
                         sorted_remaining = sorted(list(remaining_floors), key=lambda f: int(f[1:]) if f.startswith('f') and f[1:].isdigit() else f)
                     except ValueError:
                         sorted_remaining = sorted(list(remaining_floors))

                     start_rank = len(self.floor_rank) + 1
                     for i, floor in enumerate(sorted_remaining):
                         self.floor_rank[floor] = start_rank + i
                         self.rank_floor[start_rank + i] = floor


        # Store passenger destinations and collect all passenger names
        self.passenger_destin = {}
        self.all_passengers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'destin':
                 if len(parts) == 3:
                    person, floor = parts[1], parts[2]
                    self.passenger_destin[person] = floor
                    self.all_passengers.add(person)
                 # else: ignore malformed 'destin' fact

        # Add passengers mentioned in initial 'origin' facts if not already added
        # This ensures we know about passengers even if they don't have a 'destin' fact (invalid PDDL)
        # or if 'destin' facts are not in static (unlikely for miconic)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'origin':
                 if len(parts) == 3:
                    self.all_passengers.add(parts[1])
                 # else: ignore malformed 'origin' fact


    def get_floor_rank(self, floor_name):
        """Helper to get the rank of a floor, returning a large number if floor is unknown."""
        # Return a large number for unknown floors so they don't affect min/max rank calculations
        # in a way that makes sense (they are effectively unreachable/irrelevant stops)
        return self.floor_rank.get(floor_name, float('inf'))

    def get_distance(self, floor1, floor2):
        """Calculate the distance (number of moves) between two floors."""
        rank1 = self.get_floor_rank(floor1)
        rank2 = self.get_floor_rank(floor2)
        if rank1 == float('inf') or rank2 == float('inf'):
            # This indicates an issue (e.g., trying to go to/from an unknown floor)
            # For heuristic purposes, a very large distance is appropriate.
            return float('inf')
        return abs(rank1 - rank2)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        lift_curr_floor = None
        waiting_passengers_in_state = set() # Passengers with (origin p f) in state
        boarded_passengers_in_state = set() # Passengers with (boarded p) in state
        served_passengers_in_state = set()  # Passengers with (served p) in state

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'lift-at':
                if len(parts) == 2:
                    lift_curr_floor = parts[1]
            elif predicate == 'origin':
                 if len(parts) == 3:
                    waiting_passengers_in_state.add(parts[1])
            elif predicate == 'boarded':
                 if len(parts) == 2:
                    boarded_passengers_in_state.add(parts[1])
            elif predicate == 'served':
                 if len(parts) == 2:
                    served_passengers_in_state.add(parts[1])

        # Identify unserved passengers based on the full set of passengers
        unserved_passengers = self.all_passengers - served_passengers_in_state

        # If all passengers are served, the goal is reached, heuristic is 0.
        if not unserved_passengers:
            return 0

        # Identify stop floors required for unserved passengers
        stop_floors = set()
        passengers_currently_waiting_unserved = set() # Subset of unserved who are waiting
        passengers_currently_boarded_unserved = set() # Subset of unserved who are boarded

        for p in unserved_passengers:
            if p in waiting_passengers_in_state:
                # Passenger is waiting at origin, need to visit origin floor
                # Find origin floor from state facts (it must be there if (origin p f) is true)
                origin_floor = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts and parts[0] == 'origin' and len(parts) == 3 and parts[1] == p:
                        origin_floor = parts[2]
                        break
                if origin_floor: # Should always find it if p is in waiting_passengers_in_state
                    stop_floors.add(origin_floor)
                    passengers_currently_waiting_unserved.add(p)
            elif p in boarded_passengers_in_state:
                # Passenger is boarded, need to visit destination floor
                destin_floor = self.passenger_destin.get(p)
                if destin_floor: # Should always find it if PDDL is valid
                    stop_floors.add(destin_floor)
                    passengers_currently_boarded_unserved.add(p)
                # else: Passenger boarded but no destination? Invalid PDDL.

        # Calculate minimum travel cost to visit stop floors
        min_travel_cost = 0
        if stop_floors:
            # Get ranks of stop floors, filtering out any unknown floors
            valid_stop_ranks = [self.get_floor_rank(f) for f in stop_floors if f in self.floor_rank]

            if valid_stop_ranks: # Ensure there are valid floors to stop at
                min_rank = min(valid_stop_ranks)
                max_rank = max(valid_stop_ranks)

                # Find the actual floors corresponding to min/max ranks
                min_stop_floor = self.rank_floor[min_rank]
                max_stop_floor = self.rank_floor[max_rank]

                # Distance to traverse the range of stop floors
                range_dist = self.get_distance(min_stop_floor, max_stop_floor)

                # Distance from current lift floor to the ends of the range
                # Ensure lift_curr_floor is known, otherwise travel cost is infinite (or large)
                if lift_curr_floor in self.floor_rank:
                    dist_to_min = self.get_distance(lift_curr_floor, min_stop_floor)
                    dist_to_max = self.get_distance(lift_curr_floor, max_stop_floor)

                    # Minimum travel cost is distance to reach one end + distance to traverse range
                    min_travel_cost = min(dist_to_min, dist_to_max) + range_dist
                else:
                    # Lift location is unknown or not a valid floor - problem state is likely invalid
                    # Return a very high heuristic value to prune this state
                    return float('inf')
            # else: stop_floors had elements, but none were valid floors (e.g., malformed facts)
            # This case is unlikely with valid PDDL, but if it happens, travel cost remains 0,
            # but action cost might still be > 0 if there are waiting/boarded passengers.
            # The heuristic will still be > 0 if there are unserved passengers.

        # Calculate actions needed
        # Each waiting passenger needs 1 board + 1 depart = 2 actions
        # Each boarded passenger needs 1 depart = 1 action
        actions_cost = len(passengers_currently_waiting_unserved) * 2 + len(passengers_currently_boarded_unserved) * 1

        # Total heuristic is travel cost + action cost
        total_cost = min_travel_cost + actions_cost

        return total_cost
