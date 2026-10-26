from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the cost by summing:
    1. A cost for each unserved goal passenger:
       - 2 actions (board + depart) if waiting at their origin.
       - 1 action (depart) if boarded.
    2. An estimated movement cost for the lift to visit all necessary floors.
       Necessary floors are the origins of waiting goal passengers and the
       destinations of boarded goal passengers. Movement cost is estimated
       as the distance from the current floor to the nearest necessary floor
       plus the vertical range spanned by all necessary floors.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and goal locations.
        """
        super().__init__(task) # Call the base class constructor

        # 1. Map floors to ranks based on 'above' facts
        # (above f_above f_below) means f_above is directly above f_below
        above_facts_parts = [get_parts(fact) for fact in self.static if match(fact, "above", "*", "*")]
        all_floors = set()
        floor_above_map_inverse = {} # floor_below -> floor_above
        for _, f_above, f_below in above_facts_parts:
            all_floors.add(f_above)
            all_floors.add(f_below)
            floor_above_map_inverse[f_below] = f_above

        # Find the bottom floor: it's a floor in all_floors that is NOT a key in floor_above_map_inverse
        # (i.e., no floor is directly below it)
        bottom_floor = None
        floors_that_are_below_something = set(floor_above_map_inverse.keys())
        for floor in all_floors:
             if floor not in floors_that_are_below_something:
                 bottom_floor = floor
                 break

        # Handle case with only one floor or no floors (shouldn't happen in valid problems)
        if bottom_floor is None and all_floors:
             if len(all_floors) == 1:
                 bottom_floor = list(all_floors)[0]
             else:
                 # This indicates an issue with the above facts defining a linear order
                 # In a real scenario, you might want to log this or handle it more robustly.
                 # For this heuristic, we'll proceed, but results might be inaccurate.
                 print("Warning: Could not determine unique bottom floor from 'above' facts.")


        # Assign ranks by traversing upwards from the bottom floor
        self.floor_to_rank = {}
        self.rank_to_floor = {}
        current_floor = bottom_floor
        rank = 1
        while current_floor is not None:
            self.floor_to_rank[current_floor] = rank
            self.rank_to_floor[rank] = current_floor
            current_floor = floor_above_map_inverse.get(current_floor) # Get the floor above this one
            rank += 1
        self.num_floors = rank - 1 # Total number of floors

        # 2. Store goal locations for each passenger
        self.goal_locations = {}
        # Goal is (and (served p1) (served p2) ...). We need destin for these goal passengers.
        goal_passengers = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                  goal_passengers.add(get_parts(goal)[1])

        for passenger in goal_passengers:
            # Search static facts for (destin passenger ?f)
            destin_fact = next((fact for fact in self.static if match(fact, "destin", passenger, "*")), None)
            if destin_fact:
                 self.goal_locations[passenger] = get_parts(destin_fact)[2]
            else:
                 # This passenger is a goal but has no destination? Problem definition issue.
                 # print(f"Warning: Goal passenger {passenger} has no 'destin' fact in static.")
                 # This passenger cannot be served according to the domain.
                 # If they are not initially served, the goal is unreachable.
                 # We store None to indicate this issue.
                 self.goal_locations[passenger] = None


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        non_move_cost = 0
        required_floors = set()
        current_floor = None

        # Find current lift location
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # If lift location is unknown or floor ranking failed, the state is invalid/unhandleable
        if current_floor is None or not self.floor_to_rank or current_floor not in self.floor_to_rank:
             # print("Error: Lift location unknown or floor ranking failed.") # Avoid printing in heuristic calls
             return float('inf') # Indicate an invalid or unhandleable state

        current_rank = self.floor_to_rank[current_floor]

        # Iterate through all passengers mentioned in the goal
        for passenger, destin_floor in self.goal_locations.items():
            # If destination is unknown for a goal passenger, and they are not served, goal is unreachable.
            if destin_floor is None:
                 if f"(served {passenger})" not in state:
                      return float('inf')
                 else:
                      continue # Already served, ignore

            # Check if served
            if f"(served {passenger})" in state:
                continue # Already served

            # Check if waiting at origin
            origin_floor = None
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    origin_floor = get_parts(fact)[2]
                    break

            if origin_floor:
                # Passenger is waiting at origin
                non_move_cost += 2 # Needs board + depart actions
                required_floors.add(origin_floor)
                continue # Move to next passenger

            # Check if boarded
            if f"(boarded {passenger})" in state:
                # Passenger is boarded
                non_move_cost += 1 # Needs depart action
                required_floors.add(destin_floor)
                continue # Move to next passenger

            # If a goal passenger is not served, not at origin, and not boarded,
            # this state is invalid according to the domain transitions from a valid initial state.
            # We assume valid states are passed to the heuristic. If this case occurs,
            # it indicates a problem state, so return infinity.
            # print(f"Warning: Goal passenger {passenger} in unexpected state.") # Avoid printing
            # return float('inf') # Consider returning infinity for invalid states


        # If no required floors, all relevant passengers are served.
        # This check is redundant if self.goals <= state is checked first,
        # but provides robustness if goal definition differs or state is partial.
        if not required_floors:
             return 0


        # Calculate movement cost
        required_ranks = set()
        for f in required_floors:
             rank = self.floor_to_rank.get(f)
             if rank is None:
                  # Should not happen if required_floors come from valid facts
                  # print(f"Error: Unknown required floor '{f}'.") # Avoid printing
                  return float('inf') # Indicate invalid floor
             required_ranks.add(rank)

        if not required_ranks: # Should be covered by the required_floors check above
             return non_move_cost # No movement needed if no required floors

        min_req_rank = min(required_ranks)
        max_req_rank = max(required_ranks)

        # Estimate movement cost: distance to the nearest required floor + the vertical range
        # This assumes the lift goes to one end of the required range and sweeps across.
        dist_to_min = abs(current_rank - min_req_rank)
        dist_to_max = abs(current_rank - max_req_rank)
        moves_to_nearest_end = min(dist_to_min, dist_to_max)

        moves_to_traverse_range = max_req_rank - min_req_rank

        move_cost = moves_to_nearest_end + moves_to_traverse_range

        return non_move_cost + move_cost

