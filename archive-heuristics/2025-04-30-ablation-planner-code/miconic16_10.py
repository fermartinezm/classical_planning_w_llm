from fnmatch import fnmatch
import math

# Assuming Heuristic base class is available from the planning framework
# from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums three components:
    1. Estimated movement cost for the lift to visit all necessary floors (origin floors for unboarded passengers, destination floors for boarded passengers).
    2. Number of unboarded passengers (each needs a 'board' action).
    3. Number of boarded passengers (each needs a 'depart' action).

    # Assumptions
    - Floors are totally ordered by the 'above' predicate, forming a single vertical tower.
    - Each unserved passenger requires exactly one 'board' and one 'depart' action.
    - The lift can carry multiple passengers simultaneously.
    - The movement cost is estimated by the minimum travel needed to cover the range of required floors, starting from the current floor.

    # Heuristic Initialization
    - Extracts the total ordering of floors from 'above' static facts. It identifies the lowest floor and builds mappings between floor names and numerical indices based on the 'above' chain.
    - Extracts the destination floor for each passenger from 'destin' static facts.
    - Identifies the set of all passengers from the goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift by finding the fact `(lift-at ?f)` in the state.
    2. Identify all passengers who are not yet 'served'. A passenger `p` is unserved if the fact `(served p)` is not present in the state.
    3. Determine the set of 'required floors'. A floor `f` is required if:
       - There is an unserved passenger `p` such that `(origin p f)` is true in the state (needs pickup).
       - There is an unserved passenger `p` such that `(boarded p)` is true in the state and `(destin p f)` is true (needs dropoff).
    4. If the set of required floors is empty, it means all relevant passengers are either served or are not in a state requiring the lift. In this case, the heuristic is 0.
    5. If there are required floors:
       - Map the current lift floor and all required floors to their numerical indices using the precomputed `floor_to_index` mapping.
       - Find the minimum index (`min_req_idx`) and maximum index (`max_req_idx`) among the required floor indices.
       - Calculate the estimated movement cost:
         - If there is only one required floor, the cost is the absolute difference between the current floor index and the required floor index.
         - If there are multiple required floors, the cost is estimated as the minimum distance from the current floor index to either the minimum or maximum required floor index, plus the total span of the required floors (maximum index - minimum index). This represents the cost of traveling to one end of the required range and then sweeping across to the other end, ensuring all floors in between are visited.
    6. Count the number of unserved passengers who are currently at their origin floor (`(origin p f)` is true for some `f`). Each of these passengers requires a 'board' action.
    7. Count the number of unserved passengers who are currently 'boarded' (`(boarded p)` is true). Each of these passengers requires a 'depart' action.
    8. The total heuristic value is the sum of the estimated movement cost, the count from step 6 (board actions), and the count from step 7 (depart actions).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ordering, passenger destinations,
        and the set of all passengers.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build floor ordering and mapping
        floor_above = {}
        all_floors = set()
        floors_that_are_above = set() # Floors that appear as the second argument in (above f_lower f_upper)

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                f_lower, f_upper = parts[1], parts[2]
                floor_above[f_lower] = f_upper
                all_floors.add(f_lower)
                all_floors.add(f_upper)
                floors_that_are_above.add(f_upper)

        # Find the lowest floor: a floor that is in all_floors but not in floors_that_are_above
        lowest_floor = None
        if all_floors:
            potential_lowest = list(all_floors - floors_that_are_above)
            if len(potential_lowest) == 1:
                 lowest_floor = potential_lowest[0]
            elif len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0] # Case with only one floor
            # else: Could be disconnected components or invalid 'above' structure.
            #      We proceed, but floor_to_index might be incomplete.

        self.floor_to_index = {}
        self.index_to_floor = {}
        if lowest_floor:
            current_floor = lowest_floor
            index = 1
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                self.index_to_floor[index] = current_floor
                index += 1
                current_floor = floor_above.get(current_floor)
        # If lowest_floor is None, floor_to_index and index_to_floor remain empty.
        # The __call__ method will handle this by returning infinity if needed.


        # Extract passenger destinations
        self.passenger_destin = {}
        self.all_passengers = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin":
                p, f = parts[1], parts[2]
                self.passenger_destin[p] = f
                self.all_passengers.add(p)

        # Also get passengers from goal (in case they aren't in destin facts)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "served":
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if it's a goal state first
        if all(f'(served {p})' in state for p in self.all_passengers):
             return 0

        # If floor mapping failed during init (invalid 'above' structure) and not goal state
        if not self.floor_to_index and len(self.all_passengers) > 0:
             return float('inf') # Cannot estimate cost without floor structure

        # 1. Identify current lift floor
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # If state is not goal but missing lift location or current floor is unmapped
        if current_floor is None or current_floor not in self.floor_to_index:
             return float('inf') # Cannot estimate cost without lift location/mapping

        current_floor_idx = self.floor_to_index[current_floor]


        # 2. Identify unserved passengers and required floors
        unserved_passengers = [p for p in self.all_passengers if f'(served {p})' not in state]

        required_floors = set()
        num_unboarded = 0
        num_boarded = 0

        for p in unserved_passengers:
            is_origin = False
            is_boarded = False
            # Check if passenger is at origin
            for fact in state:
                if match(fact, "origin", p, "*"):
                    origin_floor = get_parts(fact)[2]
                    required_floors.add(origin_floor)
                    is_origin = True
                    break # Assume only one origin fact per passenger

            # Check if passenger is boarded
            if f'(boarded {p})' in state:
                 is_boarded = True

            if is_origin:
                 num_unboarded += 1
            elif is_boarded:
                 num_boarded += 1
                 # Add destination floor as required dropoff floor
                 destin_floor = self.passenger_destin.get(p)
                 if destin_floor: # Should always exist for unserved boarded passenger
                     required_floors.add(destin_floor)
                 # else: Problem: boarded passenger with no known destination?
                 #       Assume valid PDDL and destin exists.

        # 4. If no required floors, heuristic is 0 (already checked for goal state at the beginning)
        if not required_floors:
            # This case should only be reached if all passengers are served,
            # which is handled by the initial goal check.
            # If somehow unserved passengers exist but don't trigger required_floors,
            # it implies an invalid state structure for this heuristic's assumptions.
            # Returning 0 here might be misleading if not a goal state.
            # Let's rely on the initial goal check. If we are here, required_floors is not empty.
             pass


        # 5. Calculate movement cost
        required_floor_indices = []
        for f in required_floors:
            idx = self.floor_to_index.get(f)
            if idx is not None:
                required_floor_indices.append(idx)
            # else: Required floor not in mapping - problem with init or state.
            #       Handled by initial check for empty floor_to_index, but defensive here.
            #       If this happens, required_floor_indices might be empty or incomplete.

        if not required_floor_indices:
             # This should not happen if required_floors was not empty and floor_to_index is valid
             # Return infinity as we can't estimate movement
             return float('inf')

        min_req_idx = min(required_floor_indices)
        max_req_idx = max(required_floor_indices)

        if min_req_idx == max_req_idx:
            # Only one required floor
            movement_cost = abs(current_floor_idx - min_req_idx)
        else:
            # Multiple required floors
            # Cost to reach one extreme + cost to sweep to the other extreme
            dist_to_min = abs(current_floor_idx - min_req_idx)
            dist_to_max = abs(current_floor_idx - max_req_idx)
            span = max_req_idx - min_req_idx
            movement_cost = min(dist_to_min, dist_to_max) + span

        # 6. + 7. Add boarding and departing costs
        # num_unboarded and num_boarded are already counted above

        # 8. Total heuristic value
        total_cost = movement_cost + num_unboarded + num_boarded

        return total_cost
