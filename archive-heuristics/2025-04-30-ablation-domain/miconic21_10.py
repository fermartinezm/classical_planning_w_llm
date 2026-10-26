from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import sys

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    This heuristic estimates the cost to serve all passengers by summing:
    1. The estimated minimum travel distance for the lift to visit all necessary floors (origin floors for waiting passengers, destination floors for boarded passengers).
    2. The number of 'board' actions needed (equal to the number of waiting passengers).
    3. The number of 'depart' actions needed (equal to the number of boarded passengers).

    # Assumptions
    - The floor structure is linear and defined by `(above f_higher f_lower)` facts.
    - Unserved passengers are either waiting at their origin or boarded in the lift.
    - The cost of each action (move, board, depart) is 1.

    # Heuristic Initialization
    - Parses initial state and static facts to identify all floor names.
    - Parses `(above ...)` facts to determine the linear order of floors and create mappings between floor names and numerical indices. Includes fallback to sorted order if the `above` facts don't form a clear chain.
    - Parses `(destin ...)` facts from static information to store the destination floor for each passenger. Also collects all passenger names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify all passengers who are not yet served by checking for the `(served ?p)` predicate.
    3. If no passengers are unserved, the heuristic is 0.
    4. For each unserved passenger, determine if they are waiting at their origin floor (`(origin ?p ?f)`) or are boarded in the lift (`(boarded ?p)`).
    5. Collect the set of unique floors that the lift *must* visit:
       - The origin floor for every waiting passenger.
       - The destination floor for every boarded passenger.
       Let this set be `F_needed`.
    6. If `F_needed` is empty:
       - This implies all unserved passengers are currently boarded and their destination is the lift's current floor.
       - The only remaining actions are 'depart' for each of these passengers.
       - The heuristic value is the number of such passengers (cost of depart actions).
    7. If `F_needed` is not empty:
       - Get the numerical index for the current lift floor and all floors in `F_needed`.
       - Find the minimum and maximum floor indices among the floors in `F_needed`.
       - Calculate the estimated travel cost for the lift to cover the range of floors from the minimum needed index to the maximum needed index, starting from the current lift floor index. This is estimated as `min(abs(current_idx - min_needed_idx), abs(current_idx - max_needed_idx)) + (max_needed_idx - min_needed_idx)`. This represents the distance to one extreme of the needed range plus the distance to traverse the entire range.
       - Count the number of passengers currently waiting (`num_waiting`). Each needs a 'board' action.
       - Count the number of passengers currently boarded (`num_boarded`). Each needs a 'depart' action.
       - The total heuristic value is the sum of the estimated travel cost, `num_waiting`, and `num_boarded`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals

        # 1. Collect all unique floor names from initial state and static facts
        all_floor_names = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) > 1:
                all_floor_names.add(parts[1])
            elif parts and parts[0] == "origin" and len(parts) > 2:
                 all_floor_names.add(parts[2])
            elif parts and parts[0] == "destin" and len(parts) > 2: # Destin might be in initial state too
                 all_floor_names.add(parts[2])

        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) > 2:
                all_floor_names.add(parts[1]) # f_higher
                all_floor_names.add(parts[2]) # f_lower
            elif parts and parts[0] == "origin" and len(parts) > 2: # Origin might be in static too
                 all_floor_names.add(parts[2])
            elif parts and parts[0] == "destin" and len(parts) > 2:
                 all_floor_names.add(parts[2])

        if not all_floor_names:
             print("Warning: No floors found in initial state or static facts.", file=sys.stderr)
             self.floor_to_index = {}
             self.index_to_floor = {}
             self.all_passengers = set()
             self.goal_destinations = {}
             return # Cannot initialize heuristic properly

        # 2. Build higher_to_lower map from (above ...) facts
        above_facts = [get_parts(fact) for fact in task.static if match(fact, "above", "*", "*")]
        higher_to_lower = {}
        for _, f_higher, f_lower in above_facts:
            if f_higher in all_floor_names and f_lower in all_floor_names:
                 higher_to_lower[f_higher] = f_lower
            else:
                 print(f"Warning: (above {f_higher} {f_lower}) mentions unknown floors.", file=sys.stderr)


        # 3. Find the highest floor
        highest_floor = None
        # A floor is highest if it's in all_floor_names but is not a value in higher_to_lower
        potential_highest = all_floor_names - set(higher_to_lower.values())

        ordered_floors = []
        if len(potential_highest) == 1:
             highest_floor = list(potential_highest)[0]
             # 4. Build ordered floor list by traversing down from the highest floor
             current_floor = highest_floor
             while current_floor is not None and current_floor in all_floor_names:
                 ordered_floors.append(current_floor)
                 current_floor = higher_to_lower.get(current_floor)
             ordered_floors.reverse() # Reverse to get lowest to highest

        if len(ordered_floors) != len(all_floor_names):
             # Fallback: Assume sorted alphabetical order corresponds to floor order
             print(f"Warning: Built floor chain ({len(ordered_floors)}) does not include all floors ({len(all_floor_names)}). Using sorted order.", file=sys.stderr)
             ordered_floors = sorted(list(all_floor_names))
             print(f"Using sorted floor order: {ordered_floors}", file=sys.stderr)

        self.floor_to_index = {f: i for i, f in enumerate(ordered_floors)}
        self.index_to_floor = {i: f for i, f in enumerate(ordered_floors)}

        # 5. Load passenger destinations and passengers
        self._load_destinations_and_passengers(task.static, task.initial_state, task.goals)

    def _load_destinations_and_passengers(self, static_facts, initial_state, goals):
        """Helper to load destinations and passenger names."""
        self.goal_destinations = {}
        self.all_passengers = set()

        # Destinations are typically static
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, p, f_destin = get_parts(fact)
                self.goal_destinations[p] = f_destin
                self.all_passengers.add(p)

        # Origins are dynamic, found in the initial state
        for fact in initial_state:
             if match(fact, "origin", "*", "*"):
                 _, p, _ = get_parts(fact)
                 self.all_passengers.add(p)

        # Passengers in goals must also be considered (ensures we know about all passengers)
        for goal in goals:
             if match(goal, "served", "*"):
                 _, p = get_parts(goal)
                 self.all_passengers.add(p)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Handle case where floor indexing failed during init
        if not self.floor_to_index:
             print("Error: Floor indexing failed during initialization.", file=sys.stderr)
             return float('inf')

        # 1. Find current lift floor
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, lift_floor = get_parts(fact)
                break

        if lift_floor is None or lift_floor not in self.floor_to_index:
             # Should not happen in valid states, but handle defensively
             print(f"Error: Lift location '{lift_floor}' not found or unknown in state!", file=sys.stderr)
             return float('inf')

        current_idx = self.floor_to_index[lift_floor]

        # 2. Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = self.all_passengers - served_passengers

        # 3. If all passengers are served, heuristic is 0
        if not unserved_passengers:
            return 0

        # 4. Collect needed floors and count waiting/boarded passengers
        F_pickup = set()
        F_dropoff = set()
        num_waiting = 0
        num_boarded = 0

        for p in unserved_passengers:
            is_waiting = False
            # Check state for origin or boarded predicate for this passenger
            for fact in state:
                if match(fact, "origin", p, "*"):
                    _, _, origin_floor = get_parts(fact)
                    if origin_floor in self.floor_to_index:
                        F_pickup.add(origin_floor)
                        num_waiting += 1
                        is_waiting = True
                    else:
                         print(f"Warning: Origin floor '{origin_floor}' for passenger {p} is unknown.", file=sys.stderr)
                         # This passenger might be unservable with known floors.
                         # Ignore for heuristic calculation, or assign high cost?
                         # Let's ignore for now.
                    break # Passenger is waiting, no need to check for boarded

            if not is_waiting:
                # If not waiting and unserved, must be boarded (assuming valid states)
                dest_floor = self.goal_destinations.get(p)
                if dest_floor and dest_floor in self.floor_to_index:
                    F_dropoff.add(dest_floor)
                    num_boarded += 1
                else:
                     # Should not happen in valid problems or if dest is known but floor is unknown
                     print(f"Warning: Destination floor '{dest_floor}' for boarded passenger {p} is unknown or not found.", file=sys.stderr)
                     # Ignore this passenger for heuristic calculation.
                     pass


        F_needed = F_pickup.union(F_dropoff)

        # 5. Handle case where F_needed is empty (all unserved are boarded at current floor)
        if not F_needed:
             # This means all unserved passengers are boarded, AND their destination
             # is the current lift_floor. The only remaining actions are 'depart'.
             # num_boarded already counts these passengers.
             return num_boarded # Cost is number of depart actions

        # 6. Calculate travel cost
        needed_indices = {self.floor_to_index[f] for f in F_needed}

        min_needed_idx = min(needed_indices)
        max_needed_idx = max(needed_indices)

        # Estimated travel to cover the range [min_needed_idx, max_needed_idx] starting from current_idx
        # This is the distance to the closest extreme + the distance between extremes
        travel_cost = min(abs(current_idx - min_needed_idx), abs(current_idx - max_needed_idx)) + (max_needed_idx - min_needed_idx)

        # 7. Calculate total heuristic
        # Total cost = Travel cost + Actions needed (board + depart)
        total_cost = travel_cost + num_waiting + num_boarded

        return total_cost
