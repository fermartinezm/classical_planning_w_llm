from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this import path

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
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
    This heuristic estimates the remaining cost by summing the number of required board/depart actions
    and an estimate of the necessary lift movement to visit all floors where actions are needed.

    # Assumptions
    - Floors are linearly ordered based on 'above' facts.
    - The lift can carry multiple passengers simultaneously.
    - Passengers must be picked up at their origin and dropped off at their destination.
    - All passengers listed in the goal must be served.

    # Heuristic Initialization
    - Build a mapping from floor names to numerical indices (0-based) based on the 'above' facts.
    - Store the destination floor for each passenger based on the 'destin' facts found in static information.
    - Identify all passengers who need to be served from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state (all passengers served). If yes, return 0.
    2. Find the current floor of the lift.
    3. Identify all passengers who are not yet 'served'.
    4. Initialize sets for floors needing pickups (`pickup_floors`) and floors needing dropoffs (`dropoff_floors`). Initialize action cost to 0.
    5. For each non-served passenger:
       - Check if the passenger is currently at their origin floor (using the 'origin' predicate). If yes:
         - Add their origin floor to `pickup_floors`.
         - Increment action cost by 1 (for the required 'board' action).
       - Check if the passenger is currently 'boarded'. If yes:
         - Find their destination floor using the pre-calculated destination mapping.
         - Add their destination floor to `dropoff_floors`.
         - Increment action cost by 1 (for the required 'depart' action).
       (Note: A non-served passenger must be either at their origin or boarded, based on domain actions).
    6. Collect all unique floors that need either a pickup or a dropoff (`F_needed`).
    7. Calculate the movement cost:
       - If `F_needed` is empty (meaning all non-served passengers are already at the lift's current floor and can perform their action immediately, or there are no non-served passengers left), the movement cost is 0.
       - If `F_needed` is not empty:
         - Convert the floor names in `F_needed` to their numerical indices.
         - Find the minimum (`min_needed_idx`) and maximum (`max_needed_idx`) floor indices among the needed floors.
         - Get the index of the current lift floor (`current_lift_floor_idx`).
         - The movement cost is estimated as the span of floors between the minimum and maximum needed floors (`max_needed_idx - min_needed_idx`), plus the minimum distance from the current lift floor to either the minimum or maximum needed floor (`min(abs(current_lift_floor_idx - min_needed_idx), abs(current_lift_floor_idx - max_needed_idx))`). This estimates the minimum travel distance to visit all floors in `F_needed` starting from the current floor.
    8. The total heuristic value is the sum of the calculated movement cost and the action cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = task.goals
        static_facts = task.static

        # Build floor mapping from 'above' facts
        self.floor_to_idx = {}
        self.idx_to_floor = {}
        above_facts_parsed = [get_parts(fact) for fact in static_facts if match(fact, "above", "*", "*")]

        # Find the lowest floor: a floor that is never the second argument of an 'above' fact
        all_floors = set()
        higher_floors = set()
        for _, f_lower, f_higher in above_facts_parsed:
            all_floors.add(f_lower)
            all_floors.add(f_higher)
            higher_floors.add(f_higher)

        lowest_floor = None
        # Check floors mentioned in 'above' facts first
        candidate_lowest = all_floors - higher_floors
        if candidate_lowest:
             # If there are multiple candidates (e.g., disconnected towers), pick one deterministically
             # assuming floor names are like f1, f2, ... and sort numerically
             try:
                 lowest_floor = sorted(list(candidate_lowest), key=lambda f: int(f[1:]))[0]
             except (ValueError, IndexError):
                 # Fallback if floor names are not in f<number> format
                 lowest_floor = sorted(list(candidate_lowest))[0]
        elif all_floors:
             # If no clear lowest floor from 'above' (e.g., single floor or cycle),
             # assume floors are named f1, f2, ... and sort numerically.
             try:
                 floor_names = sorted(list(all_floors), key=lambda f: int(f[1:]))
                 if floor_names:
                     lowest_floor = floor_names[0]
             except (ValueError, IndexError):
                 # Fallback if floor names are not in f<number> format
                 floor_names = sorted(list(all_floors))
                 if floor_names:
                     lowest_floor = floor_names[0]


        if lowest_floor:
            # Build the linear mapping starting from the lowest floor
            current_floor = lowest_floor
            idx = 0
            # Create a mapping from lower floor to higher floor for easy traversal
            above_map = {f_lower: f_higher for _, f_lower, f_higher in above_facts_parsed}

            while current_floor:
                self.floor_to_idx[current_floor] = idx
                self.idx_to_floor[idx] = current_floor
                idx += 1
                current_floor = above_map.get(current_floor)

        # Store passenger destinations
        self.passenger_to_dest = {}
        # Destinations are in static facts
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_to_dest[passenger] = floor

        # Identify all passengers from goals
        self.all_passengers = set()
        for goal in self.goals:
             # Goal is (served p)
             if match(goal, "served", "*"):
                 _, passenger = get_parts(goal)
                 self.all_passengers.add(passenger)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal state
        if all(f'(served {p})' in state for p in self.all_passengers):
            return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        # If no lift location or no floor mapping (problem setup issue), return infinity
        if current_lift_floor is None or current_lift_floor not in self.floor_to_idx:
             # This indicates a potentially malformed state or problem definition
             return float('inf')

        current_lift_floor_idx = self.floor_to_idx[current_lift_floor]

        # Identify non-served passengers and their required actions/floors
        pickup_floors = set()
        dropoff_floors = set()
        action_cost = 0

        for passenger in self.all_passengers:
            if f'(served {passenger})' not in state:
                is_boarded = f'(boarded {passenger})' in state
                is_at_origin = False
                origin_floor = None

                # Find if passenger is at origin
                # Iterate through state facts to find the origin fact for this passenger
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        _, _, origin_floor = get_parts(fact)
                        is_at_origin = True
                        break

                if is_at_origin: # Passenger is at origin, needs pickup
                    pickup_floors.add(origin_floor)
                    action_cost += 1 # Cost for board action
                elif is_boarded: # Passenger is boarded, needs dropoff
                    dest_floor = self.passenger_to_dest.get(passenger)
                    # Ensure destination exists and is a valid floor
                    if dest_floor and dest_floor in self.floor_to_idx:
                        dropoff_floors.add(dest_floor)
                        action_cost += 1 # Cost for depart action
                    # else: boarded passenger with no valid destination - potentially invalid state, ignore for heuristic
                # else: non-served passenger is neither at origin nor boarded.
                # This state shouldn't be reachable in a valid plan execution
                # if they started at origin and are not served.
                # We ignore such passengers for heuristic calculation assuming they don't need actions yet.


        # Combine needed floors
        needed_floors = list(pickup_floors.union(dropoff_floors))

        movement_cost = 0
        if needed_floors:
            # Filter out any needed floors that weren't in our floor mapping (e.g., from malformed input)
            valid_needed_floors = [f for f in needed_floors if f in self.floor_to_idx]
            if valid_needed_floors:
                needed_floor_indices = [self.floor_to_idx[f] for f in valid_needed_floors]
                min_needed_idx = min(needed_floor_indices)
                max_needed_idx = max(needed_floor_indices)

                # Movement estimate: span + min distance from current floor to ends of span
                span = max_needed_idx - min_needed_idx
                dist_to_min = abs(current_lift_floor_idx - min_needed_idx)
                dist_to_max = abs(current_lift_floor_idx - max_needed_idx)
                movement_cost = span + min(dist_to_min, dist_to_max)
            # If needed_floors was not empty but none were in floor_to_idx, movement_cost remains 0.
            # This is unlikely with valid PDDL.

        # Total heuristic is sum of movement and action costs
        total_heuristic = movement_cost + action_cost

        return total_heuristic
