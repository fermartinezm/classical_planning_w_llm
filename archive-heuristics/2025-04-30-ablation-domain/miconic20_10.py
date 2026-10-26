from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like "(predicate)" with no arguments
    content = fact[1:-1].strip()
    if not content:
        return [content] # Or perhaps just [] depending on desired behavior for empty facts
    return content.split()

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
    It sums the number of 'board' actions needed (for waiting passengers),
    the number of 'depart' actions needed (for boarded passengers), and
    an estimate of the minimum 'move' actions required for the lift to visit
    all necessary floors (origins of waiting passengers and destinations of boarded passengers).

    # Assumptions
    - The lift can carry multiple passengers.
    - Floors are ordered sequentially. The 'above' facts define this order.
    - Each 'move-up' or 'move-down' action moves the lift exactly one floor.
    - 'board' and 'depart' actions take 1 unit of cost. 'move' actions take 1 unit of cost per floor moved.
    - Floor names are structured such that sorting them based on the integer part after 'f' gives the correct order (e.g., f1, f2, f10).

    # Heuristic Initialization
    - Extracts static facts: passenger destinations and floor ordering ('above' relations).
    - Builds a mapping from floor names to their numerical index based on the 'above' relations.
    - Stores passenger destinations in a dictionary.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current location of the lift from the state facts.
    2. Get the numerical index for the current lift floor using the pre-calculated floor index map.
    3. Initialize counters for needed 'board' actions (`num_waiting`) and 'depart' actions (`num_boarded`).
    4. Initialize a set to store the floors the lift *must* stop at (`required_stop_floors`).
    5. Identify all passengers mentioned in the goal state (those who need to be served).
    6. For each passenger identified in step 5:
       - Check if the passenger is currently in the state as `(origin p f_o)`. If so, increment `num_waiting` and add `f_o` to `required_stop_floors`.
       - Check if the passenger is currently in the state as `(boarded p)`. If so, increment `num_boarded` and add their destination floor `f_d` (looked up from pre-calculated destinations) to `required_stop_floors`.
       - If the passenger is in the state as `(served p)`, they are done and do not contribute to these counts or required stops.
    7. If the set of `required_stop_floors` is empty, it means all relevant passengers are either served or at the lift's current location and need to go to the current location. In a valid problem, this implies all passengers are served, so the heuristic is 0.
    8. If `required_stop_floors` is not empty, calculate the estimated move cost:
       - Find the minimum and maximum numerical floor indices among the `required_stop_floors`.
       - The estimated move cost is the distance between the minimum and maximum required floor indices (`max_idx - min_idx`), plus the minimum distance from the lift's current floor index (`idx_c`) to either the minimum or maximum required floor index (`min(abs(idx_c - min_idx), abs(idx_c - max_idx))`). This estimates the travel needed to reach the range of required floors and then traverse that range.
    9. The total heuristic value is the sum of `num_waiting`, `num_boarded`, and the estimated move cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Passenger destinations.
        - Floor ordering from 'above' facts.
        """
        self.goals = task.goals  # Goal conditions (e.g., (served p1), (served p2), ...)
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract passenger destinations from static facts
        self.destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor

        # Build floor ordering and index mapping from 'above' facts
        # Assume floor names are like 'f1', 'f2', 'f10', etc., and the number indicates the order.
        all_floors = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f1, f2 = get_parts(fact)
                all_floors.add(f1)
                all_floors.add(f2)

        # Sort floors based on the integer part of their name (e.g., f1 -> 1, f10 -> 10)
        # Assign 1-based index according to this sorted order.
        sorted_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
        self.floor_indices = {floor: index + 1 for index, floor in enumerate(sorted_floors)}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break # Assuming only one lift-at fact

        if current_lift_floor is None:
             # Should not happen in a valid miconic state
             # If lift location is unknown, cannot estimate moves.
             # Return a high value or number of unserved passengers as a fallback.
             # Let's count unserved passengers as a minimum estimate.
             unserved_count = sum(1 for goal in self.goals if goal not in state)
             return unserved_count if unserved_count > 0 else 0


        idx_c = self.floor_indices[current_lift_floor]

        num_waiting = 0
        num_boarded = 0
        required_stop_floors = set()

        # Identify passengers who need to be served (from goal state)
        passengers_to_serve = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Track current state of relevant passengers
        passenger_current_state = {} # Map passenger to their state predicate (origin, boarded, served)
        passenger_origin_floor = {} # Map passenger to origin floor if waiting

        for fact in state:
            parts = get_parts(fact)
            if len(parts) >= 2 and parts[1] in passengers_to_serve:
                 predicate = parts[0]
                 passenger = parts[1]
                 passenger_current_state[passenger] = predicate
                 if predicate == "origin":
                     # Store origin floor for waiting passengers
                     if len(parts) >= 3:
                         passenger_origin_floor[passenger] = parts[2]


        # Iterate through passengers who need to be served
        for passenger in passengers_to_serve:
            state_predicate = passenger_current_state.get(passenger)

            if state_predicate == "served":
                # Passenger is already served, contributes 0 to heuristic
                continue

            elif state_predicate == "origin":
                # Passenger is waiting at origin floor f_o
                num_waiting += 1
                f_o = passenger_origin_floor.get(passenger)
                if f_o: # Ensure origin floor was found
                    required_stop_floors.add(f_o)
                else:
                    # Invalid state: passenger has origin predicate but no floor?
                    # Or origin fact wasn't in state? Assume valid states.
                    pass

            elif state_predicate == "boarded":
                # Passenger is boarded, needs to go to destination floor f_d
                num_boarded += 1
                f_d = self.destinations.get(passenger)
                if f_d: # Ensure destination exists
                    required_stop_floors.add(f_d)
                else:
                    # Invalid state: boarded passenger with no destination?
                    # Assume valid problems.
                    pass
            else:
                 # Passenger is unserved but not origin/boarded? (e.g., not in initial state)
                 # Assume valid states where unserved passengers are either origin or boarded.
                 pass


        # Calculate estimated move cost
        move_cost = 0
        if required_stop_floors:
            # Ensure all required floors are in our index map (should be if problem is valid)
            valid_required_indices = [self.floor_indices[f] for f in required_stop_floors if f in self.floor_indices]

            if valid_required_indices:
                min_idx = min(valid_required_indices)
                max_idx = max(valid_required_indices)

                # Cost to reach the range [min_idx, max_idx] from idx_c
                distance_to_min = abs(idx_c - min_idx)
                distance_to_max = abs(idx_c - max_idx)
                distance_to_range = min(distance_to_min, distance_to_max)

                # Cost to sweep the range
                range_sweep_cost = max_idx - min_idx

                move_cost = range_sweep_cost + distance_to_range
            else:
                 # Should not happen if required_stop_floors is not empty and problem is valid
                 pass


        # Total heuristic is sum of board actions, depart actions, and estimated moves
        total_heuristic = num_waiting + num_boarded + move_cost

        return total_heuristic

