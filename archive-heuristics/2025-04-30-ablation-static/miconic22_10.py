from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy base class if not running in the specific environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
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
    This heuristic estimates the required number of actions (board, depart, and move)
    to serve all passengers that are part of the goal. It sums the number of
    board and depart actions needed for unserved passengers and adds an estimate
    of the minimum vertical movement required to visit all floors where unserved
    passengers are waiting or need to be dropped off.

    # Assumptions
    - Floors are named 'f<number>' (e.g., f1, f2, f10) and are ordered numerically
      (f1 is the lowest, f2 is next, etc.). The `(above f_higher f_lower)` predicate
      confirms this numerical ordering corresponds to vertical position.
    - Passenger destinations are static and defined in the initial state/goal.
    - The goal is to serve a specific set of passengers.

    # Heuristic Initialization
    - Extracts all floor objects and determines their order by sorting based on
      the numerical suffix in their names, storing it as a mapping from floor
      name to index.
    - Extracts the destination floor for each passenger that needs to be served
      according to the task goals, storing it in a dictionary.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Current Lift Location:** Find the floor where the lift is currently located using the `(lift-at ?f)` fact.
    2.  **Identify Unserved Passengers:** Determine which passengers included in the task's goal
        conditions (`(served ?p)`) have not yet been served in the current state. If all goal passengers
        are served, the heuristic is 0.
    3.  **Count Non-Move Actions:**
        -   For each unserved passenger who is currently waiting at an origin floor
            (`(origin p f)` is true in the current state), count 1 `board` action needed.
        -   For each unserved passenger who is currently boarded (`(boarded p)` is true in the current state),
            count 1 `depart` action needed.
        -   The total non-move cost is the sum of these board and depart actions.
    4.  **Identify Floors to Visit:** Determine the set of floors the lift must visit
        to make progress on unserved passengers:
        -   Include the origin floor for every unserved passenger who is currently waiting.
        -   Include the destination floor for every unserved passenger who is currently boarded.
    5.  **Estimate Move Actions:**
        -   If there are no floors to visit, the move cost is 0.
        -   If there are floors to visit, find the lowest and highest floor indices among
            the needed floors using the pre-calculated floor order. Let these be `min_idx` and `max_idx`.
        -   Let `current_idx` be the index of the current lift floor.
        -   The lift must traverse the vertical span between `min_idx` and `max_idx`,
            which requires at least `max_idx - min_idx` moves.
        -   Additionally, the lift must travel from its current floor (`current_idx`)
            to reach this span of floors and potentially travel back and forth within the span.
            -   If `current_idx` is below `min_idx`, it must move up at least `min_idx - current_idx` floors to reach the span. Total moves = `(min_idx - current_idx) + (max_idx - min_idx) = max_idx - current_idx`.
            -   If `current_idx` is above `max_idx`, it must move down at least `current_idx - max_idx` floors to reach the span. Total moves = `(current_idx - max_idx) + (max_idx - min_idx) = current_idx - min_idx`.
            -   If `current_idx` is within the span `[min_idx, max_idx]`, it must still cover the span `max_idx - min_idx`. To visit floors both below and above `current_idx` within the span, it must travel from `current_idx` to `min_idx` and then to `max_idx` (or vice versa). The minimum additional travel to cover the span from within is `min(current_idx - min_idx, max_idx - current_idx)`. Total moves = `(max_idx - min_idx) + min(current_idx - min_idx, max_idx - current_idx)`.
    6.  **Sum Costs:** The total heuristic value is the sum of the non-move actions and the estimated move actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.task = task # Store task for access to initial_state and goals

        # 1. Extract and sort floors
        floor_names = set()
        # Floors appear in lift-at, origin, destin, above predicates
        for fact in task.initial_state | task.static:
             parts = get_parts(fact)
             if parts:
                 if parts[0] in ["lift-at"]:
                     if len(parts) > 1: floor_names.add(parts[1])
                 elif parts[0] in ["origin", "destin"]:
                     if len(parts) > 2: floor_names.add(parts[2])
                 elif parts[0] == "above":
                     if len(parts) > 2:
                         floor_names.add(parts[1])
                         floor_names.add(parts[2])


        # Sort floors based on numerical suffix (assuming f1, f2, ...)
        # This assumes f1 is the lowest floor, f2 is the next, etc.
        # The (above f_higher f_lower) predicate confirms this ordering.
        try:
            sorted_floors = sorted(list(floor_names), key=lambda f: int(f[1:]))
        except (ValueError, IndexError):
             # Fallback if floor names are not strictly f<number> format
             # This might not be correct if naming is inconsistent, but handles basic cases
             print(f"Warning: Floor names {floor_names} not strictly in f<number> format. Sorting alphabetically.")
             sorted_floors = sorted(list(floor_names))


        self.floor_indices = {floor: i for i, floor in enumerate(sorted_floors)}

        # 2. Extract passenger destinations from initial state/goals
        self.passenger_destinations = {}
        # Goal is (served p), need to find (destin p f) in initial state
        self.goal_passengers = {get_parts(goal)[1] for goal in task.goals if match(goal, "served", "*")}

        for fact in task.initial_state:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    p, f = parts[1:]
                    if p in self.goal_passengers:
                         self.passenger_destinations[p] = f


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Current Lift Location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown, state is likely invalid or initial parsing failed
        if current_lift_floor is None or current_lift_floor not in self.floor_indices:
             return float('inf') # Indicate an unreachable or invalid state

        current_idx = self.floor_indices[current_lift_floor]


        # 2. Identify Unserved Passengers (among those in the goal)
        unserved_passengers = {
            p for p in self.goal_passengers
            if "(served {})".format(p) not in state # Direct string check is faster than match here
        }

        if not unserved_passengers:
            return 0 # Goal reached for all relevant passengers

        # 3. Count Non-Move Actions and Identify Floors to Visit
        board_actions = 0
        depart_actions = 0
        floors_to_visit = set()

        # Create quick lookups for origin and boarded facts in the current state
        state_origins = {} # {passenger: floor}
        state_boarded = set() # {passenger}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "origin" and len(parts) == 3:
                    state_origins[parts[1]] = parts[2]
                elif parts[0] == "boarded" and len(parts) == 2:
                    state_boarded.add(parts[1])

        for p in unserved_passengers:
            is_waiting = p in state_origins
            is_boarded = p in state_boarded

            if is_waiting:
                board_actions += 1
                floors_to_visit.add(state_origins[p])

            elif is_boarded:
                depart_actions += 1
                destin_floor = self.passenger_destinations.get(p)
                if destin_floor: # Ensure destination is known
                    floors_to_visit.add(destin_floor)

            # Note: A passenger cannot be both waiting and boarded in a valid state.
            # If they are neither waiting nor boarded, but unserved, something is wrong
            # with the state representation or domain logic (e.g., passenger vanished).
            # We assume valid states where unserved passengers are either waiting or boarded.


        total_non_move_actions = board_actions + depart_actions

        # 4. Floors to Visit (already collected in step 3)

        # 5. Estimate Move Actions
        move_actions = 0
        if floors_to_visit:
            # Filter out any floors not found during initialization (shouldn't happen)
            needed_indices = [self.floor_indices[f] for f in floors_to_visit if f in self.floor_indices]

            if needed_indices: # Ensure there are valid floors to visit
                min_idx = min(needed_indices)
                max_idx = max(needed_indices)

                if current_idx < min_idx:
                    # Must go up to at least min_idx, then cover the span up to max_idx
                    move_actions = (min_idx - current_idx) + (max_idx - min_idx)
                elif current_idx > max_idx:
                    # Must go down to at least max_idx, then cover the span down to min_idx
                    move_actions = (current_idx - max_idx) + (max_idx - min_idx)
                else: # current_idx is within [min_idx, max_idx]
                    # Must cover the span (max_idx - min_idx)
                    # Plus travel from current to one end (min or max) and then to the other
                    move_actions = (max_idx - min_idx) + min(current_idx - min_idx, max_idx - current_idx)
            # else: floors_to_visit was not empty, but contained no recognized floors.
            # This case implies a parsing issue or invalid state. Heuristic remains 0 if needed_indices is empty.
            # A large value might be better here? But if floors_to_visit is non-empty, there's work.
            # Let's assume valid floors are always present if floors_to_visit is non-empty.


        # 6. Sum Costs
        return total_non_move_actions + move_actions
