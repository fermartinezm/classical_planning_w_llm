from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the necessary board and depart actions for unserved passengers
    and adds an estimate of the minimum lift movement actions needed to visit
    all relevant floors (origins of waiting passengers, destinations of boarded passengers).

    # Assumptions
    - Floors are linearly ordered by the `above` predicate, forming a single chain.
    - Each waiting passenger requires one board and one depart action.
    - Each boarded passenger requires one depart action.
    - The lift can carry multiple passengers.
    - Movement cost between adjacent floors is 1.
    - The minimum movement to visit a set of floors starting from the current floor
      is the distance to the nearest extreme of the required floor range plus the
      distance to traverse the entire required floor range.

    # Heuristic Initialization
    - Parses the `above` facts from static information to create a mapping from
      floor names to integer levels. Assumes floors form a single linear structure.
      If the floor structure is not a single chain, the heuristic cannot be computed
      and will return infinity.
    - Parses the `destin` facts from static information or initial state to map
      passengers to their destination floors.
    - Identifies all passengers that need to be served based on the goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, heuristic is 0.
    2. Check if the floor mapping was successfully initialized. If not, return infinity.
    3. Identify the current floor of the lift and its corresponding level. If the lift
       floor is not in the mapped floors, return infinity.
    4. Identify all passengers who are not yet served (those in the goal state that are not marked as `served` in the current state).
    5. Initialize counts: `waiting_passengers_count = 0`, `boarded_passengers_count = 0`.
    6. Initialize set of floors the lift must visit for movement calculation: `required_floors = set()`.
    7. For each unserved passenger `p`:
       - Check if `(origin p o)` is in the current state. If yes:
         - Increment `waiting_passengers_count`.
         - Add origin floor `o` to `required_floors`.
       - Check if `(boarded p)` is in the current state. If yes:
         - Increment `boarded_passengers_count`.
         - Get destination floor `d` for `p` from `self.destin_map`.
         - Add destination floor `d` to `required_floors`.
    8. Calculate the total non-movement actions: `total_non_move_actions = (waiting_passengers_count * 2) + boarded_passengers_count`. (Each waiting passenger needs 1 board + 1 depart; each boarded passenger needs 1 depart).
    9. Calculate the movement cost:
       - Get the integer level for the current lift floor (`L_current`).
       - Get the integer levels for all floors in `required_floors`. Filter out any required floors that were not successfully mapped during initialization.
       - If the set of required floor levels is empty, the movement cost is 0.
       - Otherwise, find the minimum (`L_min`) and maximum (`L_max`) levels among the required floor levels.
       - Calculate the movement cost as `min(abs(L_current - L_min) + (L_max - L_min), abs(L_current - L_max) + (L_max - L_min))`. This estimates the minimum moves needed to start at `L_current`, reach one extreme of the required range `[L_min, L_max]`, and sweep across to the other extreme.
    10. The total heuristic value is the sum of `total_non_move_actions` and the movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        super().__init__(task) # Call base class constructor

        # 1. Build floor level mapping
        above_facts = [get_parts(fact) for fact in self.static if match(fact, "above", "*", "*")]
        floors = set()
        floor_below = {}
        is_second_arg = set() # Floors that appear as the second argument in 'above' facts

        for _, f1, f2 in above_facts:
            floors.add(f1)
            floors.add(f2)
            floor_below[f1] = f2
            is_second_arg.add(f2)

        # Find the highest floor (appears as f1 but not f2 in any above fact)
        # This is the floor that is a key in floor_below but not a value
        keys = set(floor_below.keys())
        values = set(floor_below.values())
        potential_highest = list(keys - values)

        highest_floor = None
        if len(potential_highest) == 1:
             highest_floor = potential_highest[0]
        elif len(floors) == 1 and len(above_facts) == 0:
             # Case with only one floor and no above facts
             highest_floor = list(floors)[0]
        # else: Multiple potential highest floors (disconnected chains) or no floors.
        # We assume a single linear chain for miconic. If not found, the mapping fails.

        self.level_map = {}
        if highest_floor:
            current_floor = highest_floor
            level = len(floors) # Start level from total number of floors
            mapped_floors_count = 0
            # Traverse down the chain
            while current_floor is not None and current_floor in floors:
                 self.level_map[current_floor] = level
                 mapped_floors_count += 1
                 level -= 1
                 current_floor = floor_below.get(current_floor) # Use .get for lowest floor

            # If not all floors were mapped, the structure is not a single chain.
            # This heuristic relies on a single chain. Invalidate map if incomplete.
            if mapped_floors_count != len(floors):
                 self.level_map = {} # Invalidate map if incomplete

        # If level_map is empty and there were floors, it means mapping failed.
        # This heuristic cannot be computed reliably.
        self.is_mapping_valid = len(floors) == 0 or len(self.level_map) == len(floors)


        # 2. Store passenger destinations
        self.destin_map = {}
        # Destinations are static facts. They can be in initial_state or static.
        # According to PDDL spec, static predicates can appear in :init.
        # Let's check both task.static and task.initial_state.
        facts_to_check = set(self.static) # Start with static facts
        if hasattr(task, 'initial_state'):
             facts_to_check.update(task.initial_state) # Add initial state facts

        for fact in facts_to_check:
             if match(fact, "destin", "*", "*"):
                 p, f = get_parts(fact)[1:]
                 self.destin_map[p] = f

        # 3. Identify all passengers that need to be served (from goal)
        self.passengers_to_serve = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached (redundant if called by a search algorithm that checks this, but safe)
        if self.goals <= state:
            return 0

        # Check if floor mapping was successful during initialization
        if not self.is_mapping_valid:
             # Cannot compute heuristic without valid floor levels
             return float('inf')

        # 1. Get current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown or floor not mapped, return infinity (should not happen in valid state)
        if current_lift_floor is None or current_lift_floor not in self.level_map:
             return float('inf')

        current_lift_level = self.level_map[current_lift_floor]

        # 2. Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = self.passengers_to_serve - served_passengers

        # If no unserved passengers, goal is reached (already checked)
        if not unserved_passengers:
             return 0

        # 3. Count waiting/boarded and collect required floors
        waiting_passengers_count = 0
        boarded_passengers_count = 0
        required_floors = set() # Floors the lift must visit

        # Get current state of all passengers (origin or boarded) that are unserved
        passenger_current_state = {} # Map passenger -> ('origin', floor) or ('boarded', None)
        for fact in state:
             if match(fact, "origin", "*", "*"):
                 p, f = get_parts(fact)[1:]
                 if p in unserved_passengers:
                     passenger_current_state[p] = ('origin', f)
             elif match(fact, "boarded", "*"):
                 p = get_parts(fact)[1]
                 if p in unserved_passengers:
                     passenger_current_state[p] = ('boarded', None)

        for p in unserved_passengers:
            if p in passenger_current_state:
                state_type, floor = passenger_current_state[p]
                if state_type == 'origin':
                    waiting_passengers_count += 1
                    if floor in self.level_map:
                        required_floors.add(floor)
                    # else: Origin floor not mapped? Problematic instance handled by is_mapping_valid.
                elif state_type == 'boarded':
                    boarded_passengers_count += 1
                    # Add destination floor to required floors
                    if p in self.destin_map and self.destin_map[p] in self.level_map:
                        required_floors.add(self.destin_map[p])
                    # else: Destination unknown or floor not mapped? Problematic instance.
            # else: Unserved passenger is not in origin or boarded state? Problematic instance.


        # 4. Calculate total non-movement actions
        # Each waiting passenger needs 1 board + 1 depart = 2 actions
        # Each boarded passenger needs 1 depart = 1 action
        total_non_move_actions = (waiting_passengers_count * 2) + boarded_passengers_count

        # 5. Calculate movement cost
        # Only consider required floors that were successfully mapped
        required_levels = {self.level_map[f] for f in required_floors if f in self.level_map}

        movement_cost = 0
        if len(required_levels) > 0:
            L_min = min(required_levels)
            L_max = max(required_levels)
            L_c = current_lift_level

            # Cost to go from L_c to L_min and sweep up to L_max
            cost1 = abs(L_c - L_min) + (L_max - L_min)
            # Cost to go from L_c to L_max and sweep down to L_min
            cost2 = abs(L_c - L_max) + (L_max - L_min)

            movement_cost = min(cost1, cost2)

        # 6. Total heuristic value
        total_heuristic = total_non_move_actions + movement_cost

        return total_heuristic
