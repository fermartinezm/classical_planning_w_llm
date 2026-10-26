from fnmatch import fnmatch
import math

# Assuming heuristic_base.py is available in the same directory or in PYTHONPATH
# from heuristics.heuristic_base import Heuristic
# If Heuristic base class is not provided, define a minimal one for compatibility
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: heuristics.heuristic_base not found. Using a minimal mock class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic must implement __call__")


def get_parts(fact):
    """Helper to parse PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the cost to reach the goal by summing:
        1. A base cost for pending board/depart actions (twice the number of waiting passengers plus the number of boarded passengers).
        2. An estimated movement cost for the lift to visit all necessary floors.

    Assumptions:
        - Floors are linearly ordered by the `(above f1 f2)` facts.
        - The state representation includes `(origin ?p ?f)`, `(destin ?p ?f)`, `(boarded ?p)`, `(served ?p)`, and `(lift-at ?f)` facts.
        - Static facts include all `(above ?f1 ?f2)` and `(destin ?p ?f)` facts.
        - All passengers that need to be served are listed in the goal state.

    Heuristic Initialization:
        - Parses floor names from static facts and assigns a numerical level to each floor based on the `(above f1 f2)` relationships. The lowest floor is assigned level 1. Includes a fallback for non-standard floor definitions.
        - Stores passenger destination floors from static facts.
        - Stores all passenger names from goal facts.

    Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current floor of the lift from the state. If the lift location is not found or the floor is not recognized, return infinity.
        2. Identify passengers who are waiting at their origin floors (`(origin ...)`), passengers who are boarded (`(boarded ...)`), and passengers who are served (`(served ...)`). Store origin floors for waiting passengers.
        3. Filter the waiting and boarded passenger sets to exclude those who are already served.
        4. Calculate the base action cost: This is an estimate of the board and depart actions needed. Each waiting passenger requires a board action and a depart action (cost 2). Each boarded passenger requires a depart action (cost 1). Sum these costs: `2 * len(waiting_passengers) + len(boarded_passengers)`.
        5. If the base action cost is 0, it means no unserved passengers are waiting or boarded, implying all passengers are served. In this case, the heuristic is 0 (goal state).
        6. If the base action cost is greater than 0 (i.e., not a goal state), calculate the movement cost:
           - Identify the set of 'relevant' floors the lift must visit. These are the origin floors of all passengers currently waiting at their origin, and the destination floors of all passengers currently boarded.
           - Find the minimum and maximum floor levels among these relevant floors using the pre-computed floor level mapping.
           - The estimated movement cost is calculated as `(max_level - min_level) + min(abs(current_level - min_level), abs(current_level - max_level))`. This formula estimates the total vertical travel needed to reach the range of required floors and traverse that range, starting from the current lift floor.
        7. The total heuristic value is the sum of the base action cost and the movement cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse floor levels based on (above f1 f2) facts
        floors = set()
        above_pairs = set() # Store (f_higher, f_lower) pairs
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                f1, f2 = parts[1:]
                floors.add(f1)
                floors.add(f2)
                above_pairs.add((f1, f2))

        self.floor_levels = {}
        # Determine floor levels: level of f is 1 + number of floors f is below it
        # f is below f_higher if (above f_higher f) is true
        for f in floors:
            count_below = sum(1 for f_higher, f_lower in above_pairs if f_lower == f)
            # In a linear order f_n < ... < f_1, f_n is below 0 floors (level 1),
            # f_{n-1} is below 1 floor (f_n) (level 2), ..., f_1 is below n-1 floors (level n).
            # So level = 1 + count_below.
            self.floor_levels[f] = 1 + count_below

        # Fallback for any floors not assigned a level (e.g., disconnected components)
        if len(self.floor_levels) != len(floors):
             print(f"Warning: Could not assign levels to all floors based on (above) facts. Found {len(floors)} floors, assigned levels to {len(self.floor_levels)}.")
             remaining_floors = sorted(list(floors - set(self.floor_levels.keys())))
             start_level = max(self.floor_levels.values()) + 1 if self.floor_levels else 1
             for i, f in enumerate(remaining_floors):
                 self.floor_levels[f] = start_level + i
             print(f"Assigned arbitrary levels to {len(remaining_floors)} remaining floors.")


        # 2. Store passenger destinations
        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                p, destin_f = get_parts(fact)[1:]
                self.passenger_destinations[p] = destin_f

        # 3. Get all passenger names from goals (assuming all passengers need to be served)
        self.all_passengers = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 p = get_parts(goal)[1]
                 self.all_passengers.add(p)


    def __call__(self, node):
        state = node.state

        # 1. Find current lift floor
        current_f = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_f = get_parts(fact)[1]
                break

        # Should always find the lift location in a valid state
        if current_f is None or current_f not in self.floor_levels:
             # This state is likely invalid or represents a failure case
             return math.inf

        current_level = self.floor_levels[current_f]

        # 2. Identify passenger status and relevant floors
        waiting_passengers = set() # passengers with (origin ...)
        boarded_passengers = set() # passengers with (boarded ...)
        served_passengers = set()  # passengers with (served ...)

        origin_floor_map = {} # Map passenger -> origin_f

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                p, origin_f = parts[1:]
                waiting_passengers.add(p)
                origin_floor_map[p] = origin_f
            elif parts[0] == "boarded":
                p = parts[1]
                boarded_passengers.add(p)
            elif parts[0] == "served":
                p = parts[1]
                served_passengers.add(p)

        # Filter out served passengers from waiting and boarded sets
        waiting_passengers -= served_passengers
        boarded_passengers -= served_passengers

        # 4. Calculate base action cost
        # Each waiting passenger needs board (1) + depart (1) = 2 actions
        # Each boarded passenger needs depart (1) action
        base_action_cost = 2 * len(waiting_passengers) + len(boarded_passengers)

        # 5. If base_action_cost is 0, it means no unserved passengers are waiting or boarded.
        # This implies all passengers are served (assuming self.all_passengers is correct).
        # In this case, the heuristic is 0 (goal state).
        if base_action_cost == 0:
             return 0

        # 6. Calculate movement cost
        active_origin_floors = {origin_floor_map[p] for p in waiting_passengers if p in origin_floor_map}
        active_destin_floors = {self.passenger_destinations[p] for p in boarded_passengers if p in self.passenger_destinations}

        all_relevant_floors = active_origin_floors.union(active_destin_floors)

        # If base_action_cost > 0, all_relevant_floors must be non-empty.
        # We can assert this or rely on min/max handling empty sets (which they don't).
        # The logic implies it's non-empty, so we proceed.

        min_level = min(self.floor_levels[f] for f in all_relevant_floors)
        max_level = max(self.floor_levels[f] for f in all_relevant_floors)

        # Movement cost: distance to cover the range + distance to reach the closer end of the range
        movement_cost = (max_level - min_level) + min(abs(current_level - min_level), abs(current_level - max_level))

        # 7. Total heuristic
        return base_action_cost + movement_cost
