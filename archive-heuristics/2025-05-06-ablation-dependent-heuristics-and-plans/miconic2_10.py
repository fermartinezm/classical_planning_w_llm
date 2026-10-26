from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided by the environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


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
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts 2 actions (board + depart) for each unserved passenger and adds
    an estimate of the minimum number of lift movement actions required to visit
    all necessary floors (origins of unboarded passengers and destinations of
    boarded passengers).

    # Assumptions
    - Floors are ordered linearly (e.g., f1, f2, f3, ...). The heuristic assumes
      alphabetical sorting of floor names corresponds to their physical order
      from lowest to highest.
    - All passengers have a defined origin and destination in the static facts.
    - The state representation is consistent (an unserved passenger is either
      at their origin or boarded).

    # Heuristic Initialization
    - Extracts all passenger names from goal facts.
    - Extracts passenger origin and destination floors from static facts.
    - Determines the floor ordering and creates a mapping from floor name to
      its level (1-based index) by sorting floor names alphabetically.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift.
    2. Identify all passengers who have not yet been served (goal condition not met).
    3. If all passengers are served, the heuristic is 0.
    4. For each unserved passenger:
       - Determine their current status: are they at their origin floor (unboarded) or are they boarded?
       - If unboarded at their origin floor, the lift must visit this origin floor to pick them up. Add this origin floor to the set of required stops.
       - If boarded, the lift must visit their destination floor to drop them off. Add this destination floor to the set of required stops.
    5. If the set of required stops is empty (this should only happen if all passengers are served, handled in step 3), the heuristic is 0.
    6. Map the required stop floors to their corresponding levels using the precomputed floor-to-level map.
    7. Find the minimum and maximum levels among the required stops.
    8. Calculate the estimated minimum number of lift movement actions:
       - This is the distance from the current lift floor's level to the level of the closest required extreme floor (either the minimum or maximum required level), plus the total span of the required floor levels (maximum required level - minimum required level).
       - Estimated moves = min(abs(current_lift_level - min_required_level), abs(current_lift_level - max_required_level)) + (max_required_level - min_required_level).
    9. The total heuristic value is the sum of:
       - 2 actions (board + depart) for each unserved passenger.
       - The estimated minimum lift movement actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about
        floors, passengers, origins, and destinations.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Get all passenger names from goals
        self.all_passengers = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                self.all_passengers.add(get_parts(goal)[1])

        # 2. Get passenger origins and destinations from static facts
        self.passenger_origins = {}
        self.passenger_destinations = {}
        all_floors_set = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "origin":
                self.passenger_origins[parts[1]] = parts[2]
                all_floors_set.add(parts[2])
            elif parts[0] == "destin":
                self.passenger_destinations[parts[1]] = parts[2]
                all_floors_set.add(parts[2])
            elif parts[0] == "above":
                 all_floors_set.add(parts[1])
                 all_floors_set.add(parts[2])

        # 3. Determine floor ordering and levels
        # Assuming floor names like f1, f2, ... fn imply order
        sorted_floors = sorted(list(all_floors_set))
        self.floor_to_level_map = {floor: i + 1 for i, floor in enumerate(sorted_floors)}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        # Assuming valid states, lift location is always present.

        # 2. Identify served passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # 3. Identify unserved passengers
        unserved_passengers = self.all_passengers - served_passengers

        # 4. If all passengers are served, heuristic is 0
        if not unserved_passengers:
            return 0

        # 5. Collect required floors (origins for unboarded, destinations for boarded)
        required_floors = set()
        for p in unserved_passengers:
            if f"(boarded {p})" in state:
                # Passenger is boarded, needs to go to destination
                required_floors.add(self.passenger_destinations[p])
            else:
                # Passenger is unboarded, needs to be picked up at origin
                # Find origin floor from state facts
                origin_floor = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == "origin" and parts[1] == p:
                        origin_floor = parts[2]
                        break
                if origin_floor:
                    required_floors.add(origin_floor)
                # else: Unserved passenger not boarded and not at origin? Invalid state.
                # Assuming valid states reachable from initial state.

        # 6. If no required floors (should not happen in valid states if unserved > 0)
        if not required_floors:
             # This case implies unserved passengers are in an unexpected state.
             # For a greedy heuristic, we can return a base cost or rely on
             # the assumption of valid states. Let's return 0 here, although
             # a non-zero value might be more robust if invalid states are possible.
             # However, if unserved > 0, required_floors should be > 0 in valid states.
             return 0 # Should not be reached if unserved_passengers is not empty and state is valid

        # 7. Get levels for required floors
        required_levels = {self.floor_to_level_map[f] for f in required_floors}

        # 8. Calculate min/max required levels and current lift level
        min_s_level = min(required_levels)
        max_s_level = max(required_levels)
        lift_level = self.floor_to_level_map[current_lift_floor]

        # 9. Calculate estimated moves
        # Distance from current lift floor to the closest extreme required floor
        dist_to_min = abs(lift_level - min_s_level)
        dist_to_max = abs(lift_level - max_s_level)
        dist_to_closest_extreme = min(dist_to_min, dist_to_max)

        # Span of required floors
        span_required_floors = max_s_level - min_s_level

        # Estimated moves = distance to closest extreme + span
        estimated_moves = dist_to_closest_extreme + span_required_floors

        # 10. Total heuristic = 2 actions per unserved passenger + estimated moves
        total_cost = len(unserved_passengers) * 2 + estimated_moves

        return total_cost
