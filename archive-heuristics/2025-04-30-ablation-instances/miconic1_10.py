from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, though PDDL facts should be consistent
        # print(f"Warning: Unexpected fact format: {fact}")
        return [] # Return empty list for safety

    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args for a potential match
    if len(parts) != len(args):
        return False
        
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved
    passengers. It sums the number of unserved passengers (representing the
    minimum number of board/depart actions needed) and an estimate of the
    vertical movement cost for the lift to visit all necessary floors.

    # Assumptions
    - Floors are ordered linearly (e.g., f1, f2, ..., fN).
    - The `(above f_higher f_lower)` predicate implies `f_higher` is at a higher
      level than `f_lower`. Based on the example static facts provided,
      this implies f1 > f2 > ... > fN in terms of level.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding or departing a passenger is 1.
    - The heuristic does not account for optimal sequencing of pickups and dropoffs
      or the capacity of the lift (which is effectively infinite in this domain).

    # Heuristic Initialization
    - Extract all passenger names and their destination floors from the static facts.
    - Extract all floor names and create a mapping from floor name to numerical level.
      Based on the example static facts `(above f1 f2)`, `(above f2 f3)`, etc.,
      f1 is the highest floor, f2 is the next highest, and so on. The mapping
      assigns level N to f1, N-1 to f2, ..., 1 to fN, where N is the total number
      of floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify all unserved passengers. An unserved passenger is one for whom
       `(served ?p)` is not true in the current state.
    3. Count the total number of unserved passengers. This contributes directly
       to the heuristic as each unserved passenger will eventually require
       at least one board (if waiting) and one depart action.
    4. Determine the set of floors the lift *must* visit:
       - For each waiting passenger `p` (i.e., `(origin p f)` is true), add their
         origin floor `f` to the set of required pickup floors.
       - For each boarded passenger `p` (i.e., `(boarded p)` is true), add their
         destination floor (looked up from initialization data) to the set of
         required dropoff floors.
       - The set of floors to visit is the union of pickup and dropoff floors.
    5. If the set of floors to visit is empty, all unserved passengers must be
       waiting at their origin floors, and no one is boarded. In a standard
       miconic problem where all passengers must be served, this implies all
       passengers are already served. The heuristic is 0.
    6. If the set of floors to visit is not empty:
       - Find the numerical level for the current lift floor using the pre-computed map.
       - Find the minimum and maximum numerical levels among the floors to visit.
       - Estimate the movement cost as the total vertical distance covered by
         traveling from the current floor to the range of required floors and
         then traversing that range. This is calculated as
         `max(current_level, max_level_to_visit) - min(current_level, min_level_to_visit)`.
    7. The total heuristic value is the sum of the number of unserved passengers
       and the estimated movement cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # The goals are not directly used in this heuristic calculation,
        # but the base class might expect it.
        self.goals = task.goals
        static_facts = task.static

        self.passenger_dest = {}
        all_passengers = set()
        all_floors = set()

        # Extract passenger destinations and collect all passenger and floor names
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "destin" and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.passenger_dest[passenger] = floor
                all_passengers.add(passenger)
                all_floors.add(floor)
            elif parts[0] == "above" and len(parts) == 3:
                # Collect floor names from above predicates
                all_floors.add(parts[1])
                all_floors.add(parts[2])
            # Add other predicates if they might contain object names not in destin/above
            # For miconic, origin, boarded, served, lift-at also contain passenger/floor names
            elif parts[0] in ["origin", "lift-at"] and len(parts) >= 2: # origin has 3 parts, lift-at has 2
                 if len(parts) == 3: # origin
                     all_passengers.add(parts[1])
                     all_floors.add(parts[2])
                 elif len(parts) == 2: # lift-at
                     all_floors.add(parts[1])
            elif parts[0] in ["boarded", "served"] and len(parts) == 2:
                 all_passengers.add(parts[1])


        self.all_passengers = frozenset(all_passengers)

        # Build floor_to_level mapping
        # Assuming floor names are f1, f2, ..., fN and f1 > f2 > ... > fN
        # based on example static facts (above f1 f2), (above f2 f3), etc.
        # Sort floor names numerically ascending (f1, f2, ..., fN)
        # Handle potential non-numeric floor names if necessary, but assuming f<number> format
        try:
            sorted_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
        except ValueError:
             # Handle cases where floor names are not in f<number> format
             # Fallback to alphabetical sort or raise error
             # print("Warning: Floor names not in f<number> format. Sorting alphabetically.")
             sorted_floors = sorted(list(all_floors)) # Fallback sort

        num_floors = len(sorted_floors)
        self.floor_to_level = {}
        # Assign levels in descending order based on this sort
        # f1 gets level N, f2 gets N-1, ..., fN gets level 1
        for i, floor in enumerate(sorted_floors):
             self.floor_to_level[floor] = num_floors - i

        # Example check: if floors are f1, f2, f3 (N=3)
        # sorted_floors = ['f1', 'f2', 'f3']
        # f1 -> level 3-0 = 3
        # f2 -> level 3-1 = 2
        # f3 -> level 3-2 = 1
        # This means f1 > f2 > f3, consistent with (above f1 f2), (above f2 f3)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # Should always find lift-at in a valid state
        if current_lift_floor is None or current_lift_floor not in self.floor_to_level:
             # This indicates a state inconsistent with domain structure or initialization
             # Returning infinity signals this state is likely unreachable or invalid
             return float('inf')

        # 2. Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = self.all_passengers - served_passengers
        num_unserved = len(unserved_passengers)

        # If no unserved passengers, goal is reached
        if num_unserved == 0:
            return 0

        # 3. Determine floors to visit
        pickup_floors = set()
        boarded_passengers = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "origin" and len(parts) == 3:
                # Passenger is waiting at origin floor
                passenger, floor = parts[1], parts[2]
                if passenger in unserved_passengers: # Only care about unserved waiting passengers
                    pickup_floors.add(floor)
            elif parts[0] == "boarded" and len(parts) == 2:
                 # Passenger is boarded
                 passenger = parts[1]
                 if passenger in unserved_passengers: # Only care about unserved boarded passengers
                     boarded_passengers.add(passenger)

        # Determine dropoff floors for currently boarded unserved passengers
        dropoff_floors = set()
        for passenger in boarded_passengers:
             # Ensure passenger has a destination (should be true for all passengers)
             if passenger in self.passenger_dest:
                  dropoff_floors.add(self.passenger_dest[passenger])
             # else: print(f"Warning: Boarded passenger {passenger} has no destination.")


        floors_to_visit = pickup_floors.union(dropoff_floors)

        # 5. If no floors to visit, it implies all unserved passengers are waiting
        # at origins that don't need visiting (e.g., already at destination?)
        # or the state is inconsistent. In a standard problem, unserved implies
        # needing pickup or dropoff. If floors_to_visit is empty, num_unserved
        # should be 0. If num_unserved > 0 and floors_to_visit is empty, something
        # is wrong with the state or problem definition.
        if not floors_to_visit:
             # This case should ideally not be reached if num_unserved > 0
             # Returning num_unserved provides a baseline cost
             # print("Warning: Unserved passengers exist but no floors to visit.")
             return num_unserved # Fallback

        # 6. Estimate movement cost
        current_level = self.floor_to_level[current_lift_floor]
        
        # Ensure all floors to visit are in our mapping (should be true if initialization is correct)
        valid_floors_to_visit = {f for f in floors_to_visit if f in self.floor_to_level}
        if not valid_floors_to_visit:
             # This implies floors collected from state are not in static facts
             # Should not happen in valid problems.
             # print("Warning: Floors to visit not found in floor_to_level map.")
             return num_unserved # Fallback

        levels_to_visit = {self.floor_to_level[f] for f in valid_floors_to_visit}
        min_level_to_visit = min(levels_to_visit)
        max_level_to_visit = max(levels_to_visit)

        # Movement cost estimate: distance from current floor to the range [min_level, max_level]
        # plus the distance to traverse the range.
        # This simplifies to max(current_level, max_level_to_visit) - min(current_level, min_level_to_visit)
        movement_cost = max(current_level, max_level_to_visit) - min(current_level, min_level_to_visit)

        # 7. Total heuristic
        # Add the number of unserved passengers (representing board/depart actions)
        # and the estimated movement cost.
        total_heuristic = num_unserved + movement_cost

        return total_heuristic
