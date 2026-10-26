from fnmatch import fnmatch
# Assuming Heuristic base class is available in a module named heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
from collections import deque

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
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic: # Inherit from Heuristic base class in actual use
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the cost as the sum of the estimated cost for each unserved passenger.
    The estimated cost for a passenger is:
    - If waiting at origin: 2 actions (board + depart) + moves (current->origin + origin->destin).
    - If boarded: 1 action (depart) + moves (current->destin).
    - If served: 0.

    This heuristic is non-admissible as it sums costs independently, potentially
    double-counting lift movements. It aims to guide a greedy best-first search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        Assumes floors are ordered linearly by `(above lower higher)`.
        Attempts to parse floor levels from f<number> names first, falling back
        to building levels from the 'above' fact chain.
        """
        self.goals = task.goals
        self.static_facts = task.static

        self.floors = set()
        # Collect all floor names mentioned in static facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] in ["above", "origin", "destin"]:
                 for part in parts[1:]:
                     # Simple check if it looks like a floor name
                     if part.startswith('f') and len(part) > 1 and part[1:].isdigit():
                         self.floors.add(part)
            # Also collect floors from goal facts if they mention locations (though miconic goal is just served)
            # For robustness, could parse initial state facts too if available here.

        # Build floor level map: floor_name -> level (integer)
        # Assume floors are named f1, f2, ... and level corresponds to the number.
        self.floor_to_level = {}
        try:
            temp_floor_to_level = {}
            f_number_parsing_successful = True
            for floor_name in self.floors:
                if floor_name.startswith('f') and floor_name[1:].isdigit():
                    temp_floor_to_level[floor_name] = int(floor_name[1:])
                else:
                    # Handle cases where floor names don't follow f<number> pattern
                    f_number_parsing_successful = False
                    break # Stop trying f<number> if one fails

            if f_number_parsing_successful and temp_floor_to_level:
                 self.floor_to_level = temp_floor_to_level
                 # Optional: Check consistency with 'above' facts
                 consistent = True
                 for fact in self.static_facts:
                     if match(fact, "above", "*", "*"):
                         _, f_lower, f_higher = get_parts(fact)
                         if f_lower in self.floor_to_level and f_higher in self.floor_to_level:
                             if self.floor_to_level[f_lower] >= self.floor_to_level[f_higher]:
                                 # Inconsistent order
                                 consistent = False
                                 break
                 if not consistent:
                     print("Warning: Floor levels from f<number> inconsistent with (above lower higher). Attempting to build levels from 'above' chain.")
                     self.floor_to_level = self._build_levels_from_above()
            else:
                 print("Warning: Could not parse floor levels from f<number> names. Attempting to build levels from 'above' facts.")
                 self.floor_to_level = self._build_levels_from_above()

        except (ValueError, IndexError):
            print("Warning: Error during f<number> parsing. Attempting to build levels from 'above' facts.")
            self.floor_to_level = self._build_levels_from_above()

        if not self.floor_to_level and self.floors:
             print("Error: Could not determine floor levels for any floor.")


        # Store passenger destinations
        self.passenger_destin = {}
        for fact in self.static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_destin[passenger] = floor

        # Store goal served passengers
        self.goal_served_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

    def _build_levels_from_above(self):
        """Build floor levels based on the (above lower higher) chain."""
        above_map = {} # lower -> higher
        is_lower = set()
        is_higher = set()
        all_floors_in_above = set()

        for fact in self.static_facts:
            if match(fact, "above", "*", "*"):
                _, f_lower, f_higher = get_parts(fact)
                above_map[f_lower] = f_higher
                is_lower.add(f_lower)
                is_higher.add(f_higher)
                all_floors_in_above.add(f_lower)
                all_floors_in_above.add(f_higher)

        floor_to_level = {}
        # Find the lowest floor (the one that is never a 'higher' floor in the chain)
        potential_lowest = all_floors_in_above - is_higher

        if len(potential_lowest) == 1:
             lowest_floor = list(potential_lowest)[0]
             q = deque([(lowest_floor, 1)])
             visited = {lowest_floor}
             while q:
                 current_floor, level = q.popleft()
                 floor_to_level[current_floor] = level
                 next_floor = above_map.get(current_floor)
                 if next_floor and next_floor not in visited:
                     visited.add(next_floor)
                     q.append((next_floor, level + 1))
        elif len(potential_lowest) > 1:
             print("Warning: Multiple potential lowest floors found from 'above' facts. Assuming disconnected floor sets or non-linear structure.")
             # Cannot build a simple linear level map.
             return {}
        elif not all_floors_in_above and self.floors:
             # No 'above' facts found, but floors exist (e.g., from origin/destin).
             # Cannot determine order.
             print("Warning: No 'above' facts found to determine floor order.")
             return {}

        return floor_to_level


    def get_floor_level(self, floor_name):
        """Returns the integer level for a floor name, or None if not found."""
        return self.floor_to_level.get(floor_name)

    def get_level_distance(self, floor1_name, floor2_name):
        """Returns the absolute difference in levels between two floors."""
        level1 = self.get_floor_level(floor1_name)
        level2 = self.get_floor_level(floor2_name)
        if level1 is None or level2 is None:
            # Cannot determine distance, return 0 as a fallback
            # print(f"Warning: Could not get level for {floor1_name} or {floor2_name}. Distance is 0.")
            return 0
        return abs(level1 - level2)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        current_lift_floor = None
        waiting_passengers = {} # {p: origin_floor}
        boarded_passengers = set() # {p}
        served_passengers = set() # {p}

        # Identify current lift location and passenger states
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
            elif match(fact, "origin", "*", "*"):
                _, p, f = get_parts(fact)
                waiting_passengers[p] = f
            elif match(fact, "boarded", "*"):
                _, p = get_parts(fact)
                boarded_passengers.add(p)
            elif match(fact, "served", "*"):
                _, p = get_parts(fact)
                served_passengers.add(p)

        # Identify unserved passengers from the goal set
        unserved_passengers = self.goal_served_passengers - served_passengers

        total_cost = 0

        # Check if floor levels were successfully parsed and current floor is mapped
        # If not, we can only return a simple action count heuristic
        if not self.floor_to_level or current_lift_floor not in self.floor_to_level:
             # print(f"Warning: Floor levels not available or current floor {current_lift_floor} not mapped. Returning action count only.")
             for p in unserved_passengers:
                 # Count 2 actions (board + depart) if waiting, 1 (depart) if boarded
                 if p in waiting_passengers:
                     total_cost += 2
                 elif p in boarded_passengers:
                     total_cost += 1
             return total_cost

        # Calculate cost for each unserved passenger independently
        for p in unserved_passengers:
            destin_floor = self.passenger_destin.get(p)
            if destin_floor is None:
                 # Destination not found for a goal passenger. This indicates an issue
                 # with the problem definition or static facts provided.
                 # We cannot calculate cost for this passenger.
                 # print(f"Warning: Destination not found for goal passenger {p}. Skipping.")
                 continue

            if p in waiting_passengers:
                # Passenger is waiting at origin, needs board and depart
                origin_floor = waiting_passengers[p]
                # Cost = board (1) + depart (1) + moves (current->origin + origin->destin)
                total_cost += 2
                total_cost += self.get_level_distance(current_lift_floor, origin_floor)
                total_cost += self.get_level_distance(origin_floor, destin_floor)

            elif p in boarded_passengers:
                # Passenger is boarded, needs depart
                # Cost = depart (1) + moves (current->destin)
                total_cost += 1
                total_cost += self.get_level_distance(current_lift_floor, destin_floor)

            # Note: If a passenger is unserved (in goal) but neither waiting nor boarded
            # in the current state, this implies they are already at their destination
            # but not yet served (e.g., just departed). The goal requires (served ?p).
            # The current state does not have (served ?p). This case shouldn't
            # contribute to the heuristic based on the above logic, which is correct
            # because the next step for such a passenger would be (served ?p), which
            # is not an action but a goal condition achieved by (depart). The cost
            # is implicitly covered when they were boarded and moved to destination.
            # However, if the goal is *only* (served ?p), and the state has (at ?p ?d)
            # and (lift-at ?d) but not (served ?p), the heuristic should be 1 (depart).
            # My current logic misses this specific state if the passenger is not (boarded ?p).
            # Let's adjust: if unserved and not waiting/boarded, they must be at destin floor.
            # This requires checking (at ?p ?f) facts, which are not in Miconic domain.
            # Miconic only tracks (origin), (boarded), (served).
            # So, unserved passengers are either waiting or boarded. The logic is correct.


        return total_cost

