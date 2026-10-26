from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to parse PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to match fact parts with arguments using fnmatch."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    # Use fnmatch for flexible matching (e.g., '*' wildcard)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the cost to reach the goal by summing the number of board
        and depart actions needed for unserved passengers and the estimated
        minimum lift movement cost to visit all necessary floors.
        Necessary floors include the origin floors of waiting passengers
        and the destination floors of boarded passengers. The movement cost
        is estimated as the distance to reach one end of the required floor
        range from the current lift position, plus the distance covering
        the entire required floor range.

    Assumptions:
        - The PDDL domain is miconic as provided.
        - The floor structure defined by (above f_higher f_lower) facts forms a total order.
        - The state and static facts are provided as frozensets of strings
          like '(predicate arg1 arg2)'.
        - The goal is a conjunction of (served ?person) for all passengers.
        - Passenger names start with 'p' and floor names start with 'f'.

    Heuristic Initialization:
        - Stores the task object.
        - Parses the (above f_higher f_lower) static facts to determine the ordered list
          of floors and create a mapping from floor name to its index.
        - Extracts the destination floor for each passenger from the static facts.
        - Identifies all passengers present in the problem from initial state
          and static destination facts.

    Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current floor of the lift from the state.
        2. Identify all passengers who are not yet served (i.e., (served p)
           is not in the state).
        3. For each unserved passenger:
           - If (origin p f) is in the state: The passenger is waiting at floor f.
             This floor f must be visited to pick them up. The passenger's
             destination floor (destin p d) must be visited to drop them off.
           - If (boarded p) is in the state: The passenger is in the lift.
             Their destination floor (destin p d) must be visited to drop them off.
        4. Collect the set of all floors that must be visited (origin floors
           of waiting passengers + destination floors of boarded passengers).
           Only include floors that were successfully parsed into the floor order.
        5. If the set of floors to visit is empty, the movement cost is 0.
        6. If the set is not empty:
           - Find the minimum and maximum floor indices among the floors to visit.
           - Calculate the estimated movement cost:
             distance from current lift floor index to the closer of the min/max
             required floor indices, plus the distance between the min and max
             required floor indices.
             Movement cost = min(|current_idx - min_visit_idx|, |current_idx - max_visit_idx|)
                             + (max_visit_idx - min_visit_idx)
             If the current lift floor was not successfully parsed, movement cost is 0.
        7. Count the number of 'board' actions needed: This is the number of
           waiting passengers among the unserved passengers.
        8. Count the number of 'depart' actions needed: This is the number of
           boarded passengers among the unserved passengers.
        9. The total heuristic value is the sum of estimated movement cost,
           number of board actions needed, and number of depart actions needed.
    """
    def __init__(self, task):
        super().__init__(task)
        self.task = task # Store the task object

        # 1. Parse floor order and create floor_to_index map
        self.floor_to_index = self._parse_floor_order()
        self.index_to_floor = {v: k for k, v in self.floor_to_index.items()}
        self.num_floors = len(self.floor_to_index)

        # 2. Extract passenger destinations
        self.passenger_destinations = self._parse_passenger_destinations()

        # 3. Identify all passengers
        self.all_passengers = self._get_all_passengers() # Use stored task.initial_state


    def _parse_floor_order(self):
        """Parses (above f_higher f_lower) facts to determine floor order and indices."""
        above_relations = [] # List of (f_higher, f_lower) tuples
        all_floors = set()
        floors_that_are_below = set() # Floors that appear as the second argument (f_lower)

        for fact in self.task.static: # Use stored task
            if match(fact, "above", "*", "*"):
                f_higher, f_lower = get_parts(fact)[1:]
                above_relations.append((f_higher, f_lower))
                all_floors.add(f_higher)
                all_floors.add(f_lower)
                floors_that_are_below.add(f_lower)

        if not all_floors:
             # Handle case with no floors or no above relations (e.g., single floor)
             # Try to find floors from initial state if static is empty or lacks above facts
             for fact in self.task.initial_state: # Use stored task
                 parts = get_parts(fact)
                 if len(parts) > 1 and parts[0] in ['lift-at', 'origin', 'destin']:
                     for part in parts[1:]:
                         # Simple heuristic: assume floor names start with 'f'
                         if isinstance(part, str) and part.startswith('f'):
                             all_floors.add(part)

             if not all_floors: # Still no floors found
                 return {} # Cannot build floor order

        # Find the lowest floor: a floor in all_floors that is NOT in floors_that_are_below
        potential_lowest_floors = list(all_floors - floors_that_are_below)

        if len(potential_lowest_floors) != 1:
             # This implies the 'above' facts don't form a single linear chain,
             # or it's a single floor case.
             if len(all_floors) == 1:
                  lowest_floor = list(all_floors)[0]
             else:
                  # Multiple potential lowest floors or none found (cycle?).
                  # This shouldn't happen in standard miconic.
                  # Fallback: sort by number if possible, otherwise pick one.
                  try:
                      sorted_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
                      # The lowest floor relationally is the one never appearing as f_lower.
                      # If multiple potential_lowest_floors exist relationally, the structure is broken.
                      # Picking the numerically lowest as a fallback might be misleading.
                      # Let's prioritize the relational structure. If it's broken, pick one relationally lowest.
                      if potential_lowest_floors:
                           lowest_floor = potential_lowest_floors[0]
                      elif all_floors: # Should not happen if all_floors is not empty and no lowest found
                           lowest_floor = list(all_floors)[0]
                      else:
                           return {} # Should not happen

                  except (ValueError, IndexError):
                      # Floor names are not f1, f2, ...
                      if potential_lowest_floors:
                           lowest_floor = potential_lowest_floors[0]
                      elif all_floors:
                           lowest_floor = list(all_floors)[0]
                      else:
                           return {}

        else: # Exactly one potential lowest floor found
             lowest_floor = potential_lowest_floors[0]


        floor_order = []
        current_floor = lowest_floor
        floor_to_index = {}
        index = 0

        # Build the ordered list by following the 'above' relations upwards
        # Start with the lowest floor. Find the floor F_above such that (above F_above current_floor) is true.
        # That F_above is the next floor up.
        # We need a mapping from f_lower to f_higher
        lower_to_higher = {f_lower: f_higher for f_higher, f_lower in above_relations}

        while current_floor is not None and current_floor not in floor_to_index:
            floor_order.append(current_floor)
            floor_to_index[current_floor] = index
            index += 1

            # Find the floor directly above the current_floor
            next_floor = lower_to_higher.get(current_floor)

            current_floor = next_floor

        # Check if all floors were included
        if len(floor_order) != len(all_floors):
             # This indicates a problem with the 'above' facts (e.g., not a single chain)
             # The floor_to_index map will be incomplete. The heuristic might be inaccurate.
             # Assume valid miconic problems have a single linear floor structure.
             pass # Proceed with the potentially incomplete map

        return floor_to_index

    def _parse_passenger_destinations(self):
        """Extracts destination floor for each passenger from static facts."""
        passenger_destinations = {}
        for fact in self.task.static: # Use stored task
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                passenger_destinations[passenger] = floor
        return passenger_destinations

    def _get_all_passengers(self):
        """Identifies all passengers from initial state and static destinations."""
        all_passengers = set()
        # From initial state facts
        for fact in self.task.initial_state: # Use stored task
            parts = get_parts(fact)
            if len(parts) > 1 and parts[0] in ['origin', 'boarded', 'served']:
                 # Assuming passenger names start with 'p' and are the second argument
                 if isinstance(parts[1], str) and parts[1].startswith('p'):
                     all_passengers.add(parts[1])
        # From static destination facts
        all_passengers.update(self.passenger_destinations.keys())
        # Goals are served facts, passengers should be covered by initial state or destinations
        # No need to explicitly parse goals for passenger names.

        return all_passengers


    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.task.goals <= state: # Use stored task
            return 0

        # 1. Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown or not in our floor map, we cannot estimate movement.
        # This indicates a problem with the state or floor parsing.
        # In this case, we return a heuristic based only on board/depart counts.
        current_floor_idx = self.floor_to_index.get(current_lift_floor)
        can_estimate_movement = (current_floor_idx is not None)


        # 2. Identify unserved passengers and their status/location
        unserved_passengers = {p for p in self.all_passengers if '(served ' + p + ')' not in state}

        waiting_passengers_count = 0
        boarded_passengers_count = 0
        required_floors_to_visit = set() # Floors relevant to unserved passengers

        for p in unserved_passengers:
            is_waiting = False
            for fact in state:
                if match(fact, "origin", p, "*"):
                    floor = get_parts(fact)[2]
                    waiting_passengers_count += 1
                    # Add origin floor to required visits if it's in our floor map
                    if floor in self.floor_to_index:
                         required_floors_to_visit.add(floor)
                    is_waiting = True
                    break # Assuming a passenger can only wait at one floor

            if not is_waiting: # If not waiting, check if boarded
                 if '(boarded ' + p + ')' in state:
                     boarded_passengers_count += 1
                     # Add destination floor to required visits if it's in our floor map
                     dest_floor = self.passenger_destinations.get(p)
                     if dest_floor and dest_floor in self.floor_to_index:
                          required_floors_to_visit.add(dest_floor)
                     # else: Destination unknown or not in floor map - cannot use for movement estimate

        # 3. Calculate estimated movement cost
        movement_cost = 0
        if can_estimate_movement and required_floors_to_visit:
            required_floors_indices = {self.floor_to_index[f] for f in required_floors_to_visit}

            min_idx = min(required_floors_indices)
            max_idx = max(required_floors_indices)

            # Estimated moves to reach one end of the range + traverse the range
            movement_cost = min(abs(current_floor_idx - min_idx), abs(current_floor_idx - max_idx)) + (max_idx - min_idx)

        # 4. Total heuristic
        # Each waiting passenger needs a board action (cost 1).
        # Each boarded passenger needs a depart action (cost 1).
        # Movement cost is estimated separately.
        total_cost = movement_cost + waiting_passengers_count + boarded_passengers_count

        # If there are unserved passengers but the total cost is 0, something is wrong.
        # This shouldn't happen with the current logic unless all required floors
        # were not in the floor_to_index map AND board/depart counts are 0 (which implies no unserved).
        # The only way unserved > 0 and board+depart == 0 is if passenger status is neither origin nor boarded,
        # which is an invalid state representation for unserved passengers in miconic.
        # Let's trust the state representation. If unserved > 0, then waiting_passengers_count + boarded_passengers_count > 0.
        # So total_cost will be > 0 if unserved_passengers > 0.

        return total_cost
