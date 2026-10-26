# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

# Define a dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    # Remove outer parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
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
    The heuristic estimates the number of actions required to serve all passengers.
    It considers the number of unserved passengers (each needing board/depart)
    and the estimated lift travel distance to visit all floors where passengers
    need to be picked up or dropped off.

    # Assumptions
    - Floors are arranged linearly, and the 'above' predicate defines the immediate
      floor above.
    - The lift can carry multiple passengers.
    - The cost of each action (board, depart, up, down) is 1.

    # Heuristic Initialization
    - Parses the 'above' static facts to build a mapping from floor names to
      numerical indices, representing their order in the building (lowest floor is 1).
    - Extracts the destination floor for each passenger from the static facts.
    - Identifies all passengers involved in the problem from the goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all passengers that have not yet been served based on the goal
       conditions and the current state.
    2. If all passengers are served, the heuristic is 0.
    3. Determine the current floor of the lift.
    4. Identify the set of 'service floors':
       - Floors where unserved passengers are waiting (origin).
       - Floors where boarded unserved passengers need to be dropped off (destination).
    5. If there are no service floors (meaning all unserved passengers are
       boarded and their destination is the current lift floor), the heuristic
       is simply the number of unserved passengers (each needs a 'depart' action).
    6. If there are service floors:
       - Find the minimum and maximum floor indices among the service floors.
       - Calculate the estimated lift travel cost: This is the distance from the
         current lift floor to the nearest service floor, plus the distance
         between the minimum and maximum service floors. This estimates the
         travel needed to reach the "action zone" and then traverse it.
         Specifically, if the current floor is outside the range [min_idx, max_idx],
         travel is `abs(current_idx - nearest_end_idx) + (max_idx - min_idx)`.
         If the current floor is inside the range, travel is `(max_idx - min_idx) + min(current_idx - min_idx, max_idx - current_idx)`.
       - Calculate the passenger action cost: Each unserved passenger who is
         waiting needs a board (1) and a depart (1) action. Each unserved
         passenger who is boarded needs a depart (1) action.
       - The total heuristic value is the sum of the estimated travel cost and
         the passenger action cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the floor index map and storing
        passenger destinations.
        """
        super().__init__(task) # Call base class constructor if necessary

        # Build floor index map: floor_name -> index (e.g., 'f1' -> 1, 'f2' -> 2)
        self.floor_indices = {}
        below_to_above_map = {} # Maps f_below to f_above
        all_floors = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'above':
                f_above, f_below = parts[1], parts[2]
                below_to_above_map[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Find the lowest floor: The floor that is in all_floors but is not a value in below_to_above_map
        # (i.e., no floor is above it)
        floors_above_some = set(below_to_above_map.values())
        potential_lowest = all_floors - floors_above_some

        lowest_floor = None
        if len(potential_lowest) == 1:
             lowest_floor = potential_lowest.pop()
        elif len(all_floors) == 1:
             # Case with only one floor
             lowest_floor = list(all_floors)[0]
        # else: Handle cases with disconnected floors or no clear lowest floor (assume valid miconic structure)

        if lowest_floor:
            current_floor = lowest_floor
            current_index = 1
            while current_floor in all_floors:
               self.floor_indices[current_floor] = current_index
               # Find the floor immediately above current_floor
               next_floor = below_to_above_map.get(current_floor)
               if next_floor:
                 current_floor = next_floor
                 current_index += 1
               else: # Reached the highest floor
                 break
        # else: If lowest_floor wasn't found (e.g., no above facts, or complex structure),
        # floor_indices remains empty. The heuristic will return inf if floor is not found.


        # Store passenger destinations
        self.passenger_destinations = {}
        self.all_passengers = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'served':
                  passenger = parts[1]
                  self.all_passengers.add(passenger)

        # Need to find destinations from static facts
        for fact in self.static:
             parts = get_parts(fact)
             if parts and parts[0] == 'destin':
                  passenger, floor = parts[1], parts[2]
                  self.passenger_destinations[passenger] = floor


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unserved passengers
        unserved_passengers = {
            p for p in self.all_passengers if f'(served {p})' not in state
        }

        # 2. If all passengers are served, heuristic is 0
        if not unserved_passengers:
            return 0

        # 3. Determine current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None or current_lift_floor not in self.floor_indices:
             # This state is likely invalid or terminal without lift or unindexed floor
             return float('inf') # Should not happen in valid miconic states

        # 4. Identify service floors and count waiting/boarded unserved passengers
        F_pickup = set()
        F_dropoff = set()

        num_waiting_unserved = 0
        num_boarded_unserved = 0

        for p in unserved_passengers:
            is_waiting = False
            # Check if passenger is waiting at origin
            for fact in state:
                 if match(fact, "origin", p, "*"):
                      F_pickup.add(get_parts(fact)[2])
                      is_waiting = True
                      break # Found origin, move to next passenger check

            if is_waiting:
                 num_waiting_unserved += 1
            elif f'(boarded {p})' in state:
                 dest_floor = self.passenger_destinations.get(p)
                 if dest_floor: # Should always have a destination
                      F_dropoff.add(dest_floor)
                      num_boarded_unserved += 1
            # else: passenger is unserved but not waiting and not boarded? Invalid state.

        F_visit = F_pickup | F_dropoff

        # 6. Calculate travel cost
        travel_cost = 0
        if F_visit: # Only calculate travel if there are floors to visit
            Indices_visit = {self.floor_indices[f] for f in F_visit if f in self.floor_indices}
            if not Indices_visit: # Should not happen if F_visit is not empty and floors are indexed
                 # This implies a service floor was not indexed, which is an issue with init or problem definition
                 return float('inf') # Safety break

            min_idx = min(Indices_visit)
            max_idx = max(Indices_visit)
            current_idx = self.floor_indices[current_lift_floor]

            # 7. Calculate estimated travel cost
            if current_idx < min_idx:
                travel_cost = abs(current_idx - min_idx) + (max_idx - min_idx)
            elif current_idx > max_idx:
                travel_cost = abs(current_idx - max_idx) + (max_idx - min_idx)
            else: # min_idx <= current_idx <= max_idx
                travel_cost = (max_idx - min_idx) + min(current_idx - min_idx, max_idx - current_idx)

        # 8. Calculate passenger action cost
        # Each waiting needs board (1) + depart (1) = 2
        # Each boarded needs depart (1) = 1
        passenger_action_cost = (num_waiting_unserved * 2) + num_boarded_unserved

        # Total heuristic
        h = travel_cost + passenger_action_cost

        return h
