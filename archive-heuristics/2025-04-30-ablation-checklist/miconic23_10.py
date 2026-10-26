from fnmatch import fnmatch
# Assuming a base Heuristic class is available as described in the problem
# For standalone execution or testing, you might need a minimal definition:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         raise NotImplementedError

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the remaining cost by summing the individual costs for each
    unserved passenger. The cost for a passenger is estimated as:
    - If waiting at origin O: Distance(Lift, O) + 1 (board) + Distance(O, D) + 1 (depart)
    - If boarded: Distance(Lift, D) + 1 (depart)

    Distances between floors are calculated based on a parsed floor level mapping
    derived from the 'above' predicates. The level of a floor is determined by
    counting how many other floors are strictly below it according to the
    'above' facts.

    This heuristic is non-admissible as it counts lift movements independently
    for each passenger, ignoring potential synergies (e.g., picking up/dropping
    off multiple passengers on the same trip). However, it provides a greedy
    estimate that considers the spatial aspect of the problem.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by parsing floor levels and passenger destinations.
        """
        super().__init__(task)

        # 1. Parse floor levels from static 'above' facts
        self.floor_to_level = {}
        self.level_to_floor = {}
        all_floors = set()

        # Collect all floor objects mentioned in initial state or static facts
        for fact in task.initial_state | task.static:
             parts = get_parts(fact)
             if parts[0] in ["lift-at"] and len(parts) > 1:
                 all_floors.add(parts[1])
             elif parts[0] in ["origin", "destin"] and len(parts) > 2:
                 all_floors.add(parts[2])
             elif parts[0] in ["above"] and len(parts) > 2:
                 all_floors.add(parts[1])
                 all_floors.add(parts[2])

        # Determine floor levels based on the number of floors below each floor
        # A floor f_a is below f_b if (above f_b f_a) is true.
        # The level of a floor is 1 + (number of floors below it).
        floor_levels_mapping = {} # {floor_name: num_floors_below}
        for f in all_floors:
            num_below = 0
            for f_other in all_floors:
                if f != f_other and f"(above {f} {f_other})" in task.static:
                    # f_other is below f if f is above f_other
                    num_below += 1
            floor_levels_mapping[f] = num_below

        # Sort floors by the number of floors below them to assign levels
        # The floor with 0 floors below is level 1, 1 floor below is level 2, etc.
        sorted_floors = sorted(floor_levels_mapping.keys(), key=lambda f: floor_levels_mapping[f])

        for level, floor_name in enumerate(sorted_floors, 1):
            self.floor_to_level[floor_name] = level
            self.level_to_floor[level] = floor_name

        if not self.floor_to_level and (task.initial_state or task.goals):
             # Only warn if there are objects/goals but no floors found
             print("Warning: Could not parse floor levels. Heuristic may be inaccurate.")


        # 2. Parse passenger destinations from static facts
        self.goal_locations = {}
        for fact in task.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination = get_parts(fact)
                self.goal_locations[passenger] = destination

        # 3. Parse initial origins from initial state
        self.initial_origins = {}
        for fact in task.initial_state:
             if match(fact, "origin", "*", "*"):
                 _, passenger, origin = get_parts(fact)
                 self.initial_origins[passenger] = origin

        # 4. Collect all passengers mentioned in initial state or static destin facts
        self.all_passengers = set(self.goal_locations.keys()) | set(self.initial_origins.keys())
        # Also add passengers from initial state 'boarded' or 'served' if any
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ["boarded", "served"] and len(parts) > 1:
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown or floor is unparsed, heuristic is broken.
        # Return a large value or number of unserved passengers as a penalty.
        current_lift_level = self.floor_to_level.get(current_lift_floor, -1)
        if current_lift_level == -1:
             # Fallback: Count unserved passengers as a basic estimate
             unserved_count = sum(1 for p in self.all_passengers if "(served {})".format(p) not in state)
             # Add a large penalty multiplier if floor levels are missing
             return unserved_count * 1000 if self.floor_to_level else unserved_count


        total_cost = 0

        # Build current passenger status maps for quick lookup
        passenger_is_served = {p: False for p in self.all_passengers}
        passenger_is_boarded = {p: False for p in self.all_passengers}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "served" and len(parts) > 1 and parts[1] in self.all_passengers:
                passenger_is_served[parts[1]] = True
            elif parts[0] == "boarded" and len(parts) > 1 and parts[1] in self.all_passengers:
                passenger_is_boarded[parts[1]] = True

        # Iterate through all known passengers and sum their individual costs
        for passenger in self.all_passengers:
            if passenger_is_served[passenger]:
                continue # Passenger is already served

            destination_floor = self.goal_locations.get(passenger)
            if destination_floor is None:
                 # This passenger has no destination defined, likely an issue with problem file
                 # or they are not part of the main goal. Skip them.
                 continue

            destination_level = self.floor_to_level.get(destination_floor, -1)
            if destination_level == -1:
                 # Destination floor not found in parsed levels. Skip this passenger.
                 print(f"Warning: Unknown destination floor '{destination_floor}' for passenger '{passenger}'.")
                 continue

            if passenger_is_boarded[passenger]:
                # Passenger is boarded, needs to reach destination and depart
                # Cost = moves from current lift floor to destination + 1 (depart)
                cost_for_passenger = abs(current_lift_level - destination_level) + 1
                total_cost += cost_for_passenger

            else:
                 # Passenger is not served and not boarded. They must be waiting at their origin.
                 # Their origin is the one from the initial state.
                 origin_floor = self.initial_origins.get(passenger)
                 if origin_floor is None:
                      # Passenger is unserved/unboarded but wasn't in initial origins.
                      # This is unexpected in a standard miconic problem. Skip.
                      print(f"Warning: Initial origin not found for unserved/unboarded passenger '{passenger}'.")
                      continue

                 origin_level = self.floor_to_level.get(origin_floor, -1)
                 if origin_level == -1:
                      # Origin floor not found in parsed levels. Skip this passenger.
                      print(f"Warning: Unknown origin floor '{origin_floor}' for passenger '{passenger}'.")
                      continue

                 # Cost = moves from current lift floor to origin + 1 (board)
                 #        + moves from origin to destination + 1 (depart)
                 cost_for_passenger = abs(current_lift_level - origin_level) + 1
                 cost_for_passenger += abs(origin_level - destination_level) + 1
                 total_cost += cost_for_passenger

        return total_cost

