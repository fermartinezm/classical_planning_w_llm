import math # For float('inf')

from heuristics.heuristic_base import Heuristic
from task import Task

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the remaining cost by summing the number of board/depart
        actions needed for unserved passengers and the estimated minimum
        lift movement cost to visit all required floors (origins for unboarded,
        destinations for boarded).

    Assumptions:
        - Floor names follow the pattern 'fX' where X is an integer, and
          '(above fY fX)' implies Y > X. This allows mapping floor names to
          integer levels based on the number in the name.
        - The initial state contains '(origin p f)' for all passengers p.
        - The static facts contain '(destin p f)' for all passengers p.
        - The static facts contain '(above f1 f2)' facts defining the floor order.

    Heuristic Initialization:
        - Parses static facts to build a mapping from passenger to destination floor.
        - Parses initial state facts to build a mapping from passenger to initial origin floor.
        - Parses floor names and '(above f1 f2)' facts to build a mapping from
          floor name to an integer level, assuming 'fX' corresponds to level X.
        - Identifies all passengers and floors in the problem.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state and identify the lift's current floor.
        2. Identify all passengers who are not yet served.
        3. If no passengers are unserved, the heuristic is 0 (goal state).
        4. Separate unserved passengers into two groups: those who need to be
           picked up (not boarded) and those who need to be dropped off (boarded).
        5. Determine the set of floors the lift *must* visit: origin floors for
           passengers to be picked up, and destination floors for passengers
           to be dropped off.
        6. Calculate the number of board actions needed (equal to the number
           of passengers to be picked up).
        7. Calculate the number of depart actions needed (equal to the number
           of passengers to be dropped off).
        8. If there are floors to visit (step 5), find the minimum and maximum
           floor levels among them.
        9. Estimate the minimum lift movement cost: This is the distance from
           the current floor level to the closest required floor level (either
           min or max), plus the distance between the min and max required
           floor levels. If no floors need visiting, movement cost is 0.
        10. The total heuristic value is the sum of board actions, depart actions,
            and the estimated movement cost.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task

        # Data structures for static information
        self.destinations = {}  # passenger_name -> destin_floor_name
        self.origins_initial = {} # passenger_name -> origin_floor_name (from initial state)
        self.floor_levels = {}  # floor_name -> level_integer
        self.passengers = set() # set of all passenger names
        self.floors = set() # set of all floor names

        # Parse static facts and initial state
        self._parse_static_and_initial(task)

        # Ensure floor levels are populated correctly based on 'above' facts
        # A more robust way than just parsing 'fX' might be needed if floor names are arbitrary.
        # Let's stick to 'fX' parsing for now based on examples.
        self._assign_floor_levels()


    def _parse_static_and_initial(self, task: Task):
        """Parses static facts and initial state to populate data structures."""
        # Collect all objects first to identify passengers and floors
        all_objects = set()
        # task.facts contains all possible ground facts, listing all objects
        for fact_str in task.facts:
             parsed = self._parse_fact(fact_str)
             if parsed: # Check if parsing was successful
                 all_objects.update(parsed[1:]) # Add objects, skip predicate

        for obj in all_objects:
            if obj.startswith('p'): # Assuming passenger names start with 'p'
                self.passengers.add(obj)
            elif obj.startswith('f'): # Assuming floor names start with 'f'
                self.floors.add(obj)

        # Parse destinations from static facts
        for fact_str in task.static:
            if fact_str.startswith('(destin '):
                parsed = self._parse_fact(fact_str)
                if parsed and len(parsed) == 3: # Expecting (destin passenger floor)
                    _, p, f = parsed
                    self.destinations[p] = f

        # Parse initial origins from initial state
        for fact_str in task.initial_state:
             if fact_str.startswith('(origin '):
                parsed = self._parse_fact(fact_str)
                if parsed and len(parsed) == 3: # Expecting (origin passenger floor)
                    _, p, f = parsed
                    self.origins_initial[p] = f

        # Note: The 'above' facts define the floor order, but the simple
        # 'fX' parsing handles this implicitly if the assumption holds.
        # If floor names were arbitrary (e.g., 'ground', 'first', 'second'),
        # we would need to build a graph from 'above' facts and perform a
        # topological sort or BFS/DFS to assign levels.
        # For this domain, 'fX' parsing seems sufficient based on examples.


    def _assign_floor_levels(self):
        """Assigns integer levels to floors based on 'fX' name pattern."""
        # Assuming floor names are like 'f1', 'f2', etc.
        # Sort floors based on the integer part of their name.
        # Assign levels starting from 1.
        if not self.floors:
            return # No floors to assign levels

        try:
            # Extract the numeric part and sort
            floor_number_pairs = []
            for f in self.floors:
                if f.startswith('f') and f[1:].isdigit():
                    floor_number_pairs.append((int(f[1:]), f))
                else:
                    # Handle floors that don't match the pattern if necessary,
                    # or just skip them. For this heuristic, we need levels.
                    # If the problem guarantees 'fX' format, this else is not needed.
                    # Based on examples, let's assume 'fX' format for all floors.
                    # If not, the heuristic might be inaccurate.
                    # print(f"Warning: Floor name '{f}' does not match 'fX' pattern. Heuristic might be inaccurate.")
                    pass # Skip floors that don't match the expected pattern


            sorted_floors = [f for num, f in sorted(floor_number_pairs)]

            # If some floors were skipped because they didn't match the pattern,
            # the heuristic might be incomplete.
            if len(sorted_floors) != len(self.floors):
                 # print(f"Warning: Only {len(sorted_floors)} out of {len(self.floors)} floors matched 'fX' pattern. Heuristic might be inaccurate.")
                 pass # Continue with assigning levels for matched floors


            for i, floor in enumerate(sorted_floors):
                self.floor_levels[floor] = i + 1

        except ValueError:
            # This catch block might be redundant if the check inside try works,
            # but keep for safety against unexpected errors during int conversion.
            # print("Warning: Error parsing floor names to integers. Heuristic might be inaccurate.")
            # If parsing fails completely, floor_levels remains empty or incomplete.
            pass


    def _parse_fact(self, fact_str: str):
        """Parses a fact string into a list of strings [predicate, obj1, obj2, ...]."""
        # Remove leading/trailing brackets and split by space
        if not fact_str or not fact_str.startswith('(') or not fact_str.endswith(')'):
             # Invalid fact format
             return None
        return fact_str[1:-1].split()

    def __call__(self, node):
        """Computes the heuristic value for the given state."""
        state = node.state

        # 1. Identify current lift floor
        current_floor = None
        for fact_str in state:
            if fact_str.startswith('(lift-at '):
                parsed = self._parse_fact(fact_str)
                if parsed and len(parsed) == 2: # Expecting (lift-at floor)
                    _, current_floor = parsed
                    break

        # If lift-at fact is not found, the state is likely invalid or a dead end.
        # Or if the floor name is not recognized/assigned a level.
        if current_floor is None or current_floor not in self.floor_levels:
             # print(f"Warning: Could not determine current lift floor ({current_floor}) or its level.")
             return math.inf # Should indicate a dead end or invalid state

        current_level = self.floor_levels[current_floor]


        # 2. Identify unserved passengers
        unserved_passengers = {p for p in self.passengers if '(served ' + p + ')' not in state}

        # 3. Goal check
        if not unserved_passengers:
            return 0 # Goal state

        # 4. Separate passengers to pickup and dropoff
        passengers_to_pickup = set()
        passengers_to_dropoff = set()

        for p in unserved_passengers:
            if '(boarded ' + p + ')' in state:
                passengers_to_dropoff.add(p)
            else:
                # If not served and not boarded, must be at origin
                # Ensure origin is known (should be from initial state)
                if p in self.origins_initial:
                    passengers_to_pickup.add(p)
                # else: # Passenger has no initial origin? Invalid problem setup?
                    # print(f"Warning: Passenger {p} is unserved and not boarded but has no initial origin.")
                    # Returning inf might be appropriate here, but let's assume valid problems.


        # 5. Determine required floors
        required_pickup_floors = {self.origins_initial[p] for p in passengers_to_pickup if p in self.origins_initial}
        required_dropoff_floors = {self.destinations[p] for p in passengers_to_dropoff if p in self.destinations}
        all_required_floors = required_pickup_floors | required_dropoff_floors

        # 6. Calculate board actions
        board_actions_needed = len(passengers_to_pickup)

        # 7. Calculate depart actions
        depart_actions_needed = len(passengers_to_dropoff)

        # 8. & 9. Estimate movement cost
        movement_cost = 0
        if all_required_floors:
            # Ensure all required floors have known levels
            required_levels = []
            for f in all_required_floors:
                if f in self.floor_levels:
                    required_levels.append(self.floor_levels[f])
                # else: # Required floor level not found? Invalid problem?
                    # print(f"Warning: Required floor {f} level not found.")
                    # return math.inf # Indicate unsolvable/invalid state

            if required_levels: # Check if list is not empty after filtering
                min_required_level = min(required_levels)
                max_required_level = max(required_levels)

                # Cost to reach the range + cost to traverse the range
                # This assumes the lift can pick up/drop off passengers opportunistically
                # while traversing the range [min_required_level, max_required_level].
                # The minimum travel to cover the entire range starting from current_level is:
                # distance from current to one end + distance between ends.
                dist_to_min = abs(current_level - min_required_level)
                dist_to_max = abs(current_level - max_required_level)
                range_dist = max_required_level - min_required_level

                # Option 1: Go to min, then traverse to max
                cost1 = dist_to_min + range_dist
                # Option 2: Go to max, then traverse to min
                cost2 = dist_to_max + range_dist

                movement_cost = min(cost1, cost2)
            # else: # No required floors had valid levels? This case is covered by the outer if all_required_floors:
                # print("Warning: No required floors had valid levels.")
                # return math.inf # Indicate unsolvable/invalid state


        # 10. Total heuristic value
        h_value = board_actions_needed + depart_actions_needed + movement_cost

        return h_value
