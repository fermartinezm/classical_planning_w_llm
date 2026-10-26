import sys
import os
from fnmatch import fnmatch

# Ensure the path includes the directory containing heuristic_base
# Assuming the script is run from a directory where 'heuristics' is a subdirectory
# or that the necessary paths are already set up.
# If heuristic_base is in the same directory, this might not be needed.
# sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Adjust as needed

try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Fallback if the import structure is different
    # This might happen if the file is run directly or the environment setup is different
    # You might need to adjust the path based on your project structure
    class Heuristic: # Dummy base class if import fails
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(at obj loc)" -> ["at", "obj", "loc"]
    """
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    Estimates the cost to serve all passengers by summing individual estimated costs
    for each unserved passenger. The cost for a passenger includes boarding (if needed),
    departing, and the lift movement required to pick them up (if needed) and drop
    them off at their destination, calculated from the lift's current position.

    # Assumptions
    - The `(above f_lower f_higher)` predicate means `f_higher` is directly above `f_lower`,
      implying a linear arrangement of floors.
    - The cost of each action (up, down, board, depart) is 1.
    - The heuristic does not need to be admissible. It aims to guide a greedy search effectively.
    - Passengers' destinations (`destin`) are static.
    - The problem instance describes a valid, connected floor structure.

    # Heuristic Initialization
    - Parses static facts to determine passenger destinations (`destin`).
    - Parses static `(above f_lower f_higher)` facts to build a representation of floor
      levels and compute distances between floors. It finds the bottom-most floor and
      assigns levels incrementally upwards.
    - Stores all passenger names found in static facts or goals.
    - Stores all floor names.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the current floor of the lift (`lf`) from the state. If the lift
        location is unknown, return infinity (invalid state for this heuristic).
    2.  Determine the status of each passenger `p` (served, boarded, or waiting at origin)
        by parsing the current state predicates (`served`, `boarded`, `origin`).
    3.  Initialize `total_heuristic_value = 0`.
    4.  For each passenger `p` identified during initialization:
        a.  If `p` is already served (`served p` is true), add 0 to the total cost.
        b.  If `p` is currently boarded (`boarded p` is true):
            i.  Find the destination floor `df` of `p` using the precomputed `destin` map.
            ii. Calculate the vertical distance `d = distance(lf, df)` using precomputed floor levels.
            iii.Add `1 (for the depart action) + d (for move actions)` to `total_heuristic_value`.
        c.  If `p` is currently waiting at their origin floor (`origin p of` is true):
            i.  Find the origin floor `of` of `p` from the state.
            ii. Find the destination floor `df` of `p` using the precomputed `destin` map.
            iii.Calculate the distance from the current lift floor to the origin: `d1 = distance(lf, of)`.
            iv. Calculate the distance from the origin floor to the destination floor: `d2 = distance(of, df)`.
            v.  Add `1 (board action) + 1 (depart action) + d1 (move to origin) + d2 (move to destination)`
                to `total_heuristic_value`.
        d.  If a passenger is unserved but neither boarded nor waiting (an unexpected state),
            ignore them or handle as an error case. Assume valid states.
    5.  After iterating through all passengers, check if the calculated `total_heuristic_value` is 0.
        If it is 0 but the state is not a goal state (checked by comparing served passengers
        against goal requirements), return 1 to ensure non-goal states have a non-zero heuristic value.
        Otherwise, return the calculated `total_heuristic_value`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.
        - Extracts passenger destinations.
        - Builds floor level mapping based on 'above' predicates.
        - Identifies all passengers and floors.
        """
        super().__init__(task) # Initialize base class if necessary
        self.goals = task.goals
        static_facts = task.static

        self.passengers = set()
        self.destin = {} # passenger -> destination_floor
        self.floors = set()
        self.floor_levels = {} # floor -> level (int)

        # Temporary structures to build floor graph
        higher_floor = {} # lower_floor -> higher_floor (directly above)
        lower_floor = {}  # higher_floor -> lower_floor (directly below)

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "destin":
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor
                self.passengers.add(passenger)
                self.floors.add(floor)
            elif predicate == "above":
                # Assuming (above f_lower f_higher) means f_higher is directly above f_lower
                f_lower, f_higher = parts[1], parts[2]
                higher_floor[f_lower] = f_higher
                lower_floor[f_higher] = f_lower
                self.floors.add(f_lower)
                self.floors.add(f_higher)

        # Ensure all passengers mentioned in goals are tracked
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "served":
                 self.passengers.add(parts[1])

        # Add floors from passenger origins if not already present (though they should be)
        # This requires access to initial state, which isn't typically part of static info.
        # We rely on 'above' and 'destin' to define all relevant floors.

        # Determine floor levels assuming a linear structure
        if not self.floors:
             # No floors defined, heuristic is trivial (0) or problem is invalid
             return

        # Find the bottom floor: a floor that is never the 'higher' floor in an 'above' fact,
        # or equivalently, is not a key in lower_floor map.
        bottom_floor = None
        potential_bottoms = self.floors - set(lower_floor.keys())

        if len(potential_bottoms) == 1:
            bottom_floor = potential_bottoms.pop()
        elif not potential_bottoms and len(self.floors) == 1: # Handle single-floor case
             bottom_floor = list(self.floors)[0]
        elif not potential_bottoms and not self.floors: # No floors
             pass # Already handled
        else:
            # This might indicate multiple disconnected floor segments or a non-linear structure.
            # For standard miconic, we expect a single linear chain.
            # Fallback: Try finding a floor that is not a value in higher_floor map.
            potential_bottoms_alt = self.floors - set(higher_floor.values())
            if len(potential_bottoms_alt) == 1:
                 bottom_floor = potential_bottoms_alt.pop()
            elif self.floors:
                 # If still ambiguous, raise error or make a best guess.
                 # Choosing arbitrarily might lead to incorrect distances.
                 # Let's raise an error for clarity in unexpected situations.
                 raise ValueError("Could not determine a unique bottom floor. Check 'above' predicates.")

        # Perform BFS from the bottom floor to assign levels
        if bottom_floor is not None:
            queue = [(bottom_floor, 0)]
            visited = {bottom_floor}
            self.floor_levels[bottom_floor] = 0
            processed_floors = 1

            while queue:
                curr_f, curr_level = queue.pop(0)

                # Check floor directly above
                if curr_f in higher_floor:
                    next_f = higher_floor[curr_f]
                    if next_f not in visited:
                        visited.add(next_f)
                        self.floor_levels[next_f] = curr_level + 1
                        queue.append((next_f, curr_level + 1))
                        processed_floors += 1

            # Sanity check: ensure all known floors were assigned a level
            if processed_floors != len(self.floors):
                unprocessed = self.floors - visited
                # This indicates disconnected floors or issues with 'above' predicates.
                raise ValueError(f"Floor structure seems disconnected. Unprocessed floors: {unprocessed}")

    def _get_distance(self, floor1, floor2):
        """Calculates the number of 'up'/'down' actions between two floors."""
        if floor1 not in self.floor_levels or floor2 not in self.floor_levels:
            # This should not happen if initialization was successful and inputs are valid floors
            raise ValueError(f"Cannot calculate distance: Unknown floor level for {floor1} or {floor2}.")
        
        level1 = self.floor_levels[floor1]
        level2 = self.floor_levels[floor2]
        return abs(level1 - level2)

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state
        total_cost = 0

        # 1. Find current lift location
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                lift_floor = parts[1]
                break

        if lift_floor is None or lift_floor not in self.floor_levels:
             # Lift location unknown or invalid, cannot compute heuristic reliably.
             return float('inf')

        # 2. Determine passenger status
        boarded_passengers = set()
        waiting_passengers = {} # passenger -> origin_floor
        served_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "boarded":
                boarded_passengers.add(parts[1])
            elif predicate == "origin":
                # Ensure passenger is known (should be, from init)
                if parts[1] in self.passengers:
                    waiting_passengers[parts[1]] = parts[2]
            elif predicate == "served":
                served_passengers.add(parts[1])

        # 3. Calculate cost for each unserved passenger
        goal_achieved = True # Assume goal is achieved until proven otherwise
        for p in self.passengers:
            # Check if this passenger needs to be served for the goal
            is_goal_passenger = False
            for goal in self.goals:
                goal_parts = get_parts(goal)
                if goal_parts[0] == "served" and goal_parts[1] == p:
                    is_goal_passenger = True
                    break

            if p in served_passengers:
                continue # This passenger is already served

            # If a passenger required by the goal is not served, the goal is not achieved
            if is_goal_passenger:
                goal_achieved = False

            # Get passenger's destination
            dest_floor = self.destin.get(p)
            if dest_floor is None:
                # Should not happen for passengers identified in init
                # print(f"Warning: Destination unknown for unserved passenger {p}. Skipping.")
                continue

            if p in boarded_passengers:
                # Cost = 1 depart action + movement actions
                distance = self._get_distance(lift_floor, dest_floor)
                total_cost += 1 + distance
            elif p in waiting_passengers:
                # Cost = 1 board + 1 depart + move to origin + move to destination
                origin_floor = waiting_passengers[p]
                if origin_floor not in self.floor_levels:
                     # Origin floor is invalid?
                     return float('inf') # Cannot compute heuristic

                dist_to_origin = self._get_distance(lift_floor, origin_floor)
                dist_origin_to_dest = self._get_distance(origin_floor, dest_floor)
                total_cost += 1 + 1 + dist_to_origin + dist_origin_to_dest
            else:
                # Unserved passenger is not boarded and not waiting at origin.
                # This state might be unusual or indicate an error in the problem/domain logic.
                # For robustness, could assign a default high cost or ignore.
                # Let's assume valid states where unserved passengers are either boarded or waiting.
                # print(f"Warning: Unserved passenger {p} in unexpected state (not boarded, not waiting).")
                pass # Ignore this passenger for cost calculation for now

        # 5. Final adjustments
        if goal_achieved:
            # If all goal passengers are served, the heuristic value should be 0.
            return 0
        elif total_cost == 0:
            # If the calculation resulted in 0 but the goal is not achieved,
            # return 1. This ensures non-goal states have a cost > 0.
            # This might happen if, e.g., a passenger is boarded at their destination
            # floor but hasn't departed yet. The cost calculation correctly includes '1' for depart.
            # So, total_cost should only be 0 if all relevant passengers are served.
            # This check handles potential edge cases or if no passengers are relevant to the goal.
            return 1
        else:
            return total_cost

