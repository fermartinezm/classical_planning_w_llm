import math
from collections import deque
# Assuming the planner infrastructure provides the Heuristic base class
# and the Task class structure as shown in the examples.
# If Heuristic base class is not available, this class can stand alone.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts the components of a PDDL fact string by removing parentheses
    and splitting by space.
    Example: "(at obj loc)" -> ["at", "obj", "loc"]
    """
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state,
    where all passengers are served (i.e., have reached their destination).
    The estimate is based on the number of board and depart actions needed for
    unserved passengers, plus an estimate of the lift's travel distance required
    to visit all necessary floors (origins of waiting passengers and destinations
    of waiting or boarded passengers). It is designed for Greedy Best-First Search
    and is not necessarily admissible.

    # Assumptions
    - The `(above f1 f2)` predicate means floor `f1` is directly one level above floor `f2`.
    - The floors form a single, linear sequence, allowing unique level assignments.
    - The lift moves between adjacent floors using 'up' and 'down' actions (cost 1 per move).
    - Passengers must first be at their origin, then board the lift, travel, and finally depart at their destination.

    # Heuristic Initialization
    - Parses static facts from the task definition to build internal data structures:
        - `self.destinations`: A dictionary mapping each passenger to their destination floor.
        - `self.floor_levels`: A dictionary mapping each floor object to its integer level (height), assuming the lowest floor is level 0. This is computed by analyzing the `(above f1 f2)` static predicates.
    - Stores the goal conditions (`self.goals`) for checking goal achievement.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check Goal:** If the current state satisfies all goal conditions (`(served p)` for all relevant passengers), the heuristic value is 0.
    2.  **Analyze State:** Identify the current state components:
        - The lift's current floor (`lift_floor`).
        - Passengers currently waiting at their origin (`current_waiting`: dict mapping passenger to origin floor).
        - Passengers currently boarded on the lift (`current_boarded`: set of passengers).
        - Passengers already served (`served_passengers`: set of passengers).
        These sets/dicts only include passengers relevant to the goal.
    3.  **Calculate Board/Depart Costs:** Estimate the minimum number of board/depart actions needed.
        - Each passenger in `current_waiting` needs one `board` action and one `depart` action later. Cost: `len(current_waiting) * 2`.
        - Each passenger in `current_boarded` needs one `depart` action. Cost: `len(current_boarded) * 1`.
        - Total board/depart cost = `2 * len(current_waiting) + len(current_boarded)`.
    4.  **Calculate Movement Cost:** Estimate the lift's travel distance.
        a.  **Identify Target Floors:** Determine the set of floors the lift must visit to serve the remaining passengers:
            - The origin floors of all passengers in `current_waiting`.
            - The destination floors of all passengers in `current_waiting`.
            - The destination floors of all passengers in `current_boarded`.
            Let `AllTargetFloors` be the union of these floor sets.
        b.  **Determine Travel Range:** If `AllTargetFloors` is not empty, find the minimum (`min_target_level`) and maximum (`max_target_level`) floor levels among these target floors using the precomputed `self.floor_levels`.
        c.  **Estimate Travel Distance:** Calculate the estimated travel distance. The lift starts at `level_lf = self.floor_levels[lift_floor]`. The estimate assumes the lift travels from its current position to one end of the required range (`min_target_level` or `max_target_level`) and then sweeps across the range to the other end. The minimum cost of the two possible sweep directions (start from min or start from max) is taken:
            `cost_go_min_sweep = abs(level_lf - min_target_level) + (max_target_level - min_target_level)`
            `cost_go_max_sweep = abs(level_lf - max_target_level) + (max_target_level - min_target_level)`
            `move_cost = min(cost_go_min_sweep, cost_go_max_sweep)` if targets exist, otherwise 0.
        d.  Add `move_cost` to the heuristic value.
    5.  **Return Total:** The final heuristic value is the sum of the board/depart costs and the estimated movement cost (`h = h_board_depart + h_move`). If the state is invalid (e.g., missing floor levels), return infinity.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information from the task.
        - Extracts passenger destinations from `(destin p f)` facts.
        - Computes floor levels based on `(above f1 f2)` predicates.
        """
        self.goals = task.goals
        static_facts = task.static
        all_facts = task.facts # All possible predicates in the domain/problem

        # Extract passenger destinations
        self.destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor

        # --- Compute floor levels ---
        self.floor_levels = {}
        floors = set()
        successors = {} # floor -> set of floors directly above
        predecessors = {} # floor -> set of floors directly below

        # Find all floor objects by checking relevant predicates and types
        for fact in all_facts:
             parts = get_parts(fact)
             # Check predicates involving floors
             if parts[0] in ['lift-at', 'origin', 'destin'] and len(parts) >= 2:
                 # The object in the last position for these predicates is a floor
                 floors.add(parts[-1])
             elif parts[0] == 'above' and len(parts) == 3:
                 floors.add(parts[1])
                 floors.add(parts[2])
             # Check for type predicates if they exist in all_facts
             elif parts[0] == 'floor' and len(parts) == 2:
                 floors.add(parts[1])

        # Initialize adjacency lists
        for f in floors:
            successors[f] = set()
            predecessors[f] = set()

        # Populate adjacency lists from 'above' relations
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above' and len(parts) == 3:
                f_up, f_down = parts[1], parts[2]
                if f_up in floors and f_down in floors:
                    # f_up is directly above f_down
                    successors[f_down].add(f_up)
                    predecessors[f_up].add(f_down)
                else:
                    print(f"Warning: Ignoring 'above' relation with unknown floor: {fact}")


        # Find bottom floors (level 0) - those with no floor below them
        bottom_floors = {f for f in floors if not predecessors[f]}

        if not bottom_floors and floors:
            # Fallback if no clear bottom floor (e.g., cyclic, incomplete PDDL)
            print("Warning: No bottom floor found based on 'above' predicates. "
                  "Attempting fallback or assuming single floor.")
            if len(floors) == 1:
                 bottom_floors = floors # Single floor is its own bottom floor
            else:
                 # Try finding floors not mentioned as being 'above' any other floor
                 mentioned_as_up = {parts[1] for fact in static_facts if get_parts(fact)[0] == 'above'}
                 possible_bottom = floors - mentioned_as_up
                 if possible_bottom:
                     bottom_floors = possible_bottom
                     print(f"Fallback: Using floors not mentioned as 'up': {bottom_floors}")
                 else:
                     # Last resort: pick lexicographically first floor
                     if floors:
                         fallback_floor = sorted(list(floors))[0]
                         bottom_floors = {fallback_floor}
                         print(f"Fallback: Using lexicographically first floor: {fallback_floor}")

        # Perform BFS starting from bottom floors to assign levels
        queue = deque([(f, 0) for f in bottom_floors])
        visited = set(bottom_floors)
        processed_count = 0

        while queue:
            current_floor, current_level = queue.popleft()
            if current_floor in self.floor_levels: # Should not happen in linear structure
                continue
            self.floor_levels[current_floor] = current_level
            processed_count += 1
            for next_floor in successors.get(current_floor, set()):
                if next_floor not in visited:
                    visited.add(next_floor)
                    queue.append((next_floor, current_level + 1))

        # Validation: Check if all floors were assigned a level
        if processed_count != len(floors):
            unprocessed = floors - set(self.floor_levels.keys())
            print(f"Error: Miconic heuristic could not assign levels to all floors. "
                  f"Processed: {processed_count}, Total: {len(floors)}. "
                  f"Floors without levels: {unprocessed}. Heuristic may return infinity.")
            # Mark that initialization might be incomplete, heuristic might fail
            self.initialization_incomplete = True
        else:
            self.initialization_incomplete = False
            # print(f"Floor levels computed: {self.floor_levels}") # Debug print


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Returns an estimate of the remaining actions, or infinity if the state
        is invalid or requires floors with unassigned levels.
        """
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # If floor level calculation failed, return infinity
        if self.initialization_incomplete:
            return float('inf')

        # --- Analyze State ---
        lift_floor = None
        passengers_at_origin = {} # passenger -> origin_floor
        passengers_boarded = set() # passenger
        passengers_served = set() # passenger

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == 'lift-at' and len(parts) == 2:
                lift_floor = parts[1]
            elif predicate == 'origin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                passengers_at_origin[passenger] = floor
            elif predicate == 'boarded' and len(parts) == 2:
                passenger = parts[1]
                passengers_boarded.add(passenger)
            elif predicate == 'served' and len(parts) == 2:
                passenger = parts[1]
                passengers_served.add(passenger)

        # Filter out served passengers to find who still needs service
        current_waiting = {p: f for p, f in passengers_at_origin.items() if p not in passengers_served}
        current_boarded = {p for p in passengers_boarded if p not in passengers_served}

        # Check for invalid state components
        if lift_floor is None:
            print("Error: Lift location not found in state.")
            return float('inf')
        if lift_floor not in self.floor_levels:
            print(f"Error: Lift floor '{lift_floor}' has no assigned level.")
            return float('inf')

        # --- Calculate Heuristic Components ---
        h_board_depart = 0
        h_move = 0

        # 1. Board/Depart Costs
        num_waiting = len(current_waiting)
        num_boarded = len(current_boarded)
        # Each waiting passenger needs 1 board + 1 depart
        # Each boarded passenger needs 1 depart
        h_board_depart = num_waiting * 2 + num_boarded * 1

        # 2. Movement Cost
        origins = set(current_waiting.values())
        destinations_waiting = set()
        for p in current_waiting:
            if p in self.destinations:
                destinations_waiting.add(self.destinations[p])
            else:
                 print(f"Warning: Destination not found for waiting passenger {p}")

        destinations_boarded = set()
        for p in current_boarded:
             if p in self.destinations:
                 destinations_boarded.add(self.destinations[p])
             else:
                 print(f"Warning: Destination not found for boarded passenger {p}")

        all_target_floors = origins.union(destinations_waiting).union(destinations_boarded)

        if not all_target_floors:
            h_move = 0
        else:
            # Check if all required floors have levels
            required_floors_for_move = all_target_floors.union({lift_floor})
            missing_levels = {f for f in required_floors_for_move if f not in self.floor_levels}
            if missing_levels:
                 print(f"Error: Missing floor levels for calculation: {missing_levels}")
                 return float('inf')

            level_lf = self.floor_levels[lift_floor]
            target_levels = {self.floor_levels[f] for f in all_target_floors}
            min_target_level = min(target_levels)
            max_target_level = max(target_levels)

            # Estimate travel: min distance to reach the range [min, max] + sweep the range
            cost_go_min_sweep = abs(level_lf - min_target_level) + (max_target_level - min_target_level)
            cost_go_max_sweep = abs(level_lf - max_target_level) + (max_target_level - min_target_level)
            h_move = min(cost_go_min_sweep, cost_go_max_sweep)

        # --- Total Heuristic Value ---
        total_heuristic = h_board_depart + h_move

        # Ensure heuristic is > 0 if goal is not met.
        # If h=0 but goal not met, it implies an issue or edge case not covered.
        # For instance, if all passengers are served but the goal includes other conditions?
        # Miconic goals are typically just (served p) for all p.
        # If h=0, it means num_waiting=0, num_boarded=0, and h_move=0.
        # This implies no unserved passengers and no required movement, which should only happen at the goal.
        if total_heuristic == 0 and not (self.goals <= state):
             # This case should ideally not happen if logic is correct. Return 1 as minimum cost.
             # print("Warning: Heuristic calculated 0 for a non-goal state.")
             return 1

        return total_heuristic

