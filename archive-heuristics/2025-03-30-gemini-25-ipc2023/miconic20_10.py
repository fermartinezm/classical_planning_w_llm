import sys
import os
from fnmatch import fnmatch
from collections import defaultdict, deque

# Ensure the base class 'Heuristic' is available.
# If the script is run standalone, a dummy class is defined.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    class Heuristic:
        """Dummy base class for Heuristic."""
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError

# Helper functions for parsing PDDL facts
def get_parts(fact: str) -> list[str]:
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(lift-at f2)" -> ["lift-at", "f2"]
    """
    # Remove parentheses and split, handling potential extra spaces
    return fact.strip()[1:-1].split()

def match(fact: str, *pattern: str) -> bool:
    """
    Checks if a PDDL fact string matches a given pattern.
    Uses fnmatch for wildcard matching ('*').
    Example: match("(at ball1 rooma)", "at", "*", "rooma") -> True
    """
    parts = get_parts(fact)
    if len(parts) != len(pattern):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    This heuristic estimates the remaining cost to reach the goal state (all required
    passengers served). The estimate is calculated by summing:
    1. The number of 'board' actions still required (one for each waiting passenger).
    2. The number of 'depart' actions still required (one for each unserved passenger).
    3. An estimate of the minimum lift movement cost, calculated as the vertical distance
       (span in floor levels) between the highest and lowest floors relevant to the
       remaining tasks (current lift position, origins of waiting passengers, destinations
       of unserved passengers).
    This heuristic is designed for Greedy Best-First Search and is not guaranteed to be admissible.

    # Assumptions
    - The lift moves vertically between adjacent floors using 'up' and 'down' actions.
    - The static predicate '(above f1 f2)' means floor f1 is vertically higher than floor f2.
    - The goal requires a specific set of passengers to be in the '(served p)' state.
    - The floor layout forms a connected structure (possibly with branches).

    # Heuristic Initialization
    - The constructor (`__init__`) preprocesses the static information from the task:
      - It parses '(destin p f)' facts to store the destination floor for each passenger.
      - It identifies the set of passengers that must be served from the goal conditions.
      - It parses '(above f1 f2)' facts to understand the floor hierarchy and identify all floors.
      - It determines floor adjacency by identifying pairs (f1, f2) where one is 'above'
        the other, and no intermediate floor f_mid exists between them according to 'above'.
      - It calculates a numerical 'level' for each floor using Breadth-First Search (BFS)
        based on adjacency and the 'above' relationship to determine relative heights.
        The levels are normalized so the lowest floor has level 0. This allows efficient
        calculation of vertical distance between floors. Handles potentially disconnected floor graphs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Goal Check: The `__call__` method first checks if the current state `node.state`
       already satisfies all goal conditions (i.e., all required passengers are 'served').
       If so, the goal is reached, and the heuristic value is 0.
    2. State Parsing: If the goal is not reached, the method parses the current state to find:
       - `lift_floor`: The current floor of the lift from '(lift-at f)'.
       - `waiting_passengers`: A dictionary mapping passengers `p` to their origin floor `f`
         for those who are in state '(origin p f)' and are required for the goal but not yet served.
       - `boarded_passengers`: A set of passengers `p` who are in state '(boarded p)' and
         are required for the goal but not yet served.
       - `served_passengers`: A set of passengers `p` who are in state '(served p)' and
         are required for the goal.
    3. Action Cost Calculation:
       - Initialize the heuristic value `H = 0`.
       - Add the number of `waiting_passengers` to `H`. Each waiting passenger requires one 'board' action.
       - Add the number of unserved goal passengers (`len(goal_passengers - served_passengers)`)
         to `H`. Each unserved passenger (whether waiting or boarded) requires one 'depart' action eventually.
    4. Movement Cost Estimation:
       - Identify the set of `target_floors`: This includes the origin floors of all `waiting_passengers`
         and the destination floors (looked up from precomputed `self.destinations`) of all unserved passengers.
       - If `target_floors` is not empty:
         - Determine the set of `relevant_floors` = `target_floors` union `{lift_floor}`.
         - Collect the floor levels for all `relevant_floors` that have a precomputed level.
         - If all relevant floors have levels, estimate the movement cost as the total vertical span:
           `movement_cost = max(levels) - min(levels)`. Add `movement_cost` to `H`.
         - If some relevant floors are missing levels (e.g., disconnected graph) or no relevant floors
           have levels, add a fallback penalty equal to the number of `target_floors` to `H`.
       - Error Handling: If the lift's location is missing or a passenger's destination is undefined,
         appropriate warnings are printed or fallback values (infinity or penalties) are used.
    5. Return Value: The method returns the final calculated heuristic value `H`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse destinations
        self.destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    self.destinations[parts[1]] = parts[2] # passenger -> destination_floor

        # 2. Identify goal passengers
        self.goal_passengers = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                parts = get_parts(goal)
                if len(parts) == 2:
                    self.goal_passengers.add(parts[1])

        # 3. Determine floor levels
        self.floor_levels = {}
        self.all_floors = set()
        above_pairs = set() # Store (higher_floor, lower_floor) pairs

        # Collect all floors and 'above' relationships
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     f_higher, f_lower = parts[1], parts[2]
                     above_pairs.add((f_higher, f_lower))
                     self.all_floors.add(f_higher)
                     self.all_floors.add(f_lower)

        # Discover floors from other predicates if 'above' is missing
        if not self.all_floors:
             potential_floors = set()
             # Check initial state and static facts for floors mentioned
             for fact in task.initial_state | static_facts:
                 parts = get_parts(fact)
                 # Look for floors in relevant predicates
                 if parts[0] == "lift-at" and len(parts) == 2:
                     potential_floors.add(parts[1])
                 elif parts[0] in ["origin", "destin"] and len(parts) == 3:
                     potential_floors.add(parts[2])
             self.all_floors.update(potential_floors)

        if not self.all_floors:
             # No floors found, likely an empty or invalid problem
             print("Warning: No floors found in the problem description.")
             return # self.floor_levels remains empty

        # Infer adjacency based on 'above' facts
        adj = defaultdict(set)
        for f1 in self.all_floors:
            for f2 in self.all_floors:
                if f1 == f2: continue
                # Check if f1 is directly above f2
                if (f1, f2) in above_pairs:
                    is_directly_above = True
                    for f_mid in self.all_floors:
                        # Check if f_mid is strictly between f1 and f2
                        if f_mid != f1 and f_mid != f2 and \
                           (f1, f_mid) in above_pairs and (f_mid, f2) in above_pairs:
                            is_directly_above = False
                            break
                    if is_directly_above:
                        adj[f1].add(f2)
                        adj[f2].add(f1) # Store symmetric adjacency for BFS

                # Check if f2 is directly above f1 (avoid redundant checks/adds)
                elif (f2, f1) in above_pairs:
                     if f1 not in adj[f2]: # Check if already added via the other direction
                         is_directly_above = True
                         for f_mid in self.all_floors:
                             # Check if f_mid is strictly between f2 and f1
                             if f_mid != f1 and f_mid != f2 and \
                                (f2, f_mid) in above_pairs and (f_mid, f1) in above_pairs:
                                 is_directly_above = False
                                 break
                         if is_directly_above:
                             adj[f1].add(f2)
                             adj[f2].add(f1)

        # Calculate levels using BFS for each connected component
        visited_overall = set()
        all_levels_relative = {}

        for floor_start_node in self.all_floors:
            if floor_start_node in visited_overall:
                continue # Already processed this component

            # Start BFS for a new component
            component_levels = {floor_start_node: 0}
            queue = deque([(floor_start_node, 0)]) # (floor, relative_level)
            visited_component = {floor_start_node}

            while queue:
                curr_f, curr_l = queue.popleft()
                for neighbor_f in adj[curr_f]: # Use inferred adjacency
                     if neighbor_f not in visited_component:
                        level_diff = 0
                        if (neighbor_f, curr_f) in above_pairs: # neighbor is higher
                            level_diff = 1
                        elif (curr_f, neighbor_f) in above_pairs: # neighbor is lower
                            level_diff = -1
                        else:
                            # This case indicates an issue with adjacency inference or 'above' facts
                            print(f"Warning: Inconsistency finding level difference between adjacent {curr_f} and {neighbor_f}")
                            continue # Skip this neighbor

                        neighbor_level = curr_l + level_diff
                        component_levels[neighbor_f] = neighbor_level
                        visited_component.add(neighbor_f)
                        queue.append((neighbor_f, neighbor_level))

            # Store levels for this component and mark as visited
            all_levels_relative.update(component_levels)
            visited_overall.update(visited_component)

        # Normalize levels globally so minimum is 0
        if all_levels_relative:
            min_level = min(all_levels_relative.values())
            self.floor_levels = {f: l - min_level for f, l in all_levels_relative.items()}
            # Assign level to floors potentially missed if graph was disconnected and not fully covered by BFS starts
            for f in self.all_floors:
                if f not in self.floor_levels:
                    print(f"Warning: Floor {f} was not reached by BFS, assigning default level 0.")
                    self.floor_levels[f] = 0 # Assign default level
        elif self.all_floors: # Handle case with floors but no relative levels (e.g., single floor)
             self.floor_levels = {f: 0 for f in self.all_floors}


    def __call__(self, node):
        state = node.state

        # Efficient goal check
        served_passengers = set() # Track served passengers found in state
        for fact in state:
             if match(fact, "served", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 2 and parts[1] in self.goal_passengers:
                     served_passengers.add(parts[1])

        if served_passengers == self.goal_passengers:
             return 0 # Goal reached

        # State parsing
        lift_floor = None
        waiting_passengers = {} # p -> origin_floor
        boarded_passengers = set() # p

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts
            predicate = parts[0]

            if predicate == "lift-at" and len(parts) == 2:
                lift_floor = parts[1]
            elif predicate == "origin" and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                # Consider only if it's a goal passenger and not already served
                if passenger in self.goal_passengers and passenger not in served_passengers:
                    waiting_passengers[passenger] = floor
            elif predicate == "boarded" and len(parts) == 2:
                passenger = parts[1]
                # Consider only if it's a goal passenger and not already served
                if passenger in self.goal_passengers and passenger not in served_passengers:
                    boarded_passengers.add(passenger)

        # Handle missing lift floor
        if lift_floor is None:
             unserved_passengers_check = self.goal_passengers - served_passengers
             if unserved_passengers_check:
                 print("Warning: Lift location unknown in a non-goal state.")
                 return float('inf') # Indicate error or potentially unsolvable state
             else:
                 return 0 # Should have been caught by goal check

        heuristic_value = 0
        unserved_passengers = self.goal_passengers - served_passengers

        if not unserved_passengers:
             return 0 # Should be caught by initial check

        # 1. Action Count
        num_waiting = len(waiting_passengers)
        num_unserved = len(unserved_passengers)

        heuristic_value += num_waiting # Cost for 'board' actions
        heuristic_value += num_unserved # Cost for 'depart' actions

        # 2. Movement Cost Estimation
        target_floors = set()
        # Add origins of waiting passengers
        target_floors.update(waiting_passengers.values())
        # Add destinations of unserved passengers
        for p in unserved_passengers:
            if p in self.destinations:
                 target_floors.add(self.destinations[p])
            else:
                # This indicates an issue with the problem definition (goal passenger has no destination)
                print(f"Warning: Unserved goal passenger {p} missing destination in static facts.")
                # Cannot accurately estimate movement. Add a penalty?
                heuristic_value += 1 # Small penalty for the missing info

        if target_floors:
            relevant_floors = target_floors | {lift_floor}
            relevant_levels = []
            missing_level = False
            for f in relevant_floors:
                if f in self.floor_levels:
                    relevant_levels.append(self.floor_levels[f])
                else:
                    # A floor needed for calculation doesn't have a level (e.g., disconnected graph)
                    print(f"Warning: Floor {f} needed for heuristic calculation has no precomputed level.")
                    missing_level = True
                    # No need to break, just flag it

            if not relevant_levels:
                 # None of the relevant floors had levels computed. Use fallback penalty.
                 heuristic_value += len(target_floors)
            elif missing_level:
                 # Some levels were missing. Use fallback penalty.
                 heuristic_value += len(target_floors)
            else:
                 # All relevant floors have levels, compute span
                 min_relevant_level = min(relevant_levels)
                 max_relevant_level = max(relevant_levels)
                 movement_cost = max_relevant_level - min_relevant_level
                 heuristic_value += movement_cost
        # If target_floors is empty, means all unserved passengers are boarded at their destination
        # (or error state). No movement cost needed, only depart cost (already added).

        return heuristic_value
