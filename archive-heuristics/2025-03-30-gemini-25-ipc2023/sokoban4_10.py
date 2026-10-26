import math
from collections import deque
from fnmatch import fnmatch
# Assuming the heuristic base class is available at this path.
# Ensure this import path is correct for the target environment.
from heuristics.heuristic_base import Heuristic

def get_parts(fact: str) -> list[str]:
    """Helper function to extract components of a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(at box1 loc1)" -> ["at", "box1", "loc1"]
    """
    # Handles potential empty strings or facts without parentheses
    if not fact or not fact.startswith("(") or not fact.endswith(")"):
        return []
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in Sokoban by summing two components:
    1. The sum of shortest path distances for each box from its current location to its target goal location. This estimates the total number of push actions required.
    2. The shortest path distance from the robot's current location to the nearest box that is not yet in its goal location. This estimates the number of initial move actions the robot needs to perform to start pushing.
    The heuristic aims to provide an informative (though not necessarily admissible) estimate for greedy best-first search.

    # Assumptions
    - The cost of a 'move' action (robot moves to adjacent clear cell) is 1.
    - The cost of a 'push' action (robot pushes box to adjacent clear cell) is 1.
    - The heuristic uses precomputed shortest path distances on the grid connectivity graph, ignoring dynamic obstacles like other boxes or the robot's exact position relative to the box for pushing. It assumes clear paths exist between locations if topologically possible.
    - Each box object (e.g., 'box1') has a unique, fixed goal location specified by an `(at boxN locM)` predicate in the problem's goal definition.
    - The heuristic does not explicitly detect dead-end states (e.g., boxes pushed into corners or against walls from which they cannot be moved without complex maneuvers or freeing up space). Such states might receive a finite heuristic value even if they are unsolvable.
    - The heuristic primarily focuses on achieving the box goal positions. If the goal involves only the robot's position or other conditions, the heuristic might be less informative after all boxes are placed.

    # Heuristic Initialization
    - Parses the `task.goals` to identify the target location for each box, storing them in `self.goal_locations`.
    - Parses the `task.static` facts (specifically 'adjacent' predicates) to build a graph representation (`self.adjacency_list`) of the grid connectivity.
    - Identifies the set of all unique location names (`self.all_locations`) mentioned across static facts, initial state, and goals to ensure all relevant parts of the map are considered.
    - Precomputes all-pairs shortest path distances between locations using Breadth-First Search (BFS) on the grid graph. Results are stored in `self.distances[(loc1, loc2)]`, using `float('inf')` for unreachable pairs. This represents the minimum number of steps (moves or pushes) between locations ignoring dynamic obstacles.

    # Step-By-Step Thinking for Computing Heuristic
    1. Precomputation (`__init__`):
       - Store goal locations: Iterate through `task.goals`, parse `(at box loc)` facts, and populate `self.goal_locations = {box: goal_loc}`.
       - Build adjacency list: Iterate through `task.static`, parse `(adjacent loc1 loc2 dir)` facts, and populate `self.adjacency_list = {loc: [neighbors]}`.
       - Find all locations: Collect all unique location names from `adjacent` facts, `task.initial_state`, and `task.goals` involving location predicates (`at`, `at-robot`, `clear`, `adjacent`). Store in `self.all_locations`.
       - Compute all-pairs shortest paths: For each `loc1` in `self.all_locations`, run BFS using `self.adjacency_list` to find distances to all `loc2`. Store these distances (integers or `float('inf')`) in `self.distances[(loc1, loc2)]`.
    2. Heuristic Calculation (`__call__`):
       - Input: A `node` object containing the current `state`.
       - Goal Check: If `node.task.goal_reached(state)` is true, return 0 immediately.
       - Get Current State Info: Parse `node.state` to find the robot's location (`robot_loc`) from `(at-robot loc)` and the location of each box (`box_locations = {box: loc}`) from `(at box loc)`. Perform basic validation (e.g., ensure robot location is found and valid).
       - Calculate Box-Related Costs:
         - Initialize `total_box_distance = 0`.
         - Initialize `min_robot_to_misplaced_box_dist = float('inf')`.
         - Set flag `misplaced_box_found = False`.
         - Iterate through each `box`, `current_loc` in `box_locations`.
         - Check if `box` has a defined goal location `goal_loc` in `self.goal_locations`.
         - If `current_loc != goal_loc`:
           - Mark `misplaced_box_found = True`.
           - Retrieve precomputed distance `dist_box_to_goal = self.distances.get((current_loc, goal_loc), float('inf'))`.
           - If `dist_box_to_goal` is infinite, the goal is unreachable from this state; return `float('inf')`.
           - Add `dist_box_to_goal` to `total_box_distance`.
           - Retrieve precomputed distance `dist_robot_to_box = self.distances.get((robot_loc, current_loc), float('inf'))`.
           - If `dist_robot_to_box` is finite (robot can reach the box), update `min_robot_to_misplaced_box_dist = min(min_robot_to_misplaced_box_dist, dist_robot_to_box)`.
       - Compute Final Heuristic Value:
         - If `misplaced_box_found` is False (all boxes are at their goals) but the goal state wasn't reached (initial check failed), return 1. This covers cases where only non-box goals remain (e.g., robot position) and assigns a minimal cost.
         - If `min_robot_to_misplaced_box_dist` is still infinity (meaning there are misplaced boxes, but the robot cannot reach any of them), return `float('inf')` as the state is unsolvable.
         - Otherwise, calculate the heuristic value `h = total_box_distance + min_robot_to_misplaced_box_dist`.
         - Return `max(1, int(round(h)))`. This ensures the heuristic is always at least 1 for non-goal states and returns an integer value. `round()` handles potential floating point inaccuracies, though BFS distances should be integers.
    """

    def __init__(self, task):
        """Initializes the heuristic by parsing goals, building the grid graph,
        and precomputing all-pairs shortest path distances."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify Goal Locations for boxes
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Check for (at boxN locM) pattern
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, loc = parts[1], parts[2]
                self.goal_locations[box] = loc

        # 2. Build Grid Graph & Identify All Locations
        self.adjacency_list = {}
        locations_in_static = set() # Use a broader set for all locations found
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            # Check for (adjacent loc1 loc2 dir) pattern
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2 = parts[1], parts[2]
                locations_in_static.add(loc1)
                locations_in_static.add(loc2)
                # Add edge loc1 -> loc2 based on the definition
                self.adjacency_list.setdefault(loc1, []).append(loc2)
            # Also capture locations mentioned in other static predicates if any
            # Example: (is-wall loc1), (is-storage loc1) - not present here but good practice
            # if parts[0] == 'some-static-loc-predicate' and len(parts) == 2:
            #    locations_in_static.add(parts[1])


        # Extract locations mentioned in initial state and goals as well,
        # as they might not be part of an 'adjacent' fact (e.g., isolated goals).
        locations_from_state_goal = set()
        for fact in task.initial_state.union(task.goals):
             parts = get_parts(fact)
             if not parts: continue
             # Add locations from relevant predicates
             if parts[0] == 'at-robot' and len(parts) == 2:
                 locations_from_state_goal.add(parts[1])
             elif parts[0] == 'at' and len(parts) == 3: # (at box loc)
                 locations_from_state_goal.add(parts[2]) # Box location
                 # parts[1] is the box name, not a location
             elif parts[0] == 'clear' and len(parts) == 2:
                 locations_from_state_goal.add(parts[1])
             # Adjacent might appear here too, though usually static
             elif parts[0] == 'adjacent' and len(parts) == 4:
                 locations_from_state_goal.add(parts[1])
                 locations_from_state_goal.add(parts[2])

        # Combine all found locations
        self.all_locations = frozenset(locations_in_static.union(locations_from_state_goal))
        if not self.all_locations:
             # This would be highly unusual for a Sokoban problem
             print("Warning: No locations found in the problem definition.")


        # 3. Precompute All-Pairs Shortest Path Distances using BFS
        self.distances = {}
        for loc1 in self.all_locations:
            # Run BFS starting from loc1 to find distances to all other locations
            distances_from_loc1 = self._bfs(loc1, self.adjacency_list, self.all_locations)
            for loc2 in self.all_locations:
                # Store the computed distance (can be int or inf)
                self.distances[(loc1, loc2)] = distances_from_loc1[loc2]

    def _bfs(self, start_loc, adjacency_list, all_locations):
        """Performs BFS to find shortest path distances from start_loc."""
        # Initialize distances to infinity
        distances = {loc: float('inf') for loc in all_locations}

        # If start location is not valid (e.g., not in graph), return infinities
        if start_loc not in all_locations:
             # This case should ideally not happen if start_loc comes from all_locations
             return distances

        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Explore neighbors using the adjacency list
            for neighbor in adjacency_list.get(current_loc, []):
                # If neighbor is a valid location and not visited yet (dist is inf)
                if neighbor in distances and distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Calculates the heuristic value for the given state node."""
        state = node.state

        # Goal check: If the current state is a goal state, heuristic value is 0.
        if node.task.goal_reached(state):
             return 0

        # 4. Get Current State Information (Robot and Box Locations)
        robot_loc = None
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box_locations[parts[1]] = parts[2] # Store {box_name: location}

        # Basic validity check: Robot must exist and be at a known location.
        if robot_loc is None or robot_loc not in self.all_locations:
             # Cannot compute heuristic if robot position is unknown or invalid.
             # Return infinity as it indicates an issue or unreachable state for this logic.
             return float('inf')

        # 5. Calculate Combined Distance Metric
        total_box_distance = 0
        min_robot_to_misplaced_box_dist = float('inf')
        misplaced_box_found = False

        for box, current_loc in box_locations.items():
            # Check if the box's current location is valid within our known locations
            if current_loc not in self.all_locations:
                # Box is in an unknown/invalid location, consider state unsolvable
                return float('inf')

            # Check if this box has a defined goal location
            if box in self.goal_locations:
                goal_loc = self.goal_locations[box]

                # If the box is not at its goal location
                if current_loc != goal_loc:
                    misplaced_box_found = True

                    # Retrieve precomputed distance: box -> goal
                    # Use .get for safety, though keys should exist from precomputation
                    dist_box_to_goal = self.distances.get((current_loc, goal_loc), float('inf'))

                    # If the goal is unreachable for this box, the state is unsolvable
                    if dist_box_to_goal == float('inf'):
                        return float('inf')

                    total_box_distance += dist_box_to_goal

                    # Retrieve precomputed distance: robot -> box's current location
                    dist_robot_to_box = self.distances.get((robot_loc, current_loc), float('inf'))

                    # Update the minimum distance from robot to *any* misplaced box,
                    # but only consider boxes the robot can actually reach.
                    if dist_robot_to_box != float('inf'):
                        min_robot_to_misplaced_box_dist = min(min_robot_to_misplaced_box_dist, dist_robot_to_box)

        # 6. Compute Final Heuristic Value

        # Case 1: All boxes are at their goal locations, but the goal state is not reached.
        # This implies other goal conditions exist (e.g., robot position). Assign minimal cost.
        if not misplaced_box_found:
             # We already checked goal_reached at the start, so if we are here, goal is not reached.
             return 1

        # Case 2: There are misplaced boxes, but the robot cannot reach any of them.
        # This state is considered unsolvable by this heuristic's logic.
        if min_robot_to_misplaced_box_dist == float('inf'):
             return float('inf')

        # Case 3: Normal case - misplaced boxes exist and robot can reach at least one.
        # Combine the sum of box-to-goal distances and the min robot-to-closest-misplaced-box distance.
        h_value = total_box_distance + min_robot_to_misplaced_box_dist

        # The heuristic value should be an integer or infinity.
        # Ensure it's at least 1 for any non-goal state.
        if h_value == float('inf'):
            # This case should technically be caught earlier if distances are inf,
            # but included for robustness.
            return float('inf')
        else:
            # Cast to int (distances are ints) and ensure it's at least 1.
            # round() might be needed if intermediate calculations introduced floats, but shouldn't here.
            final_h = int(round(h_value))
            return max(1, final_h) # Ensure h >= 1 for non-goal states.
