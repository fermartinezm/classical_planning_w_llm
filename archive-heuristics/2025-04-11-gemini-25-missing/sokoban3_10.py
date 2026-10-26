import sys
import os
from fnmatch import fnmatch
from collections import deque

# Attempt to import the Heuristic base class.
# This assumes the script is run in an environment where 'heuristics.heuristic_base' is accessible.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Provide a dummy base class if the import fails, e.g., for standalone testing.
    print("Warning: Could not import Heuristic from heuristics.heuristic_base. Using a dummy base class.", file=sys.stderr)
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError


def get_parts(fact):
    """
    Utility function to parse a PDDL fact string into its components.
    Removes parentheses and splits the string by spaces.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]

    Args:
        fact: A string representing a PDDL fact.

    Returns:
        A list of strings, where the first element is the predicate name
        and subsequent elements are the arguments. Returns an empty list
        if the fact format is invalid (e.g., not enclosed in parentheses).
    """
    if fact.startswith("(") and fact.endswith(")"):
        return fact[1:-1].split()
    return []


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach a goal state in Sokoban. It is designed
    for use with greedy best-first search and is not necessarily admissible. The
    heuristic value is calculated as the sum of two components:
    1. The sum of shortest path distances for each box from its current location to its
       target goal location, based on the static grid layout.
    2. The shortest path distance for the robot to reach the location of the nearest
       box that is not currently in its goal location.

    # Assumptions
    - The primary cost driver in Sokoban is pushing boxes. The sum of box-to-goal
      distances approximates the total number of pushes required.
    - Robot movement cost is secondary but important for positioning. The distance
      to the nearest misplaced box estimates the initial robot travel needed.
    - Shortest path distances (`self.distances`) are precomputed based on the static
      level layout defined by 'adjacent' predicates. This calculation ignores dynamic
      obstacles like other boxes or the robot's position at evaluation time.
    - Each box specified in the goal conditions (`task.goals`) has a unique target location.
    - The level's connectivity is fully defined by the 'adjacent' predicates in the
      static facts (`task.static`). Locations not connected via 'adjacent' facts are
      considered unreachable from each other.

    # Heuristic Initialization
    - Parses the goal conditions (`task.goals`) to identify the target location for each box,
      stored in `self.goal_locations`.
    - Builds an adjacency list (`self.adj`) representing the static connectivity of
      locations based on 'adjacent' predicates found in `task.static`. Also compiles a
      set of all known locations (`self.locations`).
    - Precomputes all-pairs shortest path distances between all known locations using
      Breadth-First Search (BFS). Results are stored in `self.distances[loc1][loc2]`.
    - Identifies all unique box objects (`self.boxes`) present in the problem by examining
      both the goal conditions and the initial state's 'at' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Goal Check:** If the input `node.state` already satisfies all goal conditions
        (`self.goals`), the estimated cost is 0, and the function returns 0.0.
    2.  **State Parsing:** Extract the current location of the robot (`robot_loc`) from
        `(at-robot ...)` facts and the current location of each object (`box_locations`)
        from `(at ...)` facts within the `node.state`.
    3.  **Validation:** Check if the `robot_loc` was found and corresponds to a known,
        valid location from the static analysis. If not, return `float('inf')` as the
        state is invalid or the heuristic cannot be computed.
    4.  **Initialization:** Initialize the total heuristic value `h = 0.0`. Create a
        dictionary `misplaced_boxes_locs` to store the locations of boxes that are
        not currently at their goal positions.
    5.  **Box Cost Component:**
        a. Iterate through all known boxes (`self.boxes`).
        b. For each box `b` that has a defined goal location `goal_loc` in
           `self.goal_locations`:
           i. Get the box's current location `current_loc` from `box_locations`.
           ii. Perform validation checks: Ensure `current_loc` and `goal_loc` exist
              and are valid known locations. Return `float('inf')` if any check fails.
           iii. If `current_loc` is different from `goal_loc`:
               - Retrieve the precomputed shortest path distance:
                 `dist = self.distances[current_loc][goal_loc]`.
               - If `dist` is `float('inf')`, the goal is statically unreachable for
                 this box. Return `float('inf')` for the entire state.
               - Add `dist` to the total heuristic value `h`.
               - Add the box and its location to `misplaced_boxes_locs`.
    6.  **Robot Cost Component:**
        a. If `misplaced_boxes_locs` is not empty (i.e., at least one box needs moving):
           i. Find the minimum precomputed shortest path distance (`min_robot_dist`)
              from the `robot_loc` to the location (`loc`) of any box currently listed
              in `misplaced_boxes_locs`.
           ii. If `min_robot_dist` is `float('inf')`, the robot cannot statically reach
              any of the misplaced boxes. Return `float('inf')`.
           iii. Add `min_robot_dist` to the total heuristic value `h`.
    7.  **Final Value:**
        a. Check for an edge case: If `h` is calculated as 0.0 but the state is not
           a goal state, return 1.0. This ensures the search makes progress even if
           the heuristic incorrectly estimates zero cost (e.g., if goals involve
           non-box predicates).
        b. Otherwise, return the calculated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information and precomputing distances.

        Args:
            task: The planning task object, containing initial state, goals, operators,
                  and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Build adjacency list graph and identify all locations from static 'adjacent' facts
        self.adj = {}
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            # Format: (adjacent ?l1 - location ?l2 - location ?d - direction)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                if l1 not in self.adj: self.adj[l1] = []
                # Add directed edge l1 -> l2 based on the predicate.
                # Assumes the PDDL provides predicates for both directions if movement is bidirectional.
                self.adj[l1].append(l2)

        # 2. Precompute all-pairs shortest paths using BFS for each location as a start node
        self.distances = {}
        for start_node in self.locations:
            # Initialize distances from start_node to infinity for all locations
            self.distances[start_node] = {loc: float('inf') for loc in self.locations}

            # Distance from start_node to itself is 0
            if start_node in self.locations:
                self.distances[start_node][start_node] = 0
                queue = deque([start_node])
                visited = {start_node} # Track visited locations for this BFS run

                while queue:
                    current_loc = queue.popleft()
                    current_dist = self.distances[start_node][current_loc]

                    # Explore neighbors if the current location has outgoing edges
                    if current_loc in self.adj:
                        for neighbor in self.adj[current_loc]:
                            # Check if neighbor is a valid location and not visited yet in this BFS
                            if neighbor in self.locations and neighbor not in visited:
                                visited.add(neighbor)
                                self.distances[start_node][neighbor] = current_dist + 1
                                queue.append(neighbor)

        # 3. Parse goals to find target locations for boxes
        self.goal_locations = {} # Maps box name -> goal location name
        self.boxes = set()       # Stores all unique box names encountered
        for goal in self.goals:
            parts = get_parts(goal)
            # Expecting goal format: (at <box_name> <location_name>)
            if parts and parts[0] == 'at' and len(parts) == 3:
                 box, loc = parts[1], parts[2]
                 # Assume the second argument in a goal 'at' predicate refers to a box.
                 # A more robust implementation might verify object types if available.
                 self.goal_locations[box] = loc
                 self.boxes.add(box)

        # 4. Identify all boxes from the initial state as well, to ensure all are tracked
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Check for (at <obj_name> <location_name>) facts
             if parts and parts[0] == 'at' and len(parts) == 3:
                 potential_box = parts[1]
                 # Add object to self.boxes if it's already known from goals
                 # or if its name suggests it's a box (e.g., starts with 'box').
                 # This is a heuristic assumption based on common naming conventions.
                 if potential_box in self.goal_locations or potential_box.startswith('box'):
                     self.boxes.add(potential_box)


    def __call__(self, node):
        """
        Calculates the heuristic value for the given search node's state.

        Args:
            node: A search node object containing the state (`node.state`)
                  for which to compute the heuristic value.

        Returns:
            A float representing the estimated cost to reach the goal from the state.
            Returns `float('inf')` if the state is determined to be statically
            unsolvable (e.g., a box or robot cannot reach required locations).
        """
        state = node.state

        # 1. Goal Check: If the current state satisfies all goals, heuristic value is 0.
        if self.goals <= state:
            return 0.0

        # 2. State Parsing: Find current locations of the robot and all objects.
        robot_loc = None
        current_obj_locations = {} # Maps object name -> current location name

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                # Store location for any object found with 'at' predicate.
                obj, loc = parts[1], parts[2]
                current_obj_locations[obj] = loc

        # 3. Validation: Check if robot location is valid.
        if robot_loc is None:
             # print("Warning: Robot location ('at-robot') not found in state.", file=sys.stderr)
             return float('inf') # Cannot compute heuristic without robot position.
        if robot_loc not in self.locations:
             # print(f"Warning: Robot location '{robot_loc}' is not a known location.", file=sys.stderr)
             return float('inf') # Robot is in an invalid or unreachable place.

        # 4. Initialization
        heuristic_value = 0.0
        misplaced_boxes_locs = {} # Stores {box_name: current_location} for boxes not at goal

        # 5. Box Cost Component: Calculate sum of distances for misplaced boxes.
        for box in self.boxes:
            # Only consider boxes that have a defined goal location.
            if box in self.goal_locations:
                goal_loc = self.goal_locations[box]
                current_loc = current_obj_locations.get(box)

                # Validate box's current location.
                if current_loc is None:
                    # print(f"Warning: Location for box '{box}' not found in current state.", file=sys.stderr)
                    return float('inf') # State seems inconsistent.
                if current_loc not in self.locations:
                    # print(f"Warning: Box '{box}' current location '{current_loc}' is not a known location.", file=sys.stderr)
                    return float('inf') # Box is in an invalid location.

                # Validate box's goal location.
                if goal_loc not in self.locations:
                     # print(f"Warning: Box '{box}' goal location '{goal_loc}' is not a known location.", file=sys.stderr)
                     return float('inf') # Goal location is invalid.

                # If the box is not at its goal location:
                if current_loc != goal_loc:
                    # Retrieve the precomputed shortest path distance.
                    dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))

                    # Check if the goal is statically reachable for this box.
                    if dist == float('inf'):
                        # print(f"Info: Box {box} at {current_loc} cannot reach goal {goal_loc} statically.", file=sys.stderr)
                        return float('inf') # State is statically unsolvable.

                    # Add the distance to the heuristic value.
                    heuristic_value += dist
                    # Record the box as misplaced.
                    misplaced_boxes_locs[box] = current_loc

        # 6. Robot Cost Component: Add distance from robot to the nearest misplaced box.
        if misplaced_boxes_locs: # Proceed only if there are boxes to move.
            min_robot_dist = float('inf')
            # Find the minimum distance from the robot to any misplaced box's location.
            for box, loc in misplaced_boxes_locs.items():
                 dist_to_box = self.distances.get(robot_loc, {}).get(loc, float('inf'))
                 min_robot_dist = min(min_robot_dist, dist_to_box)

            # Check if the robot can reach any misplaced box.
            if min_robot_dist == float('inf'):
                 # print(f"Info: Robot at {robot_loc} cannot reach any misplaced box statically.", file=sys.stderr)
                 return float('inf') # State is statically unsolvable from robot's perspective.

            # Add the minimum robot travel distance to the heuristic value.
            heuristic_value += min_robot_dist

        # 7. Final Value Adjustment: Ensure non-zero for non-goal states.
        # If the heuristic calculated to 0 but the state is not a goal state,
        # return 1.0 to prevent the search from terminating prematurely.
        if heuristic_value == 0.0 and not (self.goals <= state):
             return 1.0

        return heuristic_value

