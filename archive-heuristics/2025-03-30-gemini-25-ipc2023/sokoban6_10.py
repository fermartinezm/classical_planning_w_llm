import sys
from collections import deque

# Try to import the base class Heuristic from the expected location.
# If it's not found, define a dummy class to allow the code structure to work.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Heuristic base class not found. Defining a dummy base class.", file=sys.stderr)
    class Heuristic:
        """Dummy base class for Heuristic."""
        def __init__(self, task):
            """Initializes the dummy heuristic."""
            pass
        def __call__(self, node):
            """Placeholder call method."""
            raise NotImplementedError("Heuristic base class was not found.")

def get_parts(fact):
    """
    Extracts the predicate and arguments from a PDDL fact string.
    Removes the surrounding parentheses and splits the string by spaces.

    Args:
        fact (str): A PDDL fact string, e.g., "(at box1 loc_1_1)".

    Returns:
        list[str]: A list containing the predicate name and its arguments,
                   e.g., ["at", "box1", "loc_1_1"].
                   Returns an empty list if the fact is malformed (e.g., not enclosed
                   in parentheses or empty).
    """
    if len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Handle potential malformed facts gracefully
        print(f"Warning: Malformed fact string encountered: {fact}", file=sys.stderr)
        return []
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in a Sokoban problem.
    It calculates the sum of shortest path distances for each box from its current
    location to its designated goal location. Additionally, it adds the shortest
    path distance from the robot's current location to the location of the nearest
    misplaced box. The distances are precomputed based on the static layout of the
    level (ignoring dynamic obstacles like other boxes) using Breadth-First Search.
    This heuristic is designed for use with Greedy Best-First Search and is likely
    non-admissible, aiming for informativeness over admissibility.

    # Assumptions
    - The cost of both 'move' and 'push' actions is assumed to be 1.
    - The heuristic estimates the total number of pushes required (approximated by the
      sum of box-to-goal distances) plus the number of moves needed for the robot
      to reach the nearest box that needs pushing.
    - It does not model or detect complex interactions or deadlocks (e.g., boxes
      in corners, against walls without goals, or blocking each other), beyond
      basic reachability determined by the static map connectivity.
    - The 'adjacent' predicates in the static facts accurately and completely define
      the traversable map connectivity for the robot (when ignoring boxes).
    - All locations relevant to the problem (initial positions, goal positions,
      and intermediate path locations) are defined within the set of locations
      derived from static facts and initial/goal states.

    # Heuristic Initialization (`__init__`)
    - Parses the task's goal conditions (`task.goals`) to identify the target location
      for each box specified in the goal (`self.goal_locations`). It also stores the
      set of boxes involved in the goal (`self.goal_boxes`).
    - Parses the static facts (`task.static`) to build a graph representation (adjacency list,
      `self.adj`) of the level layout based on 'adjacent' predicates. It identifies the
      set of all unique locations (`self.locations`) appearing in static facts, initial
      state, and goals.
    - Precomputes all-pairs shortest path distances between all identified locations using
      Breadth-First Search (BFS) starting from each location. These distances represent the
      minimum number of 'move' actions for the robot between locations on an empty map.
      The results are stored in `self.distances[start_loc][end_loc]`. If a location is
      unreachable from another, the distance is stored as `float('inf')`.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Parse State:** Extract the robot's current location (`robot_loc`) and the current
        location of each goal box (`box_locations`) from the input state (`node.state`).
        Perform sanity checks to ensure the robot and all goal boxes are found and their
        locations are known within the precomputed map. If not, return infinity.
    2.  **Check Goal:** Determine if the current state satisfies all goal conditions (i.e.,
        every goal box is at its target location). If yes, the heuristic value is 0.
    3.  **Calculate Box Component:**
        - Initialize `box_heuristic = 0`.
        - Identify the set of `misplaced_boxes` (those not at their goal location).
        - For each `box` in `misplaced_boxes`:
            - Get its current location `current_loc` and target goal location `goal_loc`.
            - Retrieve the precomputed shortest path distance `d = self.distances[current_loc][goal_loc]`.
            - If `d` is infinity, the goal is unreachable for this box from its current spot
              based on map connectivity; return `float('inf')` for the state.
            - Add `d` to `box_heuristic`. This sum estimates the total pushes needed.
    4.  **Calculate Robot Component:**
        - Initialize `robot_heuristic = float('inf')`.
        - For each `box` in `misplaced_boxes`:
            - Get the box's current location `box_loc`.
            - Retrieve the precomputed distance from the robot's location to the box's location:
              `d_robot_box = self.distances[robot_loc][box_loc]`.
            - Update `robot_heuristic = min(robot_heuristic, d_robot_box)`.
        - This `robot_heuristic` estimates the moves needed for the robot to reach the *closest*
          box that requires pushing.
        - If `robot_heuristic` remains infinity, the robot cannot reach any misplaced box; return `float('inf')`.
    5.  **Combine:** The final heuristic value is `box_heuristic + robot_heuristic`. This value
        combines the estimated cost related to moving boxes and the cost for the robot to
        initiate the next push. Return the sum, ensuring it's non-negative.
    """

    def __init__(self, task):
        """
        Initializes the Sokoban heuristic. Parses goals, builds the map graph,
        and precomputes all-pairs shortest path distances between locations.
        """
        super().__init__(task) # Call base class constructor if necessary
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goal locations for boxes mentioned in the goal predicates
        self.goal_locations = {} # Stores {box_name: goal_location}
        self.goal_boxes = set()  # Stores names of boxes that have a goal position
        for goal in self.goals:
            parts = get_parts(goal)
            # Expecting goals like: (at <box> <location>)
            if parts and parts[0] == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                self.goal_locations[box] = loc
                self.goal_boxes.add(box)
            else:
                # Optionally warn about unexpected goal formats
                # print(f"Warning: Ignoring unexpected goal format: {goal}", file=sys.stderr)
                pass

        # 2. Build adjacency graph and identify all unique locations
        self.adj = {} # Adjacency list: {location: [neighbor1, neighbor2, ...]}
        self.locations = set() # Set of all unique location names

        # Populate locations and adjacency list from static 'adjacent' facts
        for fact in static_facts:
            parts = get_parts(fact)
            # Expecting facts like: (adjacent <loc1> <loc2> <direction>)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                # Add directed edge l1 -> l2 based on the adjacency predicate
                self.adj.setdefault(l1, []).append(l2)

        # Ensure locations mentioned in the initial state or goals are included
        # in self.locations, even if they are isolated (not in any 'adjacent' fact).
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue # Skip malformed facts
             # Consider locations from 'at', 'at-robot', 'clear' predicates
             if parts[0] == 'at-robot' and len(parts) == 2:
                 self.locations.add(parts[1])
             elif parts[0] == 'at' and len(parts) == 3:
                 # Location where a box is initially
                 self.locations.add(parts[2])
             elif parts[0] == 'clear' and len(parts) == 2:
                 # Location that is initially clear
                 self.locations.add(parts[1])
        # Add goal locations to the set of known locations
        for loc in self.goal_locations.values():
            self.locations.add(loc)

        # 3. Precompute all-pairs shortest paths using BFS
        self.distances = {} # Stores {start_loc: {end_loc: distance}}
        for start_loc in self.locations:
            # Initialize distances from start_loc to all locations as infinity
            bfs_distances = {node: float('inf') for node in self.locations}

            # Check if start_loc is valid before starting BFS
            if start_loc in bfs_distances:
                bfs_distances[start_loc] = 0
                queue = deque([start_loc])
                while queue:
                    current_node = queue.popleft()
                    # Explore neighbors using the pre-built adjacency list
                    for neighbor in self.adj.get(current_node, []):
                        # Check if neighbor is a known location and hasn't been reached yet
                        # or if a shorter path is found (standard BFS finds shortest first)
                        if neighbor in bfs_distances and bfs_distances[neighbor] == float('inf'):
                            bfs_distances[neighbor] = bfs_distances[current_node] + 1
                            queue.append(neighbor)
            # Store the computed distances from start_loc
            self.distances[start_loc] = bfs_distances


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.

        Args:
            node: A search node containing the current state (`node.state`).

        Returns:
            float: The estimated cost (number of actions) to reach the goal.
                   Returns 0 for a goal state.
                   Returns float('inf') if the goal seems unreachable from this state.
        """
        state = node.state

        # 1. Parse current state: find robot location and locations of goal boxes
        robot_loc = None
        box_locations = {} # Stores {box_name: current_location} for goal boxes

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                # Only track locations of boxes that are part of the goal
                if box in self.goal_boxes:
                     box_locations[box] = loc

        # --- Sanity Checks ---
        if robot_loc is None:
            # print(f"Warning: Robot location not found in state: {state}", file=sys.stderr)
            return float('inf') # Cannot evaluate state without robot position
        if robot_loc not in self.locations:
            # print(f"Warning: Robot at unknown location '{robot_loc}'.", file=sys.stderr)
            return float('inf') # Robot is outside the known map
        if len(box_locations) != len(self.goal_boxes):
            # print(f"Warning: Mismatch in goal boxes found ({len(box_locations)}) vs expected ({len(self.goal_boxes)}).", file=sys.stderr)
            return float('inf') # State seems inconsistent if a goal box is missing
        # --- End Sanity Checks ---

        # 2. Check for goal state & calculate box heuristic component
        box_heuristic = 0
        misplaced_boxes = set()
        is_goal = True

        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations[box]

            # Check if the box's current location is known
            if current_loc not in self.locations:
                 # print(f"Warning: Box '{box}' at unknown location '{current_loc}'.", file=sys.stderr)
                 return float('inf') # Box is outside the known map

            if current_loc != goal_loc:
                is_goal = False
                misplaced_boxes.add(box)

                # Retrieve precomputed distance: current_loc -> goal_loc
                # Use .get() for safe dictionary access, defaulting to infinity
                dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))

                if dist == float('inf'):
                    # If distance is infinite, the goal is statically unreachable for this box
                    return float('inf')
                box_heuristic += dist

        # If all boxes are in their goal locations, the state is a goal state
        if is_goal:
            return 0

        # If we reach here, is_goal is False, and misplaced_boxes is not empty.

        # 3. Calculate robot heuristic component (min distance to any misplaced box)
        robot_heuristic = float('inf')

        # Ensure the robot's location is a valid key in the distances map
        if robot_loc not in self.distances:
             # This should have been caught by sanity checks, but double-check
             return float('inf')

        for box in misplaced_boxes:
            box_loc = box_locations[box]
            # Retrieve precomputed distance: robot_loc -> box_loc
            dist_robot_box = self.distances[robot_loc].get(box_loc, float('inf'))
            robot_heuristic = min(robot_heuristic, dist_robot_box)

        # If robot cannot reach any misplaced box, the state is effectively a dead-end
        if robot_heuristic == float('inf'):
            return float('inf')

        # 4. Return the combined heuristic value
        final_heuristic = box_heuristic + robot_heuristic

        # Final check for potential infinity result from sum (though unlikely)
        if final_heuristic == float('inf'):
             return float('inf')
        else:
             # Heuristic value should be non-negative
             return max(0, final_heuristic)

