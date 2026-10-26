import heapq
import logging
import re

# Assume Heuristic and Task classes are available from the planner environment
# from heuristics.heuristic_base import Heuristic
# from task import Operator, Task

# Define the heuristic class
class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing two components:
    1. The sum of the shortest path distances for each box from its current location
       to its assigned goal location.
    2. The minimum shortest path distance for the robot from its current location
       to a position where it can make the first push for any box that is not
       yet at its goal.

    Assumptions:
    - The domain uses 'location', 'direction', 'box' types.
    - The grid structure is defined by 'adjacent' facts in the static information.
    - Goal states are defined by '(at ?b ?l)' facts for specific boxes at specific locations.
    - Each box has a unique goal location specified in the task goals.
    - Locations are named such that they can be treated as nodes in a graph.
    - The 'adjacent' predicate (adjacent l1 l2 dir) means moving from l1 to l2 is in direction dir.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes static information from the task:
    - It parses 'adjacent' facts to build a graph representing the grid connectivity,
      mapping each location and direction to the neighboring location.
    - It parses the goal facts to determine the target location for each box.
    - It computes all-pairs shortest path distances between all locations on the grid
      using Breadth-First Search (BFS).
    - It also stores the direction of the first step on a shortest path from any
      location to any other location.
    - It defines a mapping for opposite directions.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Check if the state is a goal state (all boxes are at their goal locations). If yes, return 0.
    2. Identify the current location of the robot and each box from the state facts.
    3. Initialize the total heuristic value `h` to 0.
    4. Initialize the minimum robot distance to a required push position `min_robot_dist_to_push_pos` to infinity.
    5. Iterate through each box that is not yet at its goal location:
        a. Get the box's current location and its assigned goal location.
        b. Calculate the shortest path distance between the box's current location and its goal location using the precomputed distances. Add this distance to `h`. If the goal is unreachable for this box, the state is likely unsolvable, return infinity.
        c. Determine the direction `dir` of the first step on a shortest path from the box's current location to its goal location using the precomputed first step directions.
        d. Determine the required robot position `rloc` to push the box from `box_loc` to `next_loc` in direction `dir`. Based on the PDDL 'push' action definition, the robot must be at `rloc` such that `adjacent(rloc, box_loc, dir)` is true. Given the interpretation that `adjacent(l1, l2, dir)` meaning `l2` is adjacent to `l1` in direction `dir`, the condition `adjacent(rloc, box_loc, dir)` means `box_loc` is adjacent to `rloc` in direction `dir`. This implies `rloc` is the location adjacent to `box_loc` in the *opposite* direction of `dir`. Find this required robot location using the precomputed graph and the opposite direction mapping.
        e. Calculate the shortest path distance from the robot's current location to this required robot push position using the precomputed distances.
        f. Update `min_robot_dist_to_push_pos` with the minimum distance found so far across all boxes not at their goals.
    6. After iterating through all boxes not at their goals, add `min_robot_dist_to_push_pos` to `h`.
    7. Return the final value of `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # Data structures to store grid information and goal mapping
        self.location_names = []
        self.graph_by_direction = {} # loc -> {direction: neighbor_loc}
        self.box_goals = {} # box_name -> goal_location_name
        self.distances = {} # loc1 -> {loc2: distance}
        self.first_step_direction = {} # loc1 -> {loc2: direction_of_first_step}
        self.opposite_direction = {
            'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'
        }

        # 1. Parse static facts to build the graph
        self._parse_static()

        # 2. Parse goal facts to get box-goal mapping
        self._parse_goals()

        # 3. Compute all-pairs shortest paths and first step directions
        self._compute_all_pairs_shortest_paths()

    def _parse_fact(self, fact_string):
        """Helper to parse a PDDL fact string into a tuple."""
        # Use regex to find content within parentheses and split by space
        match = re.match(r'\((.*?)\)', fact_string)
        if match:
            content = match.group(1)
            # Split by one or more spaces
            parts = re.split(r'\s+', content.strip())
            return tuple(parts)
        return None # Should not happen for valid facts

    def _parse_static(self):
        """Parses static facts to build the grid graph."""
        all_locations = set()
        for fact_string in self.static:
            fact = self._parse_fact(fact_string)
            if fact and fact[0] == 'adjacent' and len(fact) == 4:
                _, loc1, loc2, direction = fact
                all_locations.add(loc1)
                all_locations.add(loc2)
                if loc1 not in self.graph_by_direction:
                    self.graph_by_direction[loc1] = {}
                self.graph_by_direction[loc1][direction] = loc2

        self.location_names = list(all_locations)
        # Ensure all locations are keys in the graph, even if they have no adjacencies listed (shouldn't happen in valid sokoban)
        for loc in self.location_names:
             if loc not in self.graph_by_direction:
                 self.graph_by_direction[loc] = {}


    def _parse_goals(self):
        """Parses goal facts to get the box-goal mapping."""
        # Goals are a frozenset of fact strings
        for goal_string in self.goals:
            goal = self._parse_fact(goal_string)
            if goal and goal[0] == 'at' and len(goal) == 3:
                _, box_name, goal_location = goal
                self.box_goals[box_name] = goal_location

    def _bfs(self, start_loc):
        """
        Performs BFS from a start location to find distances and first step
        directions to all reachable locations.
        Returns (distances_from_start, first_step_directions_from_start).
        """
        distances_from_start = {loc: float('inf') for loc in self.location_names}
        first_step_directions_from_start = {loc: None for loc in self.location_names}
        queue = [(start_loc, 0, None)] # (current_loc, distance, first_step_dir)
        distances_from_start[start_loc] = 0
        visited = {start_loc}

        q_index = 0
        while q_index < len(queue):
            u, dist_u, first_dir_u = queue[q_index]
            q_index += 1

            if u in self.graph_by_direction:
                for direction, v in self.graph_by_direction[u].items():
                    if v not in visited:
                        visited.add(v)
                        distances_from_start[v] = dist_u + 1
                        # The first step direction from start_loc to v is:
                        # - 'direction' if u is the start_loc
                        # - the same first step direction as from start_loc to u, if u is not start_loc
                        if u == start_loc:
                            first_step_directions_from_start[v] = direction
                        else:
                            first_step_directions_from_start[v] = first_dir_u
                        queue.append((v, dist_u + 1, first_step_directions_from_start[v]))

        return distances_from_start, first_step_directions_from_start

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest paths and first step directions for all pairs."""
        for start_loc in self.location_names:
            distances, first_steps = self._bfs(start_loc)
            self.distances[start_loc] = distances
            self.first_step_direction[start_loc] = first_steps

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        # Check if goal is reached
        if self.task.goal_reached(state):
             return 0

        robot_loc = None
        box_locations = {} # box_name -> location_name
        current_state_facts = set(state) # Convert frozenset to set for faster lookup

        # Find robot and box locations
        for fact_string in current_state_facts:
            fact = self._parse_fact(fact_string)
            if fact and fact[0] == 'at-robot' and len(fact) == 2:
                robot_loc = fact[1]
            elif fact and fact[0] == 'at' and len(fact) == 3 and fact[1] in self.box_goals:
                box_name, loc = fact[1], fact[2]
                box_locations[box_name] = loc

        if robot_loc is None:
             # Should not happen in a valid Sokoban state, but handle defensively
             logging.warning("Robot location not found in state.")
             return float('inf')

        h_box_distances = 0
        min_robot_dist_to_push_pos = float('inf')
        found_box_not_at_goal = False

        # Calculate heuristic components for boxes not at goals
        for box_name, goal_loc in self.box_goals.items():
            if box_name not in box_locations:
                 # Should not happen in a valid Sokoban state
                 logging.warning(f"Box {box_name} not found in state.")
                 return float('inf')

            box_loc = box_locations[box_name]

            # If box is already at goal, it contributes 0 to box distance sum
            if box_loc == goal_loc:
                continue

            found_box_not_at_goal = True

            # Component 1: Box distance to goal
            if box_loc not in self.distances or goal_loc not in self.distances[box_loc]:
                 # Should not happen if _compute_all_pairs_shortest_paths covered all locations
                 logging.error(f"Distance not precomputed for {box_loc} to {goal_loc}")
                 return float('inf')

            box_dist = self.distances[box_loc][goal_loc]

            if box_dist == float('inf'):
                # Box cannot reach its goal
                return float('inf')

            h_box_distances += box_dist

            # Component 2: Robot distance to required push position for this box
            # Find the direction 'dir' of the first step on a shortest path for the box
            if box_loc not in self.first_step_direction or goal_loc not in self.first_step_direction[box_loc]:
                 # Should not happen if _compute_all_pairs_shortest_paths covered all locations
                 logging.error(f"First step direction not precomputed for {box_loc} to {goal_loc}")
                 return float('inf')

            first_step_dir = self.first_step_direction[box_loc][goal_loc]

            # Determine the required robot position 'rloc' to push the box from 'box_loc'
            # in direction 'first_step_dir'.
            # Based on PDDL: (adjacent ?rloc ?bloc ?dir) means ?bloc is adjacent to ?rloc in ?dir.
            # So, ?rloc is adjacent to ?bloc in the opposite direction of ?dir.
            opposite_dir = self.opposite_direction.get(first_step_dir)
            if opposite_dir is None:
                 logging.error(f"Unknown direction {first_step_dir}")
                 return float('inf')

            # Find the location adjacent to box_loc in the opposite direction
            required_r_pos = None
            if box_loc in self.graph_by_direction and opposite_dir in self.graph_by_direction[box_loc]:
                 required_r_pos = self.graph_by_direction[box_loc][opposite_dir]
            else:
                 # This means the box cannot be pushed from this side in this direction.
                 # This location might be a wall or edge in that direction.
                 # If the shortest path requires pushing in this direction, but the required robot position doesn't exist,
                 # it implies a potential deadlock or an invalid path step.
                 # Returning infinity seems appropriate.
                 logging.debug(f"Required push position not found for box at {box_loc} in opposite direction {opposite_dir} (original dir {first_step_dir})")
                 return float('inf')

            if required_r_pos is None:
                 # This case should be covered by the else above, but defensive check
                 logging.error(f"Could not determine required robot position for box at {box_loc} moving in {first_step_dir}")
                 return float('inf')

            # Calculate robot distance to this required position
            if robot_loc not in self.distances or required_r_pos not in self.distances[robot_loc]:
                 logging.error(f"Distance not precomputed for robot at {robot_loc} to required pos {required_r_pos}")
                 return float('inf')

            robot_dist_to_push_pos = self.distances[robot_loc][required_r_pos]

            # Update minimum robot distance
            min_robot_dist_to_push_pos = min(min_robot_dist_to_push_pos, robot_dist_to_push_pos)

        # If all boxes are at goals, found_box_not_at_goal is False, and goal_reached check returned 0.
        # If there are boxes not at goals, add the minimum robot distance component.
        if found_box_not_at_goal:
             h = h_box_distances + min_robot_dist_to_push_pos
        else:
             # This case should be caught by the initial goal_reached check,
             # but as a fallback, if no boxes are away from goals, heuristic is 0.
             h = 0 # Should already be 0 from h_box_distances

        return h
