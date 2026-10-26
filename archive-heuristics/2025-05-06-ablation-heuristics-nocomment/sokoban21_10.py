from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Removes parentheses and splits the fact string into parts."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """Checks if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing the
    minimum number of pushes required for each box not at its goal location,
    plus the minimum number of robot moves required to reach a position
    from which it can initiate the first push on any box that needs moving.
    The minimum number of pushes for a box is estimated by the shortest path
    distance on the grid graph from the box's current location to its goal
    location. The robot movement cost is the shortest path distance from the
    robot's current location to the required push location for the closest
    box (in terms of robot distance to push location) that needs moving.
    This heuristic is non-admissible as it ignores obstacles (other boxes,
    walls not represented by lack of adjacency) and the robot's movement
    cost between subsequent pushes or between different boxes.

    Assumptions:
    - The problem graph is defined by 'adjacent' predicates.
    - Locations are reachable from each other if a path exists via 'adjacent' predicates.
    - The goal state specifies the target location for each box using '(at boxX locY)' facts.
    - All boxes specified in the goal are present in the initial state and subsequent states.

    Heuristic Initialization:
    In the constructor (__init__), the heuristic precomputes the following from static facts:
    1. A graph representation of the locations and their adjacencies, including directions.
       - `adj_map`: Maps each location to a dictionary of directions and the adjacent location in that direction (e.g., `{'loc_1_1': {'right': 'loc_1_2', 'down': 'loc_2_1'}}`).
       - `rev_adj_map`: Maps each location to a dictionary of directions and the location from which one arrives at the key location by moving in that direction (e.g., `{'loc_1_2': {'left': 'loc_1_1'}, 'loc_2_1': {'up': 'loc_1_1'}}`).
    2. A mapping of opposite directions (e.g., 'up' is opposite of 'down').
    3. A list of all unique locations in the problem.
    4. All-pairs shortest path distances between all locations using BFS.
       - `dist_map`: `dist_map[l1][l2]` stores the shortest path distance from `l1` to `l2`.
    5. The first step location on a shortest path from any location `l1` to any other location `l2`.
       - `first_step_map`: `first_step_map[l1][l2]` stores the location adjacent to `l1` that is the first step on a shortest path from `l1` to `l2`.
    6. The goal location for each box from the task's goal state.

    Step-By-Step Thinking for Computing Heuristic:
    In the __call__ method for a given state:
    1. Identify the current location of the robot.
    2. Identify the current location of each box that is part of the goal.
    3. Initialize `total_box_distance` to 0 and `min_robot_dist_to_push` to infinity.
    4. Initialize a flag `boxes_to_move` to False.
    5. Iterate through each box and its corresponding goal location (from the precomputed goal map):
        a. Get the box's current location from the state. If a goal box is not found in the state, return infinity (unreachable).
        b. If the box is not at its goal location:
            i. Set `boxes_to_move` to True.
            ii. Look up the shortest path distance from the box's current location to its goal location using the precomputed `dist_map`. If unreachable, return infinity. Add this distance to `total_box_distance`. This represents the minimum number of pushes required for this box.
            iii. If the box distance is greater than 0:
                - Find the first step location (`l1`) on a shortest path from the box's current location to its goal location using the precomputed `first_step_map`. If no such step exists (should only happen if distance is 0 or infinity), return infinity.
                - Determine the direction from the box's current location to `l1` using `adj_map`.
                - Find the opposite direction using the precomputed opposite directions map.
                - Find the location (`loc_behind_box`) adjacent to the box's current location in the opposite direction using `rev_adj_map`. This is the location the robot needs to be at to push the box towards `l1`. If no such location exists (e.g., box is against a wall and cannot be pushed in that direction), return infinity.
                - Look up the shortest path distance from the robot's current location to `loc_behind_box` using `dist_map`. If unreachable, return infinity.
                - Update `min_robot_dist_to_push` with the minimum of its current value and the distance calculated in the previous step.
    6. If `boxes_to_move` is True, the heuristic value is `total_box_distance + min_robot_dist_to_push`.
    7. If `boxes_to_move` is False (all boxes are at their goal locations), the heuristic value is 0.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        self.adj_map = {}
        self.rev_adj_map = {}
        self.locations = set()
        self.opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        # Build adjacency maps and collect all locations
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)

                if loc1 not in self.adj_map:
                    self.adj_map[loc1] = {}
                self.adj_map[loc1][direction] = loc2

                if loc2 not in self.rev_adj_map:
                    self.rev_adj_map[loc2] = {}
                # rev_adj_map[loc2][direction] stores the location loc1
                # such that moving from loc1 in 'direction' leads to loc2
                self.rev_adj_map[loc2][direction] = loc1

        # Precompute all-pairs shortest paths and first steps using BFS
        self.dist_map = {}
        self.first_step_map = {} # Stores the *next* location from source on a shortest path to target

        for start_loc in self.locations:
            self.dist_map[start_loc] = {}
            self.first_step_map[start_loc] = {}
            queue = deque([start_loc])
            visited = {start_loc}
            self.dist_map[start_loc][start_loc] = 0
            # first_step_map[start_loc][start_loc] is not needed

            while queue:
                u = queue.popleft()

                # Iterate through neighbors using adj_map
                if u in self.adj_map:
                    for direction, v in self.adj_map[u].items():
                        if v not in visited:
                            visited.add(v)
                            self.dist_map[start_loc][v] = self.dist_map[start_loc][u] + 1
                            # The first step from start_loc to v is v itself if u is start_loc
                            # Otherwise, the first step from start_loc to v is the same as the first step from start_loc to u
                            if u == start_loc:
                                self.first_step_map[start_loc][v] = v
                            else:
                                # The first step from start_loc to u was already recorded when u was visited
                                # The path from start_loc to v goes through u, so the first step is the same
                                self.first_step_map[start_loc][v] = self.first_step_map[start_loc][u]
                            queue.append(v)

        # Extract box goal locations
        self.box_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                # Assuming the first argument of (at ...) in goal is always a box
                self.box_goals[parts[1]] = parts[2]

    def get_location_behind(self, current_loc, next_loc):
        """
        Finds the location loc_behind such that moving from loc_behind
        to current_loc is in the opposite direction of moving from
        current_loc to next_loc. This is the required robot position
        to push a box from current_loc to next_loc.
        Returns None if next_loc is not a direct neighbor or if no location
        exists behind current_loc in the required direction (e.g., wall).
        """
        # Find the direction from current_loc to next_loc
        direction_from_curr_to_next = None
        if current_loc in self.adj_map:
            for direction, neighbor in self.adj_map[current_loc].items():
                if neighbor == next_loc:
                    direction_from_curr_to_next = direction
                    break

        if direction_from_curr_to_next is None:
            # next_loc is not a direct neighbor of current_loc.
            # This shouldn't happen if next_loc is the first step on a shortest path
            # unless current_loc == next_loc (distance 0), which is handled elsewhere.
            return None

        # The direction the robot needs to move to get behind the box
        # is the opposite of the direction the box will move.
        direction_robot_to_box = self.opposite_dir.get(direction_from_curr_to_next)

        if direction_robot_to_box is None:
             # Should not happen with standard directions
             return None

        # Find the location adjacent to current_loc such that moving from
        # that location in direction_robot_to_box gets you to current_loc.
        # This is stored in rev_adj_map[current_loc][direction_robot_to_box].
        loc_behind = self.rev_adj_map.get(current_loc, {}).get(direction_robot_to_box)

        return loc_behind


    def __call__(self, node):
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_loc = get_parts(fact)
                break
        # Assuming robot_loc is always found in a valid state

        # Find current box locations for goal boxes
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                # Only consider boxes that are part of the goal
                if parts[1] in self.box_goals:
                     current_box_locations[parts[1]] = parts[2]

        # Check if all goal boxes are present in the current state
        if len(current_box_locations) != len(self.box_goals):
             # Some goal boxes are missing from the state, likely an unreachable state
             return float('inf')

        total_box_distance = 0
        min_robot_dist_to_push = float('inf')
        boxes_to_move = False

        for box, goal_loc in self.box_goals.items():
            current_loc = current_box_locations.get(box)

            # This check should ideally not be needed if len(current_box_locations) == len(self.box_goals)
            # but kept for safety.
            if current_loc is None:
                 return float('inf') # Goal box not found in state

            if current_loc != goal_loc:
                boxes_to_move = True

                # Distance for the box to reach its goal
                if current_loc not in self.dist_map or goal_loc not in self.dist_map[current_loc]:
                    # Box goal is unreachable from current location on the grid graph
                    return float('inf')

                box_dist = self.dist_map[current_loc][goal_loc]
                total_box_distance += box_dist

                # If box_dist is 0, the box is already at the goal, no push needed for this box
                if box_dist > 0:
                    # Find the first step location on a shortest path for the box
                    # first_step_map[current_loc][goal_loc] gives the neighbor of current_loc
                    # that is on a shortest path to goal_loc.
                    if goal_loc not in self.first_step_map.get(current_loc, {}):
                         # Should not happen if box_dist > 0 and reachable
                         return float('inf') # Error in precomputation or logic

                    first_step_loc = self.first_step_map[current_loc][goal_loc]

                    # Find the location the robot needs to be at to push the box from current_loc to first_step_loc
                    loc_behind_box = self.get_location_behind(current_loc, first_step_loc)

                    if loc_behind_box is None:
                         # Cannot find a location behind the box to push it towards the goal
                         # This box might be stuck against a wall/obstacle in a way that prevents reaching the goal
                         return float('inf')

                    # Distance for the robot to reach the push position
                    if robot_loc not in self.dist_map or loc_behind_box not in self.dist_map[robot_loc]:
                         # Robot cannot reach the required push position
                         return float('inf')

                    robot_dist = self.dist_map[robot_loc][loc_behind_box]
                    min_robot_dist_to_push = min(min_robot_dist_to_push, robot_dist)


        if boxes_to_move:
            # If min_robot_dist_to_push is still infinity, it means boxes_to_move was true
            # but we couldn't find a reachable push position for any box that needs moving.
            # This indicates an unsolvable state or a state from which no progress can be made.
            if min_robot_dist_to_push == float('inf'):
                 return float('inf')
            return total_box_distance + min_robot_dist_to_push
        else:
            # All boxes are at their goal locations
            return 0
