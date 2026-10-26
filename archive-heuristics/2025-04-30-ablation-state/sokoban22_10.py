# Required imports
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components for each box not yet at its goal:
    1. The shortest path distance the box needs to travel to reach its goal location, considering other boxes as obstacles.
    2. The shortest path distance the robot needs to travel from its current location to a position from which it can push the box one step closer to its goal, considering other boxes as obstacles.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - Boxes must be pushed to specific goal locations.
    - The robot must be adjacent to a box on the opposite side of the intended push direction to push it.
    - The heuristic assumes a path exists for the box and the robot (finite distance). Unreachable goals will result in a large heuristic value.
    - The heuristic does not explicitly check for dead-end states (e.g., box pushed into a corner it cannot leave) beyond simple reachability with current obstacles.

    # Heuristic Initialization
    - Parse `adjacent` facts from the static information to build a graph representation of the grid connectivity. The graph maps each location to its adjacent locations and the direction of movement.
    - Parse goal facts to create a mapping from each box to its target goal location.
    - Create a mapping from direction names to their opposite direction names.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot. If the robot's location is not found, return infinity.
    2. Identify the current location of each box that has a goal location. Store these in a dictionary `box_locations`.
    3. Initialize the total heuristic value `h` to 0.
    4. For each box `b` that has a goal location `loc_g_b`:
        a. Get the box's current location `loc_b` from `box_locations`. If the box is not found in the state, return infinity.
        b. If `loc_b` is the same as `loc_g_b`, this box is at its goal; continue to the next box.
        c. Identify the locations of all *other* boxes (`other_box_locations`). These locations act as obstacles for both box and robot movement BFS.
        d. Calculate the shortest path distance `dist_box` from `loc_b` to `loc_g_b` on the grid graph, treating `other_box_locations` as obstacles. Use BFS (`self.bfs_distance`). If `loc_g_b` is unreachable, return infinity for the total heuristic.
        e. Add `dist_box` to `h`. This represents the minimum number of pushes needed for this box if the robot were always in position.
        f. Calculate the minimum distance for the robot to get into a position to push the box one step closer to the goal. Initialize `min_robot_dist_to_push` to infinity.
        g. Find neighbors `loc_next` of `loc_b` using the precomputed graph (`self.graph`).
        h. For each `loc_next` adjacent to `loc_b`:
            i. Calculate the shortest path distance from `loc_next` to `loc_g_b` (`dist_next_to_g`), treating `other_box_locations` as obstacles. Use BFS (`self.bfs_distance`).
            ii. If `dist_next_to_g` is strictly less than `dist_box` (meaning `loc_next` is closer to the goal than `loc_b`):
                - Determine the required robot location `loc_r` to push `b` from `loc_b` to `loc_next`. This `loc_r` is adjacent to `loc_b` in the direction opposite to the push from `loc_b` to `loc_next`. Use the `self.get_location_behind` helper function.
                - If `loc_r` exists (i.e., the box is not against a wall/edge in that direction):
                    - Calculate the shortest path distance `dist_robot_pos` from the robot's current location `loc_robot` to `loc_r` on the grid graph, treating `other_box_locations` as obstacles. Use BFS (`self.bfs_distance`).
                    - If `dist_robot_pos` is finite, update `min_robot_dist_to_push = min(min_robot_dist_to_push, dist_robot_pos)`.
        i. If `min_robot_dist_to_push` is still infinity (robot cannot reach any required position to push towards goal):
            # Robot cannot reach any required position to push towards goal.
            # Fallback: Add the box distance again as a penalty for the robot cost.
            # This implies the box still needs to move, but the robot is blocked or the box is in a local minimum.
            total_cost += dist_box
        j. Else (robot can reach at least one valid push position):
            - Add `min_robot_dist_to_push` to `h`.
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Static facts (`adjacent` relationships).
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the grid graph from adjacent facts
        # graph[loc1] = {loc2: dir, loc3: dir, ...}
        self.graph = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, dir = get_parts(fact)
                if loc1 not in self.graph:
                    self.graph[loc1] = {}
                self.graph[loc1][loc2] = dir

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Map direction to opposite direction
        self.opposite_dirs = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

    def bfs_distance(self, start, end, obstacles):
        """
        Calculate the shortest path distance between start and end locations
        on the grid graph, avoiding specified obstacles.
        Returns float('inf') if the end is unreachable.
        """
        if start == end:
            return 0
        q = deque([(start, 0)])
        visited = {start}

        while q:
            current_loc, dist = q.popleft()

            # Get neighbors from the graph
            neighbors = self.graph.get(current_loc, {}).keys()

            for neighbor_loc in neighbors:
                # A location is an obstacle if it's in the obstacles set
                # and it's not the target location (agent can move *to* the target location even if it's an obstacle for others)
                is_obstacle = neighbor_loc in obstacles and neighbor_loc != end

                if not is_obstacle and neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    q.append((neighbor_loc, dist + 1))

        return float('inf') # Target not reachable

    def get_location_behind(self, loc_box, loc_push_to):
        """
        Find the location loc_r such that pushing from loc_r at loc_box
        results in the box moving to loc_push_to.
        This means loc_r is adjacent to loc_box in the opposite direction
        of the push from loc_box to loc_push_to.
        Returns None if no such location exists (e.g., loc_box is on the edge).
        """
        # Find direction from loc_box to loc_push_to
        direction = None
        for neighbor, dir in self.graph.get(loc_box, {}).items():
            if neighbor == loc_push_to:
                direction = dir
                break
        if direction is None:
            return None # loc_push_to is not adjacent to loc_box

        # Find opposite direction
        opp_dir = self.opposite_dirs.get(direction)
        if opp_dir is None:
            return None # Should not happen with valid directions

        # Find location loc_r such that (adjacent loc_r loc_box opp_dir)
        # We need to iterate through all locations and check their neighbors
        for loc_r, neighbors in self.graph.items():
             if loc_box in neighbors and neighbors[loc_box] == opp_dir:
                 return loc_r

        return None # No location behind found (e.g., edge of grid)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot location
        loc_robot = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                loc_robot = get_parts(fact)[1]
                break
        if loc_robot is None:
             # Robot location not found in state, likely an invalid state
             return float('inf')

        # Find box locations for all boxes that have a goal
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.goal_locations: # Only care about boxes with goals
                    box_locations[obj] = loc

        # Check if all goal boxes are present in the state
        if len(box_locations) != len(self.goal_locations):
             # Some goal boxes are not found in the state's 'at' facts.
             # This state is likely invalid or unreachable.
             return float('inf')


        total_cost = 0  # Initialize action cost counter.

        # Iterate over boxes that need to reach a goal
        for box, goal_location in self.goal_locations.items():
            current_location = box_locations[box] # We know it exists due to check above

            if current_location == goal_location:
                continue # Box is already at its goal

            # Locations of other boxes (obstacles for BFS)
            other_box_locations = {loc for b, loc in box_locations.items() if b != box}

            # --- Box Distance Component ---
            # Shortest path for the box to its goal, avoiding other boxes.
            dist_box = self.bfs_distance(current_location, goal_location, obstacles=other_box_locations)

            if dist_box == float('inf'):
                 # Goal is unreachable for this box given current obstacles
                 return float('inf') # The whole state is likely unsolvable

            total_cost += dist_box

            # --- Robot Positioning Cost Component ---
            # Find the minimum distance for the robot to get into a position
            # to push the box one step closer to the goal.

            min_robot_dist_to_push = float('inf')

            # Find neighbors of the box's current location
            box_neighbors = self.graph.get(current_location, {}).keys()

            # Calculate box distance from neighbors to goal for comparison
            # We need the box distance from neighbors *without* the current box as obstacle
            # (since the box will move out of the way).
            # However, the BFS for the box itself *does* consider other boxes as obstacles.
            # Let's recalculate the distance from neighbors to goal considering other boxes as obstacles.
            # This is the same logic as calculating dist_box, just starting from a neighbor.
            neighbor_box_distances_to_goal = {}
            for neighbor_loc in box_neighbors:
                 neighbor_box_distances_to_goal[neighbor_loc] = self.bfs_distance(neighbor_loc, goal_location, obstacles=other_box_locations)


            # Get the current box distance to goal (already calculated as dist_box)
            current_box_dist_to_goal = dist_box


            for loc_next in box_neighbors:
                # Check if pushing to loc_next moves the box strictly closer to the goal
                dist_next_to_g = neighbor_box_distances_to_goal.get(loc_next, float('inf'))

                if dist_next_to_g < current_box_dist_to_goal:
                    # This loc_next is a valid step towards the goal for the box

                    # Find the required robot location behind the box
                    loc_r = self.get_location_behind(current_location, loc_next)

                    if loc_r is not None:
                        # Calculate robot distance to loc_r
                        # Obstacles for robot BFS: other_box_locations.
                        # The robot needs to reach loc_r. loc_r must be clear for the robot to move into it.
                        # The heuristic doesn't check for clear(loc_r), it estimates path cost.
                        # The path to loc_r must avoid other boxes.
                        robot_obstacles = other_box_locations

                        dist_robot_pos = self.bfs_distance(loc_robot, loc_r, obstacles=robot_obstacles)

                        if dist_robot_pos != float('inf'):
                             min_robot_dist_to_push = min(min_robot_dist_to_push, dist_robot_pos)


            # Add the minimum robot distance required to push this box
            if min_robot_dist_to_push == float('inf'):
                 # Robot cannot reach any required position to push towards goal.
                 # Fallback: Add the box distance again as a penalty for the robot cost.
                 # This implies the box still needs to move, but the robot is blocked or the box is in a local minimum.
                 total_cost += dist_box
            else:
                 total_cost += min_robot_dist_to_push


        return total_cost
