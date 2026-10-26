# Assuming the base class Heuristic is available in the planner environment
# from heuristics.heuristic_base import Heuristic

from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def build_location_graph(static_facts):
    """Builds an adjacency list graph from 'adjacent' static facts."""
    graph = {}
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == "adjacent" and len(parts) == 4:
            loc1, loc2, direction = parts[1], parts[2], parts[3]
            if loc1 not in graph:
                graph[loc1] = set()
            if loc2 not in graph:
                graph[loc2] = set()
            graph[loc1].add(loc2)
            graph[loc2].add(loc1) # Adjacency is symmetric
    return graph

def bfs_distance(graph, start, end, obstacles=None):
    """
    Computes the shortest path distance between start and end locations
    on the graph, avoiding specified obstacles.
    """
    if obstacles is None:
        obstacles = set()

    if start == end:
        return 0

    # Check if start or end locations are valid nodes in the graph
    if start not in graph or end not in graph:
        return float('inf')

    # Cannot start or end inside an obstacle
    if start in obstacles or end in obstacles:
         return float('inf')

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        # Neighbors must be in the graph, not visited, and not obstacles
        for neighbor in graph.get(current_loc, []): # Use .get for safety
            if neighbor not in visited and neighbor not in obstacles:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # No path found


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost based on the sum of shortest path distances from
    each box to its goal, plus the shortest path distance from the robot
    to the closest box that is not yet at its goal.

    Box distances are computed on the full location graph.
    Robot distances are computed on the location graph, avoiding locations
    occupied by other boxes.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph from static facts.
        """
        # Assuming the base class Heuristic is available and initialized elsewhere
        # super().__init__(task) # If base class needs explicit init call

        self.goals = task.goals
        self.static = task.static

        # Build the location graph from adjacent facts
        self.location_graph = build_location_graph(self.static)

        # Map each box to its goal location
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal predicates are typically (at boxX locY)
            if parts and parts[0] == "at" and len(parts) == 3:
                 obj_name, loc_name = parts[1], parts[2]
                 # Assuming objects of type 'box' are the ones in goals
                 # Check if the object name starts with 'box' (heuristic assumption based on examples)
                 if obj_name.startswith("box"):
                     self.box_goals[obj_name] = loc_name
                 # Note: If the goal is something else like (at-robot loc), this heuristic ignores it.
                 # Sokoban goals are typically only about box locations.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 2) The heuristic is 0 only for goal states.
        if self.goals <= state:
            return 0

        # Find current robot location
        robot_location = None
        # Using a set for state allows efficient lookup
        state_set = set(state) # Convert frozenset to set for faster iteration/checking if needed, though iteration is fine here

        for fact in state_set:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
                break

        # If robot location is not found, the state is likely invalid or unreachable
        if robot_location is None:
             # 3) The heuristic value is finite for solvable states.
             # An unreachable state should have infinite heuristic.
             return float('inf')

        # Find current box locations for boxes that have goals
        box_locations = {}
        # Collect all 'at' facts first for efficiency
        at_facts = {}
        for fact in state_set:
             parts = get_parts(fact)
             if parts and parts[0] == "at" and len(parts) == 3:
                  at_facts[parts[1]] = parts[2]

        for box_name in self.box_goals.keys():
             current_loc = at_facts.get(box_name)
             if current_loc is None:
                  # A box with a goal is not in the state? Invalid state.
                  return float('inf')
             box_locations[box_name] = current_loc


        total_heuristic = 0
        boxes_not_at_goal = []

        # Calculate box-goal distances for boxes not at their goals
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locations[box]

            if current_loc != goal_loc:
                boxes_not_at_goal.append(box)
                # Distance for the box to reach its goal.
                # This is the minimum number of pushes required for the box itself,
                # assuming the path is always clearable. Obstacles set is empty.
                box_dist = bfs_distance(self.location_graph, current_loc, goal_loc)

                if box_dist == float('inf'):
                    # Box goal is unreachable on the static graph (e.g., isolated location)
                    # This state is likely unsolvable.
                    return float('inf')
                total_heuristic += box_dist

        # If all boxes are at goals, the heuristic is 0 (handled at the start)
        # If some boxes are not at goals, add robot cost
        if boxes_not_at_goal:
            # Find the closest box that is not at its goal
            min_robot_box_dist = float('inf')

            # Obstacles for robot movement are locations occupied by *other* boxes.
            # The robot cannot move through locations occupied by boxes it is *not* currently interacting with.
            # The location of the box being considered for the robot distance calculation is *not* an obstacle
            # for the robot's path *to* that box.
            all_box_locations = set(box_locations.values())

            for box in boxes_not_at_goal:
                 current_loc = box_locations[box]
                 # Obstacles for robot path to 'current_loc' are all box locations *except* 'current_loc' itself.
                 current_robot_obstacles = all_box_locations - {current_loc}

                 dist_robot_to_this_box = bfs_distance(self.location_graph, robot_location, current_loc, obstacles=current_robot_obstacles)

                 if dist_robot_to_this_box < min_robot_box_dist:
                     min_robot_box_dist = dist_robot_to_this_box

            # If min_robot_box_dist is still inf, it means the robot cannot reach *any* box that needs moving.
            # This state is likely unsolvable.
            if min_robot_box_dist == float('inf'):
                 return float('inf')

            # Add the distance from the robot to the closest box that needs moving
            total_heuristic += min_robot_box_dist

        return total_heuristic
