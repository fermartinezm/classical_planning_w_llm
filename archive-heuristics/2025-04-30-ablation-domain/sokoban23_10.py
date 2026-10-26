# Assuming heuristic_base is in a 'heuristics' directory relative to where this file will be placed
from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    return fact[1:-1].split()

# Helper function to match PDDL facts (optional, but good practice)
def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box from its current location to its goal location.
    It also adds the shortest path distance for the robot to reach a location
    adjacent to any box that is not yet at its goal.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - Shortest path distance on the location graph is a reasonable estimate
      for the minimum number of pushes required for a box (ignoring obstacles
      and robot positioning constraints along the path).
    - The robot needs to be adjacent to a box to push it. Adding the robot's
      distance to the nearest pushable box location encourages moving the robot
      towards the action.
    - Unreachable goal locations for a box indicate an unsolvable state, assigned
      a very high heuristic value.

    # Heuristic Initialization
    - Parses `adjacent` facts to build an undirected graph of locations.
    - Parses goal facts to identify the target location for each box.
    - Precomputes shortest path distances from all locations to each unique goal location
      using BFS. This allows quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and each box from the state.
    2. Initialize the total heuristic cost to 0.
    3. For each box that is not at its goal location:
       - Look up the precomputed shortest distance from the box's current location
         to its goal location.
       - If the goal is unreachable from the box's current location (based on precomputation),
         return a very high cost (indicating a likely dead end or unsolvable state).
       - Add this distance to the total cost. This represents the minimum number
         of pushes required for the box if the path were clear.
    4. If there are boxes not yet at their goals:
       - Find all locations that are adjacent to any box not at its goal. These are
         potential starting locations for a push action.
       - Compute the shortest path distances from the robot's current location to
         all other locations using BFS.
       - Find the minimum distance from the robot's current location to any of the
         potential push starting locations identified above.
       - Add this minimum robot distance to the total cost. This encourages the
         robot to move towards a box that needs pushing.
    5. Return the total calculated cost. If all boxes are at their goals, the cost
       will be 0.
    """

    def __init__(self, task):
        """
        Initializes the Sokoban heuristic.

        Args:
            task: The planning task object containing initial state, goals, and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the graph from adjacent facts
        self.graph = {}
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Add bidirectional edges assuming adjacency is symmetric
                # Avoid adding duplicates if facts list both directions
                if loc2 not in self.graph[loc1]:
                    self.graph[loc1].append(loc2)
                if loc1 not in self.graph[loc2]:
                    self.graph[loc2].append(loc1)


        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            # Goal is typically (at boxX loc_Y_Z)
            if match(goal, "at", "*", "*"):
                _, box, location = get_parts(goal)
                self.goal_locations[box] = location

        # Precompute shortest distances from all locations to each unique goal location
        # We will run BFS starting from each goal location.
        self.dist_to_goals = {}
        unique_goal_locations = set(self.goal_locations.values())

        # Need distances *from* any location *to* a goal location.
        # Running BFS from each goal location on the graph gives distances *from* the goal *to* other locations.
        # Since the graph is undirected, dist(A, B) == dist(B, A). So BFS from goal_loc gives us dist(any_loc, goal_loc).
        for goal_loc in unique_goal_locations:
             # Ensure the goal location exists in the graph before running BFS
             if goal_loc in self.graph:
                self.dist_to_goals[goal_loc] = self._bfs(goal_loc)
             else:
                 # This goal location is isolated or not defined by adjacency facts.
                 # Mark it as unreachable from any known location.
                 self.dist_to_goals[goal_loc] = {}

    def _bfs(self, start_node):
        """
        Performs BFS from a start_node to find distances to all reachable nodes.

        Args:
            start_node: The location node to start BFS from.

        Returns:
            A dictionary mapping reachable location nodes to their shortest distance from start_node.
        """
        distances = {start_node: 0}
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            curr_node = queue.popleft()

            # Check if curr_node exists in graph (handle potential isolated locations not in any adjacent fact)
            if curr_node not in self.graph:
                 continue

            for neighbor in self.graph[curr_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[curr_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An estimated cost (integer or float('inf')) to reach a goal state.
        """
        state = node.state

        # Find current location of robot and each box
        current_box_locations = {}
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                _, obj, loc = parts
                # Check if this object is one of the boxes we care about (i.e., has a goal)
                if obj in self.goal_locations:
                     current_box_locations[obj] = loc
            elif parts[0] == "at-robot":
                 _, loc = parts
                 robot_location = loc

        total_cost = 0
        boxes_needing_push_locs = [] # Track locations of boxes not at goal

        # Sum distances for boxes not at their goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)
            if current_loc is None:
                 # This box is not in the state? Indicates an invalid state representation.
                 # Treat as unsolvable.
                 return float('inf')

            if current_loc != goal_loc:
                boxes_needing_push_locs.append(current_loc)
                # Get distance from current_loc to goal_loc using precomputed BFS results
                if goal_loc in self.dist_to_goals and current_loc in self.dist_to_goals[goal_loc]:
                     total_cost += self.dist_to_goals[goal_loc][current_loc]
                else:
                     # Box is in a location from which its goal is unreachable based on the graph.
                     # This state is likely unsolvable.
                     return float('inf')

        # Add robot distance component if there are boxes needing pushes
        if boxes_needing_push_locs and robot_location:
            min_robot_dist_to_push_pos = float('inf')

            # Compute distances from the robot's current location
            # Ensure robot_location is in the graph before running BFS
            if robot_location in self.graph:
                robot_distances = self._bfs(robot_location)

                # Find the minimum distance from the robot to any location adjacent to a box needing a push.
                potential_push_start_locations = set()
                for box_loc in boxes_needing_push_locs:
                     if box_loc in self.graph: # Ensure box location is in the graph
                         for adj_loc in self.graph[box_loc]:
                             potential_push_start_locations.add(adj_loc)

                for push_start_loc in potential_push_start_locations:
                     if push_start_loc in robot_distances:
                         min_robot_dist_to_push_pos = min(min_robot_dist_to_push_pos, robot_distances[push_start_loc])

                # Only add robot distance if a push start location is reachable
                if min_robot_dist_to_push_pos != float('inf'):
                     total_cost += min_robot_dist_to_push_pos
                # else: robot cannot reach any location adjacent to a box needing push?
                # This state might be a dead end. Assign high cost.
                else:
                    return float('inf')
            else:
                 # Robot is in a location not in the graph? Invalid state.
                 return float('inf')


        return total_cost
