from heuristics.heuristic_base import Heuristic
from task import Task
from collections import deque

class sokobanHeuristic(Heuristic):
    """
    Sokoban Domain-Dependent Heuristic.

    Summary:
        This heuristic estimates the cost to reach a goal state by summing two components:
        1. The sum of shortest path distances for each box from its current location
           to its assigned goal location, calculated on the full grid graph.
        2. The shortest path distance for the robot from its current location to
           any location adjacent to a box that needs to be moved, calculated on
           the subgraph of locations not occupied by boxes.

    Assumptions:
        - The PDDL domain follows the structure provided (predicates at-robot, at, clear, adjacent).
        - Locations are connected via 'adjacent' facts forming a graph.
        - Goal facts specify the target location for each specific box using the 'at' predicate.
        - Fact strings are in the format '(predicate arg1 arg2 ...)' with no spaces or
          parentheses within arguments.

    Heuristic Initialization:
        - Parses the static 'adjacent' facts from the task to build an adjacency
          graph representation of the grid. This graph is stored in `self.adj_graph`.
        - Parses the goal facts from the task to create a mapping from box names
          to their target goal locations. This mapping is stored in `self.box_goals`.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Identify the current location of the robot (`robot_loc`).
        2. Identify the current location of each box (`box_locs`).
        3. Determine the set of locations currently occupied by boxes (`occupied_by_box`).
        4. Determine the set of locations the robot can traverse: all locations in the
           grid graph that are not occupied by a box (`robot_traversable_locations`).
        5. Determine which boxes are not currently at their assigned goal locations (`boxes_to_move`).
        6. If there are no boxes to move, the state is a goal state, and the heuristic value is 0.
        7. Calculate the first component: Sum of box-to-goal distances.
           For each box `b` in `boxes_to_move`:
           - Find the shortest path distance from the box's current location (`box_locs[b]`)
             to its goal location (`self.box_goals[b]`) using Breadth-First Search (BFS)
             on the full grid graph (`self.adj_graph`). Obstacles are ignored for box movement
             in this relaxation.
           - If any box's goal is unreachable on the full graph, the state is considered
             a dead end (in this relaxation), and the heuristic value is infinity.
           - Sum these distances.
        8. Calculate the second component: Robot-to-pushing-area distance.
           - Identify the set of target box locations: the current locations (`box_locs[b]`)
             for all boxes `b` in `boxes_to_move`.
           - Find the shortest path distance from the robot's current location (`robot_loc`)
             to *any* location `l` such that `l` is adjacent to a location in the target
             box locations set. This BFS is restricted: the robot can only traverse
             locations in `robot_traversable_locations`.
           - If the robot cannot reach any location adjacent to a box that needs moving
             (using only traversable locations), the heuristic value is infinity.
        9. The total heuristic value is the sum of the box-to-goal distance sum and the
           robot-to-pushing-area distance.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.adj_graph = self._build_graph(task.static)
        self.box_goals = self._get_box_goals(task.goals)

    def _parse_fact(self, fact_str):
        """Parses a PDDL fact string into a predicate and arguments."""
        # Remove leading/trailing parens and split by space
        parts = fact_str[1:-1].split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _build_graph(self, static_facts):
        """Builds an adjacency graph from 'adjacent' facts."""
        graph = {}
        all_locations = set()

        for fact_str in static_facts:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'adjacent':
                loc1, loc2, direction = args
                all_locations.add(loc1)
                all_locations.add(loc2)

                if loc1 not in graph:
                    graph[loc1] = {}
                graph[loc1][loc2] = direction # Store direction (optional for BFS distance)

                # Assuming adjacency is symmetric, add the reverse edge
                if loc2 not in graph:
                    graph[loc2] = {}
                graph[loc2][loc1] = None # Direction not used in BFS distance

        # Ensure all locations mentioned in adjacencies are keys in the graph
        for loc in all_locations:
            if loc not in graph:
                graph[loc] = {}

        return graph

    def _get_box_goals(self, goal_facts):
        """Extracts box goal locations from goal facts."""
        box_goals = {}
        for fact_str in goal_facts:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'at':
                # Goal fact is (at box_name goal_loc)
                box_name, goal_loc = args
                box_goals[box_name] = goal_loc
        return box_goals

    def _bfs_distance(self, start, end, graph):
        """
        Calculates the shortest path distance between two locations using BFS
        on the given graph. Obstacles are not considered in this version
        (used for box-to-goal distance).
        """
        if start == end:
            return 0
        if start not in graph or end not in graph:
             return float('inf') # Start or end location not in graph

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            if current_loc not in graph:
                 continue # Should not happen if graph is built correctly

            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # End not reachable

    def _bfs_robot_to_box_adjacency(self, start, box_locations_to_target, graph, robot_traversable_locations):
        """
        Calculates the shortest path distance from the robot's start location
        to any location adjacent to a target box location, traversing only
        locations in robot_traversable_locations.
        """
        # Check if start is already adjacent to any target box location
        if start in graph:
            for neighbor in graph[start]:
                if neighbor in box_locations_to_target:
                    return 0 # Robot is already adjacent to a box that needs moving

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc not in graph:
                 continue # Should not happen if graph is built correctly

            for neighbor in graph[current_loc]:
                # Robot can only move into traversable locations
                if neighbor in robot_traversable_locations and neighbor not in visited:
                    # Check if this neighbor is adjacent to any target box location
                    is_adjacent_to_target_box = False
                    if neighbor in graph: # Ensure neighbor is in graph before checking its neighbors
                        for next_neighbor in graph[neighbor]:
                            if next_neighbor in box_locations_to_target:
                                is_adjacent_to_target_box = True
                                break # Found a target adjacency

                    if is_adjacent_to_target_box:
                        return dist + 1 # Found shortest path to a location adjacent to a box

                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # Cannot reach any location adjacent to a box


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        # 1. Get current robot and box locations
        robot_loc = None
        box_locs = {}

        for fact_str in state:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'at-robot':
                robot_loc = args[0]
            elif predicate == 'at':
                box_name, current_loc = args
                box_locs[box_name] = current_loc

        # If robot_loc is None, something is wrong
        if robot_loc is None:
             return float('inf')

        # Determine locations occupied by boxes
        occupied_by_box = set(box_locs.values())

        # Determine locations the robot can traverse (all locations not occupied by a box)
        robot_traversable_locations = set(self.adj_graph.keys()) - occupied_by_box

        # 2. Identify boxes not at goal
        boxes_to_move = [b for b, loc in box_locs.items() if self.box_goals.get(b) != loc]

        # 3. If all boxes are at goal, heuristic is 0
        if not boxes_to_move:
            return 0

        # 4. Calculate sum of box-to-goal distances
        box_distance_sum = 0
        for box_name in boxes_to_move:
            current_loc = box_locs[box_name]
            goal_loc = self.box_goals.get(box_name)

            # BFS for box movement uses the full graph (obstacles ignored)
            dist = self._bfs_distance(current_loc, goal_loc, self.adj_graph)
            if dist == float('inf'):
                 # Box goal is unreachable from its current location on the full graph
                 return float('inf') # Dead end
            box_distance_sum += dist

        # 5. Calculate robot distance to a pushing position
        # Robot needs to reach a location adjacent to any box that needs moving.
        box_locations_to_target = {box_locs[b] for b in boxes_to_move}

        # Robot BFS must avoid locations occupied by boxes.
        # It can traverse any location that is not occupied by a box.
        robot_distance = self._bfs_robot_to_box_adjacency(
            robot_loc,
            box_locations_to_target,
            self.adj_graph,
            robot_traversable_locations
        )

        # If robot cannot reach any location adjacent to a box that needs moving
        if robot_distance == float('inf'):
             return float('inf') # Dead end

        # 6. Total heuristic
        total_heuristic = box_distance_sum + robot_distance

        return total_heuristic

