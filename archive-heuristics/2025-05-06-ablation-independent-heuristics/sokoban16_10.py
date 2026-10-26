from heuristics.heuristic_base import Heuristic
from task import Task # Assuming Task class is available

from collections import deque # For efficient queue


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal by summing the shortest path distances
    from each box to its specific goal location (as defined in the PDDL goal),
    plus the shortest path distance from the robot to the closest box that
    is not yet at its goal location. The distances are calculated using BFS
    on the graph defined by the 'adjacent' predicates. This heuristic is
    non-admissible and designed for greedy best-first search.

    Assumptions:
    - The state and static information are provided in the format described
      (frozensets of strings).
    - Location names are consistent identifiers used in 'at', 'at-robot',
      and 'adjacent' facts.
    - The graph defined by 'adjacent' predicates represents the traversable
      locations and connections.
    - The PDDL goal specifies the target location for each box that needs
      to be moved using '(at box_name location_name)' facts. Boxes not
      mentioned in the goal do not need to be at a specific location.
    - The only goal conditions are the locations of the boxes.

    Heuristic Initialization:
    - Parses the 'adjacent' facts from the static information to build an
      adjacency list representation of the location graph. Only locations
      listed in task.objects are included as nodes.
    - Identifies the specific goal location for each box by parsing the
      'at' facts in the task's goal state. Stores this as a mapping from
      box name to goal location (`self.box_goal_map`). Only boxes with
      specific goal locations are considered by the heuristic.
    - Stores the list of all box names and location names from task.objects.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state using task.goal_reached().
       If yes, return 0.
    2. Extract the robot's current location from the state. If the robot's
       location fact is not found or the location is not a known object,
       return infinity (state is likely invalid or unsolvable).
    3. Extract the current location of each box from the state. If any box
       that has a specific goal location (as identified during initialization)
       is not found in the state, return infinity (state is likely invalid
       or unsolvable).
    4. Calculate shortest path distances from the robot's current location
       to all other reachable locations using Breadth-First Search (BFS)
       on the location graph built during initialization.
    5. Initialize `total_box_goal_distance` to 0. This will sum the shortest
       distances each box needs to travel to reach its goal.
    6. Initialize `min_robot_to_box_distance` to infinity. This will track
       the minimum distance from the robot to any box that is not yet at
       its goal location.
    7. Initialize a flag `found_box_needing_move` to False. This flag helps
       distinguish between states where all relevant boxes are home (h=0)
       and states where boxes need moving but the robot is trapped (h=inf).
    8. Iterate through each box that has a specific goal location defined
       in `self.box_goal_map`:
        a. Get the box's current location from the state and its target
           goal location from `self.box_goal_map`.
        b. If the box's current location is not the same as its target
           goal location:
            i. Set `found_box_needing_move` to True.
            ii. Calculate shortest path distances from the box's current
                location to all other reachable locations using BFS.
            iii. Get the distance from the box's current location to its
                 target goal location using the distances from the BFS in
                 step 8.ii. If the goal location is unreachable from the
                 box's current location, return infinity (state is unsolvable).
            iv. Add this box-to-goal distance to `total_box_goal_distance`.
            v. Get the distance from the robot's location to the box's current
               location using the distances calculated in step 4. If the box
               is unreachable by the robot, return infinity (as the robot must
               reach the box to push it).
            vi. Update `min_robot_to_box_distance` with the minimum of its
                current value and the robot-to-this-box distance calculated
                in the previous step.
    9. After iterating through all boxes with specific goals, if
       `found_box_needing_move` is True but `min_robot_to_box_distance` is
       still infinity (meaning the robot cannot reach any box that needs moving),
       return infinity (state is unsolvable).
    10. If the goal was not reached (checked in step 1), and the state is not
        unsolvable (no infinities returned), the heuristic value is the sum
        of the total box-to-goal distance and the minimum robot-to-box distance
        calculated for the boxes needing movement. If `found_box_needing_move`
        is False, it implies all boxes with specific goals are already at their
        goals, which means the goal is reached (handled by step 1), so this
        case should not result in reaching this step with a non-zero heuristic.

    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.goals = task.goals
        # Get box names from task.objects dictionary {obj_name: type_name}
        self.boxes = [obj_name for obj_name, obj_type in task.objects.items() if obj_type == 'box']
        # Get location names from task.objects dictionary {obj_name: type_name}
        self.locations = [obj_name for obj_name, obj_type in task.objects.items() if obj_type == 'location']

        # Build the graph from adjacent facts
        self.graph = {}
        for loc in self.locations:
            self.graph[loc] = []

        for fact_str in task.static:
            if fact_str.startswith('(adjacent '):
                # Fact format: '(adjacent loc1 loc2 direction)'
                parts = fact_str[1:-1].split()
                # Ensure fact has enough parts and locations are valid objects
                if len(parts) >= 3:
                    loc1 = parts[1]
                    loc2 = parts[2]
                    if loc1 in self.graph and loc2 in self.graph:
                         # Add bidirectional edge
                         self.graph[loc1].append(loc2)
                         self.graph[loc2].append(loc1)
                    # else: ignore adjacent facts involving unknown locations

        # Identify specific box goal locations from task.goals
        self.box_goal_map = {} # box_name -> goal_location
        for goal_fact_str in self.goals:
             if goal_fact_str.startswith('(at '):
                  pred, args = self._parse_fact(goal_fact_str)
                  # Ensure fact is '(at box loc)' and box/loc are known objects
                  if len(args) == 2 and args[0] in self.boxes and args[1] in self.locations:
                       box_name = args[0]
                       goal_loc = args[1]
                       self.box_goal_map[box_name] = goal_loc
                  # else: ignore goal facts about non-boxes or unknown locations

        # We only care about boxes that actually have a specific goal location
        self.boxes_with_goals = list(self.box_goal_map.keys())


    def _parse_fact(self, fact_string):
         """Helper to parse a fact string into (predicate, arg1, arg2, ...)"""
         # Remove leading/trailing parentheses and split by space
         parts = fact_string[1:-1].split()
         return parts[0], parts[1:]

    def _bfs(self, start_location, graph):
        """
        Performs BFS from start_location on the graph.
        Returns a dictionary mapping reachable locations to their shortest distance.
        Returns an empty dictionary if start_location is not in the graph.
        """
        distances = {loc: float('inf') for loc in graph}
        if start_location not in graph:
             # Start location is not a known location in the graph
             return {} # Return empty dict if start is invalid

        distances[start_location] = 0
        queue = deque([start_location]) # Use deque for O(1) popleft
        visited = {start_location}

        while queue:
            current_loc = queue.popleft() # Use popleft for queue behavior

            # Check if current_loc is a valid node in the graph dictionary
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    # Check if neighbor is a valid node in the graph dictionary
                    if neighbor in graph and neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # 2. Find robot and box locations
        robot_loc = None
        box_locations = {} # box_name -> location
        # Convert state frozenset to a set of parsed facts for easier lookup
        parsed_state = {self._parse_fact(fact_str) for fact_str in state}

        for pred, args in parsed_state:
            if pred == 'at-robot' and len(args) == 1 and args[0] in self.locations:
                robot_loc = args[0]
            elif pred == 'at' and len(args) == 2 and args[0] in self.boxes and args[1] in self.locations:
                box_locations[args[0]] = args[1]

        # Basic validation: Robot must be present and all boxes mentioned in goals must be present
        if robot_loc is None:
             # Robot location fact not found in state
             return float('inf')

        for box_name in self.boxes_with_goals:
             if box_name not in box_locations:
                  # Box mentioned in goal not found in state
                  return float('inf')


        # 3. Calculate distances from robot
        dist_from_robot = self._bfs(robot_loc, self.graph)
        if not dist_from_robot: # Robot's location is not in the graph or graph is empty
             return float('inf')


        # 4. Calculate box-to-goal distances and robot-to-box distances for boxes needing move
        total_box_goal_distance = 0
        min_robot_to_box_distance = float('inf')
        found_box_needing_move = False # Flag to check if any box needs moving

        for box_name in self.boxes_with_goals:
            target_goal_loc = self.box_goal_map[box_name]
            current_box_loc = box_locations[box_name]

            if current_box_loc != target_goal_loc:
                found_box_needing_move = True

                # Calculate distance from box to its specific goal
                dist_from_box = self._bfs(current_box_loc, self.graph)
                if not dist_from_box: # Box's location is not in the graph or graph is empty
                     return float('inf')

                box_to_goal_dist = dist_from_box.get(target_goal_loc, float('inf'))

                if box_to_goal_dist == float('inf'):
                    # Box cannot reach its goal location
                    return float('inf')

                total_box_goal_distance += box_to_goal_dist

                # Update min robot to box distance for this box
                robot_to_this_box_dist = dist_from_robot.get(current_box_loc, float('inf'))
                # If robot cannot reach this box, the state is likely unsolvable
                if robot_to_this_box_dist == float('inf'):
                     return float('inf')
                min_robot_to_box_distance = min(min_robot_to_box_distance, robot_to_this_box_dist)


        # If the goal was not reached (checked in step 1), and we found boxes
        # needing movement, and the robot can reach at least one of them,
        # calculate the heuristic.
        # If found_box_needing_move is False, it means all boxes with specific
        # goals are at their goals, which implies the goal is reached (handled
        # by the initial check). So, if we reach here, found_box_needing_move
        # must be True (assuming standard Sokoban goals).

        # The check `if found_box_needing_move and min_robot_to_box_distance == float('inf'):`
        # correctly handles the case where boxes need moving but the robot is trapped.
        # If that check didn't return infinity, it means min_robot_to_box_distance is finite.

        # The heuristic is the sum of box-goal distances + distance from robot to closest box needing move
        return total_box_goal_distance + min_robot_to_box_distance
