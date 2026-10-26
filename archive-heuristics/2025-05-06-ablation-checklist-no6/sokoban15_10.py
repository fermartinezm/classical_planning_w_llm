# Assuming heuristics.heuristic_base provides the Heuristic base class
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque
import math # Used for math.inf

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class
# Inherit from Heuristic if available, otherwise just define the class
# class sokobanHeuristic(Heuristic):
class sokobanHeuristic:
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost as the sum of shortest path distances for each box
    to its goal, plus the shortest path distance for the robot to the
    closest location from which it can push a box towards its goal.

    Uses BFS to precompute shortest path distances on the static grid defined
    by 'adjacent' facts. Includes basic checks for unreachable goals or
    potential deadlocks by returning infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and
        precomputing shortest path distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the adjacency graph from static facts
        # This graph is used for robot movement and box movement paths
        self.graph = {} # loc -> set of adjacent locs
        self.all_locations = set()
        # Store directions for edges: (loc1, loc2) -> dir
        self.location_directions = {}
        # Store reverse mapping for push positions: (box_loc, push_dir) -> robot_loc_needed
        # This maps a box location and a desired push direction to the location
        # the robot must be in to perform that push.
        self.reverse_graph = {}

        for fact in static_facts:
             if match(fact, "adjacent", "*", "*", "*"):
                 _, loc1, loc2, dir = get_parts(fact)
                 self.all_locations.add(loc1)
                 self.all_locations.add(loc2)

                 # Add edge for BFS graph (assuming symmetric movement for distance)
                 # The graph represents connectivity for movement (robot or box).
                 self.graph.setdefault(loc1, set()).add(loc2)
                 self.graph.setdefault(loc2, set()).add(loc1)

                 # Store directional mapping for finding push direction later
                 self.location_directions[(loc1, loc2)] = dir

                 # Store reverse mapping for push action:
                 # The push action (push ?rloc ?bloc ?floc ?dir ?b) requires
                 # (adjacent ?rloc ?bloc ?dir) and (adjacent ?bloc ?floc ?dir).
                 # If a box is at ?bloc and needs to be pushed in direction ?dir
                 # (meaning it moves to ?floc), the robot must be at ?rloc.
                 # So, to push a box at loc2 in direction dir, the robot needs to be at loc1.
                 self.reverse_graph[(loc2, dir)] = loc1


        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.all_locations:
            self.distances[start_node] = self._bfs(start_node)

        # Store box goal locations
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                # Assuming 'at' goals in Sokoban are only for boxes
                self.box_goals[box] = location

    def _bfs(self, start_node):
        """Performs BFS from start_node to find distances to all reachable nodes."""
        distances = {node: math.inf for node in self.all_locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # If current_node is not in graph, it has no neighbors, BFS stops for this branch
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def _get_first_step_location(self, start_loc, end_loc):
        """
        Finds the location of the first step on a shortest path from start_loc to end_loc.
        Returns the location string, or None if no path or start_loc == end_loc.
        Assumes distances are precomputed in self.distances.
        """
        if start_loc == end_loc:
            return None # No step needed

        # Check if end_loc is reachable from start_loc
        if start_loc not in self.distances or end_loc not in self.distances[start_loc] or self.distances[start_loc][end_loc] == math.inf:
             return None # No path

        # Find a neighbor of start_loc that is one step closer to end_loc
        # Iterate through neighbors of start_loc
        for neighbor in self.graph.get(start_loc, []):
             # Check if the neighbor is reachable from start_loc (it is, by definition of graph)
             # and if the distance from neighbor to end_loc is one less than
             # the distance from start_loc to end_loc.
             if end_loc in self.distances[neighbor] and self.distances[neighbor][end_loc] == self.distances[start_loc][end_loc] - 1:
                 # Found the first step location (neighbor)
                 return neighbor

        # This case should ideally not be reached if end_loc is reachable and graph is consistent.
        # It might indicate an issue with graph building or distance calculation.
        # Returning None signals that a valid first step cannot be determined.
        return None


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        # If robot location is not found, state is invalid or unreachable
        if robot_loc is None or robot_loc not in self.all_locations:
             return math.inf

        # Find box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)
                 # Only consider objects that are goals (i.e., boxes)
                 if obj in self.box_goals:
                    box_locations[obj] = loc

        total_box_distance = 0
        min_robot_to_push_pos_distance = math.inf
        found_box_to_move = False

        # Iterate through all boxes that have a goal
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locations.get(box)

            # If a box is not found in the state, it's an invalid state
            if current_loc is None or current_loc not in self.all_locations:
                 return math.inf

            # If the box is not at its goal location
            if current_loc != goal_loc:
                found_box_to_move = True

                # Check if the box goal is reachable from its current location on the static graph
                if goal_loc not in self.distances[current_loc] or self.distances[current_loc][goal_loc] == math.inf:
                     return math.inf # Box goal is unreachable

                # Add the shortest path distance for the box to its goal
                # This represents the minimum number of pushes required for this box
                box_dist = self.distances[current_loc][goal_loc]
                total_box_distance += box_dist

                # Find the location of the first step for the box towards the goal
                first_step_loc = self._get_first_step_location(current_loc, goal_loc)

                # If no first step can be determined, the box might be in a deadlock
                # (e.g., in a corner not on a goal, or surrounded in a way that blocks progress).
                # _get_first_step_location returns None if goal is unreachable or start==end.
                # We already checked start!=end and goal reachability, so None here implies deadlock.
                if first_step_loc is None:
                     return math.inf # Indicate deadlock


                # Find the direction the box needs to move in the first step
                # This is the direction from current_loc to first_step_loc
                box_move_dir = self.location_directions.get((current_loc, first_step_loc))

                # This should exist if first_step_loc is a valid neighbor found via graph
                if box_move_dir is None:
                    # This is unexpected if graph and location_directions are built correctly
                    return math.inf # Indicate internal error or unexpected graph structure


                # The robot needs to be at the location adjacent to current_loc
                # in the direction *opposite* to box_move_dir to push it.
                # The reverse_graph maps (box_loc, push_dir) -> robot_loc_needed
                # Here, box_loc is current_loc, and push_dir is box_move_dir.
                required_robot_pos = self.reverse_graph.get((current_loc, box_move_dir))

                # If there is no location from which the robot can push the box
                # in the required direction according to the static adjacent facts,
                # it's a form of deadlock or an impossible state.
                if required_robot_pos is None:
                     return math.inf # Indicate deadlock (cannot get behind box in required direction)


                # Calculate robot distance from its current location to this required position
                # Check if the required push position is reachable by the robot on the static graph
                if required_robot_pos not in self.distances[robot_loc] or self.distances[robot_loc][required_robot_pos] == math.inf:
                     return math.inf # Required push position is unreachable for the robot

                robot_dist_to_push_pos = self.distances[robot_loc][required_robot_pos]

                # We want the minimum robot distance to *any* required push position
                # for *any* box that needs moving. This estimates the robot's cost
                # to get into position to start making progress on any box.
                min_robot_to_push_pos_distance = min(min_robot_to_push_pos_distance, robot_dist_to_push_pos)


        # If no boxes need to move, the heuristic is 0.
        if not found_box_to_move:
            return 0

        # If min_robot_to_push_pos_distance is still infinity, it means there are boxes
        # to move, but the robot cannot reach any required push position for any of them.
        # This indicates an unreachable state.
        if min_robot_to_push_pos_distance == math.inf:
             return math.inf


        # The heuristic is the sum of box distances (minimum pushes required for all boxes)
        # plus the robot's distance to the closest position from which it can
        # make the first push for any box that needs moving.
        # This estimates the total box movement cost and the initial robot setup cost.
        return total_box_distance + min_robot_to_push_pos_distance
