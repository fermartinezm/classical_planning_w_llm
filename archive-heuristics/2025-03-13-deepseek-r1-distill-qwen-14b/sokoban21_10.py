from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the total number of actions required to move all boxes to their goal locations. It considers both the robot's movement to reach each box and the number of pushes needed to move the box to its goal.

    # Assumptions:
    - The Sokoban domain involves moving boxes to specific locations.
    - The robot can move between adjacent locations and push boxes.
    - The shortest path between any two locations is precomputed.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task's goals.
    - Build an adjacency graph from the static 'adjacent' facts.
    - Precompute the shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box, determine its current location and its goal location.
    2. If the box is already at its goal, no actions are needed.
    3. For each box not at its goal:
       a. Calculate the shortest distance the robot needs to move from its current location to the box's current location.
       b. Calculate the shortest distance the box needs to move from its current location to its goal location.
       c. Sum these two distances for the box.
    4. Sum the distances for all boxes to get the total estimated actions.
    """

    def __init__(self, task):
        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in task.goals:
            if goal.startswith('(at box'):
                parts = get_parts(goal)
                box = parts[0]
                loc = parts[1]
                self.goal_locations[box] = loc

        # Build adjacency graph from static facts
        self.adjacency = {}
        for fact in task.static:
            if fact.startswith('(adjacent '):
                parts = get_parts(fact)
                loc1, loc2, _ = parts
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = []
                self.adjacency[loc1].append(loc2)
                if loc2 not in self.adjacency:
                    self.adjacency[loc2] = []
                self.adjacency[loc2].append(loc1)

        # Precompute shortest path distances between all pairs of locations
        self.distances = {}
        all_locations = set()
        for fact in task.static:
            if fact.startswith('(adjacent '):
                loc1, loc2, _ = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
        for loc in all_locations:
            self._compute_distances(loc)

    def _compute_distances(self, start):
        visited = {}
        queue = deque()
        queue.append(start)
        visited[start] = 0
        while queue:
            current = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
        for loc, dist in visited.items():
            self.distances[(start, loc)] = dist

    def __call__(self, node):
        state = node.state
        total = 0

        # Extract robot's current location
        robot_loc = None
        for fact in state:
            if fact.startswith('(at-robot '):
                robot_loc = get_parts(fact)[1]
                break

        # Extract current locations of boxes
        boxes = {}
        for fact in state:
            if fact.startswith('(at box'):
                parts = get_parts(fact)
                box = parts[0]
                loc = parts[1]
                boxes[box] = loc

        # For each box, calculate the required actions
        for box in boxes:
            current_loc = boxes[box]
            goal_loc = self.goal_locations.get(box)
            if current_loc != goal_loc:
                # Get distances
                robot_to_box = self.distances.get((robot_loc, current_loc), float('inf'))
                box_to_goal = self.distances.get((current_loc, goal_loc), float('inf'))
                total += robot_to_box + box_to_goal

        return total
