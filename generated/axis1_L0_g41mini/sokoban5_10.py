from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_3_5)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Summary
    -------
    This heuristic estimates the number of actions required to solve a Sokoban state,
    by accounting for the estimated cost to push all boxes onto their goal locations,
    combined with the estimated minimal cost for the robot to move to the locations necessary
    to perform those pushes. The heuristic balances between accuracy and efficiency
    to guide a greedy best-first search effectively.

    Assumptions
    -----------
    - There are one or more boxes, each with exactly one goal location.
    - The robot can only push boxes by moving adjacent to them and pushing into a free location.
    - All boxes need to be on their respective goal locations for the goal to be reached.
    - The domain static facts include adjacency relations between locations.
    - The heuristic does not need to be admissible but should remain efficiently computable.

    Heuristic Initialization
    ------------------------
    - Extract the set of all static adjacencies to build a bidirectional adjacency graph of locations.
    - Identify and store the goal locations of all boxes from the goal facts (predicate "at" with boxes).
    - Precompute shortest distances between all locations using a BFS for fast heuristic computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the current robot location and box locations from the current state.
    2. For each box:
        a) Compute the minimum shortest path distance from the box to *any* of its goal locations.
           (If multiple boxes share the same goal location, map them individually based on the goal facts.)
           - This distance estimates the minimal push-distance the box must be moved.
        b) Compute the minimum shortest path distance from the robot location to a location adjacent to the box
           (a location from where the robot can push the box towards its goal). This approximates the cost to 
           position the robot for pushing the box.
           - For each location adjacent to the box, get the robot-to-adjacent-location distance and take minimum.
    3. Sum the above distances across all boxes, weighting the robot-to-adjacent distance and box-to-goal distance
       equally or with a slight preference to box distances, since pushing is the main effort.
    4. The resulting sum is the heuristic estimate of the remaining actions needed.
    5. The heuristic value is zero if the state is a goal state (all boxes at goals).
    6. Since BFS distances are finite and all squares reachable in solvable states, the heuristic is always finite
       for solvable states.

    This heuristic disregards deadlocks, box collisions, and robot maneuvering complexities,
    so it is not guaranteed admissible but guides the search efficiently in practice.
    """

    def __init__(self, task):
        """
        Initialize the heuristic.
        """
        self.goals = task.goals
        self.static = task.static

        # Parse adjacency from static facts to build the adjacency graph:
        # adjacency is bidirectional according to the domain definition.
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                self.adjacency.setdefault(loc1, set()).add(loc2)
                self.adjacency.setdefault(loc2, set()).add(loc1)  # add reverse adjacency

        # Extract goal box locations: each box and its desired goal location(s)
        # Goals are facts like (at box loc)
        self.box_goal_locations = {}
        for goal_fact in self.goals:
            if match(goal_fact, "at", "*", "*"):
                parts = get_parts(goal_fact)
                box, goal_loc = parts[1], parts[2]
                self.box_goal_locations[box] = goal_loc

        # Locations: set of locations known from adjacency keys:
        self.locations = set(self.adjacency.keys())

        # Precompute shortest distances between all locations using BFS for efficiency
        # Store as a dict: distances[from_location][to_location] = distance
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = self._bfs_distances(loc)

    def _bfs_distances(self, start):
        """
        Compute shortest distances from start location to all reachable locations using BFS.

        Returns a dict mapping location -> distance (integer).
        """
        distances = {start: 0}
        frontier = [start]
        while frontier:
            current = frontier.pop(0)
            current_dist = distances[current]
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    frontier.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute a heuristic estimate of the number of actions remaining to solve the Sokoban puzzle
        in the state represented by 'node'.
        """
        state = node.state

        # Check if goal state: all boxes at their goal locations
        # Goal facts are (at box goal_loc) pairs; verify all are present in state
        if self.goals <= state:
            return 0

        # Extract current robot location and box locations from state
        robot_loc = None
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == "at-robot":
                robot_loc = parts[1]
            elif len(parts) == 3 and parts[0] == "at":
                box, loc = parts[1], parts[2]
                box_locations[box] = loc

        if robot_loc is None or not box_locations:
            # No robot location or no boxes - heuristic can't compute meaningfully
            return 0

        total_cost = 0.0
        # Heuristic parameters: weights for robot-to-box and box-to-goal distances
        # Pushing boxes dominates, so box_to_goal distance gets higher weight.
        weight_push = 1.2
        weight_robot_move = 1.0

        # For each box, estimate:
        #  - distance box must be pushed to goal
        #  - minimal robot distance to an adjacent location of the box (to push it)
        for box, box_loc in box_locations.items():
            # box's goal location
            goal_loc = self.box_goal_locations.get(box)
            if goal_loc is None:
                # No goal known for this box (should not happen), skip or ignore box
                continue

            # Distances from box to goal location
            dist_box_to_goal = self.distances.get(box_loc, {}).get(goal_loc)
            if dist_box_to_goal is None:
                # no path from box to goal found, set some high cost (e.g. 20)
                dist_box_to_goal = 20

            # Find all locations adjacent to box's current location (where robot can stand to push)
            adjacent_to_box = self.adjacency.get(box_loc, set())
            if not adjacent_to_box:
                # If no adjacent locations, robot can't push this box at all (probably deadlock)
                # Assign high cost so heuristic is large
                dist_robot_to_adjacent = 20
            else:
                # For each adjacent location, get robot to that location distance
                dist_robot_to_adjacent_candidates = []
                for adj_loc in adjacent_to_box:
                    dist = self.distances.get(robot_loc, {}).get(adj_loc)
                    if dist is not None:
                        dist_robot_to_adjacent_candidates.append(dist)
                if dist_robot_to_adjacent_candidates:
                    dist_robot_to_adjacent = min(dist_robot_to_adjacent_candidates)
                else:
                    # No reachable adjacent location from robot, assign high cost
                    dist_robot_to_adjacent = 20

            # Sum weighted distances for this box
            total_cost += weight_push * dist_box_to_goal + weight_robot_move * dist_robot_to_adjacent

        # Return heuristic as an integer (round up)
        return int(total_cost + 0.5)
