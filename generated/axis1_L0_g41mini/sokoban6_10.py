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

    Summary:
        This heuristic estimates the number of actions required to push all boxes from their current
        locations to their respective goal locations. It guides a greedy best-first search
        by computing an efficient, non-admissible but informative cost estimate.

    Assumptions:
        - Each box must be pushed exactly to its goal location.
        - Moving the robot without pushing is costlier and less significant than pushing boxes.
        - The robot must navigate to positions adjacent to boxes in order to push them.
        - The heuristic ignores obstacles and possible deadlocks, focusing on shortest distances in the location graph.

    Heuristic Initialization:
        - Extract and store all box goal locations from the goal conditions.
        - Extract the adjacency graph of locations from the static facts for shortest path calculations.
        - Prepare a fast lookup structure for adjacency.
        - Precompute all-pairs shortest distances between locations to speed up heuristic evaluations.

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse current state facts to identify:
            - Robot position.
            - Box positions.
            - Which locations are clear.
        2. For each box:
            - Compute the minimal distance from its current location to its goal location.
        3. For the robot:
            - Compute minimal distance to reach a location adjacent to the box in the direction from which the box can be pushed toward its goal.
        4. Sum over all boxes:
            - The minimal cost estimate is roughly the sum of:
              a) The minimum push distance required for the box (number of pushes).
              b) The minimum cost for the robot to reach the pushing position for that box (robot moves).
        5. Return the accumulated sum as the heuristic value.
    
        Notes:
          - Because the domain is grid-like, distances are computed via BFS on adjacency graph.
          - The heuristic does not consider the exact push sequence but approximates effort by these distances.
          - The heuristic is zero if and only if all boxes are at their goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by:
        - Extracting box goal locations from the goal facts.
        - Building an adjacency graph from static facts.
        - Precomputing shortest distances between all locations for efficiency.
        """
        self.goals = task.goals

        # Extract box goal positions: {box -> goal_location}
        self.box_goal_locations = dict()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                self.box_goal_locations[box] = loc

        # Build adjacency graph: dict location -> set of adjacent locations
        self.adjacency = {}
        for fact in task.static:
            if match(fact, "adjacent", "*", "*", "*"):
                parts = get_parts(fact)
                loc1 = parts[1]
                loc2 = parts[2]
                self.adjacency.setdefault(loc1, set()).add(loc2)
                # Note: adjacency is not necessarily symmetrical, so do not add inverse unless given

        # Prepare set of all locations for distance computations
        self.locations = set(self.adjacency.keys())
        # Include all adjacency targets as well (might not all be keys)
        for adjs in self.adjacency.values():
            self.locations.update(adjs)

        # Precompute shortest distances between all locations via BFS (on directed edges)
        # Store as a dict: distances[loc1][loc2] = int distance
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = self._bfs_distances(loc)

    def _bfs_distances(self, start):
        """
        Perform BFS in the directed graph starting from 'start' location.
        Returns a dict location->distance from 'start'.
        """
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for nbr in self.adjacency.get(current, ()):
                if nbr not in distances:
                    distances[nbr] = distances[current] + 1
                    queue.append(nbr)
        return distances

    def __call__(self, node):
        """
        Estimate the minimal number of actions to reach the goal state from the current node.

        Returns:
            Integer heuristic estimate (0 if goal reached).
        """
        state = node.state

        # Quick goal test: if all box goals satisfied, heuristic is 0
        goal_satisfied = True
        for box, goal_loc in self.box_goal_locations.items():
            if f"(at {box} {goal_loc})" not in state:
                goal_satisfied = False
                break
        if goal_satisfied:
            return 0

        # Extract robot location
        robot_loc = None
        for fact in state:
            if fact.startswith("(at-robot "):
                robot_loc = get_parts(fact)[1]
                break

        # Extract box locations: dict box -> location
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                loc = parts[2]
                # Only boxes considered here (boxes are keys of box_goal_locations)
                if obj in self.box_goal_locations:
                    box_locations[obj] = loc

        # Extract clear locations for push destination verification
        clear_locations = set()
        for fact in state:
            if fact.startswith("(clear "):
                cloc = get_parts(fact)[1]
                clear_locations.add(cloc)

        heuristic_value = 0

        # For each box, estimate costs:
        for box, box_loc in box_locations.items():
            goal_loc = self.box_goal_locations[box]

            # Compute push cost: minimal distance from box_loc to goal_loc via adjacency, assuming pushing boxes counts as one push per location step
            # If no path to goal_loc, assign large cost to reflect difficulty
            dist_box_to_goal = self.distances.get(box_loc, {}).get(goal_loc, None)
            if dist_box_to_goal is None:
                # Unreachable goal for box -> large heuristic value
                dist_box_to_goal = 1000  # large penalty, adjust as needed

            # Best estimated pushes needed is dist_box_to_goal

            # Next, estimate robot cost to reach a suitable location adjacent to the box for pushing it towards goal.
            # To push a box from box_loc to an adjacent loc' (towards goal), robot must be adjacent at loc_push.
            #
            # We try all adjacents adj_loc of box_loc that lead closer to the goal_loc, i.e.
            # loc_push --push--> box_loc --push--> adj_loc towards goal.
            # For Sokoban pushing, robot must stand at loc_push (adjacent to box_loc) opposite direction of pushing.
            # We approximate by:
            #   - Find neighbors adj_loc of box_loc that satisfy dist(adj_loc, goal_loc) < dist(box_loc, goal_loc)
            #   - For each such neighbor loc_push (adjacent to box_loc), compute robot distance to loc_push.
            #   - Take minimal robot travel distance to these positions.

            min_robot_to_push_pos = None
            dist_box_loc_to_goal = dist_box_to_goal
            candidates_push_positions = []

            # Explore adjacency from box_loc
            neighbors = self.adjacency.get(box_loc, set())

            for adj_loc in neighbors:
                dist_adj_to_goal = self.distances.get(adj_loc, {}).get(goal_loc, None)
                if dist_adj_to_goal is None:
                    continue
                # If moving box from box_loc to adj_loc reduces distance to goal, adj_loc is closer to goal.
                if dist_adj_to_goal < dist_box_loc_to_goal:
                    # Robot must stand at a location adjacent to box_loc but not adj_loc (robot stands "behind" box to push it into adj_loc)
                    # Robot pushes box from box_loc to adj_loc by standing on the location opposite adj_loc side relative to box_loc.
                    # We should find locations adjacent to box_loc that are not adj_loc.
                    # Since adjacency is directed, directions are implicit.

                    # We look for robot positions that are adjacent to box_loc and from which pushing in that direction moves box towards adj_loc.
                    # Because we only have adjacency facts, we find robot_positions satisfying adjacent(robot_pos, box_loc) with direction opposite to (box_loc->adj_loc).

                    # Find all locations that have an adjacency to box_loc; these are candidate robot positions for pushing box from that direction.
                    # Because adjacency is directed, we find locations that have adjacency to box_loc.

                    robot_push_positions = [
                        loc for loc, adjs in self.adjacency.items() if box_loc in adjs and loc != adj_loc
                    ]

                    candidates_push_positions.extend(robot_push_positions)

            # If no improving adjacency found (e.g. box already at goal), robot need not push box further
            if not candidates_push_positions:
                # Robot does not need to push this box; only pushes count is zero.
                push_cost = dist_box_to_goal  # 0 if at goal, >0 if no path (penalized)
                robot_cost = 0
            else:
                # Compute robot distance to each candidate push position, pick minimal
                min_robot_dist = None
                for pos in candidates_push_positions:
                    dist_robot_to_pos = self.distances.get(robot_loc, {}).get(pos, None)
                    if dist_robot_to_pos is not None:
                        if (min_robot_dist is None) or (dist_robot_to_pos < min_robot_dist):
                            min_robot_dist = dist_robot_to_pos

                if min_robot_dist is None:
                    # Robot cannot reach any position to push this box effectively.
                    min_robot_dist = 1000  # Large penalty

                push_cost = dist_box_to_goal
                robot_cost = min_robot_dist

            # Sum estimated cost for this box: pushes + robot moves to pushing position
            # We assume each push and robot move counts as one action
            heuristic_value += push_cost + robot_cost

        return heuristic_value
