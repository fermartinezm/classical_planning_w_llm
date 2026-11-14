from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(adjacent box1 location6)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the 'secret_domain' (Sokoban-like domain).

    Summary:
    ----------
    The heuristic estimates the approximate number of push operations required 
    to move each box from its current position to a state where it satisfies the 
    goal condition `(adjacent-2 box)`. Since the domain encodes the goal as each box 
    being 'adjacent-2', the heuristic tries to count how many boxes are not yet 
    in their goal adjacent-2 state, and estimates the minimal number of pushes 
    (and associated moves) required to achieve that.

    Assumptions:
    ------------
    - The goal requires all boxes to satisfy `(adjacent-2 box)`.
    - A box currently at `(adjacent-2 box)` needs no further action.
    - Boxes not at `(adjacent-2 box)` must be pushed at least once.
    - The heuristic relies on static facts describing `(at-robot box location)` positions 
      and `(clear location1 location2)` connections to estimate robot moves.
    - The heuristic is not admissible: it underestimates robot movements but captures 
      necessary push actions well.
    - All boxes start somewhere adjacent to locations, and pushing moves a box from
      `(at ?box)` to `(adjacent-2 ?box)` (or closer to it).
    - The heuristic ignores capturing the exact robot position paths but counts 
      minimal distances between robot and box locations using the clear relation.

    Heuristic Initialization:
    ------------------------
    - Extract all boxes and locations appearing in static facts.
    - Build adjacency graph of locations using `(clear loc1 loc2)`.
    - Extract static map of `(at-robot box location)` facts to get box starting locations.
    - Extract all goal boxes that must satisfy `(adjacent-2 box)`.
    - Prepare helper structures to quickly query robot adjacency (static), robot position,
      and box states in any given state.

    Step-By-Step Thinking for Computing Heuristic:
    ---------------------------------------------
    1. Identify boxes required to reach `(adjacent-2 box)` in the goal.
    2. For the current state:
       - Count how many of these boxes already satisfy `(adjacent-2 box)`; these need no pushing.
       - For each box not in `(adjacent-2 box)`, consider:
         a) The number of pushes needed to get it into adjacent-2 state (minimum 1 push).
         b) The robot must also move adjacent to the box to push it.
            We estimate minimal robot moves by computing the shortest path from the 
            robot's current location to one of the box's robot adjacency locations 
            `(at-robot box location)` using the pre-extracted static graph of clear edges.
    3. Sum for all boxes not yet adjacent-2:
       - Estimated cost = minimal robot moves to box adjacency + 1 push action.
    4. Take the maximum over all such box costs, as robot moves can be amortized when pushing boxes adjacently.
       Alternatively, sum the cost with a small penalty for multiple boxes (e.g., sum pushes only, plus 
       approximate robot move cost for closest box).
    5. Return that heuristic value, ensuring zero if all boxes are already in `(adjacent-2 box)`.

    The heuristic balances accuracy and efficiency:
    - It does not perform full shortest path computations repeatedly but uses BFS over static location graph.
    - It uses static facts to localize pushes.
    - It avoids expensive computations and provides a finite estimate guiding the search effectively.
    """

    def __init__(self, task):
        """
        Initialize heuristic by extracting:
         - Boxes that appear in `(adjacent-2 box)` goal conditions.
         - Static graph of locations connected by `(clear loc1 loc2)`.
         - Map from boxes to their adjacency locations `(at-robot box location)` (static).
         - Set of all locations for BFS computations.

        Parameters:
            task: The planning task containing initial state, goals, static facts, etc.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract boxes that must satisfy (adjacent-2 box) in goal.
        self.goal_boxes = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "adjacent-2" and len(parts) == 2:
                self.goal_boxes.add(parts[1])

        # Extract all locations from clear facts and at-robot facts (static).
        self.locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "clear" and len(parts) == 3:
                self.locations.add(parts[1])
                self.locations.add(parts[2])
            elif parts[0] == "at-robot" and len(parts) == 3:
                self.locations.add(parts[2])  # locations adjacent to boxes

        # Build graph of locations based on clear relation for robot moves.
        self.location_graph = {loc: set() for loc in self.locations}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "clear" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.location_graph[loc1].add(loc2)
                # Assuming symmetric adjacency since Sokoban-like domains usually symmetric
                self.location_graph[loc2].add(loc1)

        # Map each box to adjacency locations where robot can push it (from static facts).
        self.box_to_adj_locations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                self.box_to_adj_locations.setdefault(box, set()).add(loc)

    def __call__(self, node):
        """
        Estimate number of actions to reach a goal state from the current node.

        Strategy:
         - Return 0 if all goal boxes satisfy (adjacent-2 box).
         - Otherwise, for each box not adjacent-2:
            * Find the robot current location (state fact: (at-robot-2 location)).
            * Find shortest distance from robot location to one adjacency location of that box.
              Use BFS on static graph of locations.
            * Estimate cost as (distance + 1 push).
         - Sum 1 push per box not yet adjacent-2 plus minimal robot travel cost to closest box.
           This balances between overestimating robot movement and acknowledging pushes needed.

        Returns:
            int heuristic estimate (finite, >= 0).
        """
        state = node.state

        # Quick helper to find robot location (at-robot-2 ?loc)
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot-2" and len(parts) == 2:
                robot_location = parts[1]
                break
        if robot_location is None:
            # No robot location known, heuristic fallback: count pushes only
            robot_location = None

        # Check which boxes satisfy the goal predicate (adjacent-2 box)
        adjacent2_boxes = {get_parts(fact)[1] for fact in state if match(fact, "adjacent-2", "*")}

        # Boxes that still need pushing
        boxes_to_push = self.goal_boxes - adjacent2_boxes

        # If all boxes satisfy (adjacent-2), heuristic is zero
        if not boxes_to_push:
            return 0

        # BFS helper to find shortest distance between two locations
        def bfs_shortest_distance(start, targets):
            if start is None:
                # No known start location
                return float('inf')
            if start in targets:
                return 0
            visited = set([start])
            queue = [(start, 0)]
            while queue:
                current_loc, dist = queue.pop(0)
                for neighbor in self.location_graph.get(current_loc, []):
                    if neighbor not in visited:
                        if neighbor in targets:
                            return dist + 1
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            return float('inf')  # no path found

        # For each box, compute robot move cost + 1 push
        total_pushes = len(boxes_to_push)
        min_robot_move_cost = float('inf')
        for box in boxes_to_push:
            # Get adjacency locations where robot can push this box
            adj_locs = self.box_to_adj_locations.get(box, set())
            if not adj_locs:
                # If no adjacency locations known for this box, assume robot must move far
                box_robot_move = 1  # minimal fallback
            else:
                box_robot_move = bfs_shortest_distance(robot_location, adj_locs)
                if box_robot_move == float('inf'):
                    # No path found, minimum fallback
                    box_robot_move = 1

            # Update minimum robot move cost among boxes to push
            if box_robot_move < min_robot_move_cost:
                min_robot_move_cost = box_robot_move

        # Heuristic estimate:
        # Sum of pushes needed + minimal robot moves to closest box
        heuristic_value = total_pushes + min_robot_move_cost

        # As heuristic must be finite and non-negative:
        if heuristic_value == float('inf'):
            # Fallback to number of pushes at least
            heuristic_value = total_pushes

        return heuristic_value
