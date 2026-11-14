from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(ldlu1rwo box loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (domain kyjn6f1j).

    Summary:
    This heuristic estimates the number of actions required to achieve the goal fact(s) of the form
    (ldlu1rwo box location) in the secret_domain. The domain involves moving a fluent predicate
    (zdufuwne) "token" among locations (wlo7ezwc), and transporting boxes (rullurva) between locations.
    The heuristic focuses on the position of each box relative to its goal location, and the current
    location of the "token" (the fluent zdufuwne holds true for exactly one location at a time).
    It uses shortest path distances on a graph defined by the static predicate h80z8tvp (connectivity edges).

    Assumptions:
    - Exactly one location at a time has the predicate (zdufuwne).
    - Each box must be placed at its goal location (ldlu1rwo predicate).
    - Movement of the "token" or the boxes happens along edges h80z8tvp which connect locations with some label.
    - The action x042b0fi moves the token from one location to another.
    - The action yi0kp1zg moves a box from one location to another, but requires the token at the intermediate location and some ldlu1rwo relation.
    - The domain can be thought of as a "token" carrier that must carry boxes to their goal locations by hopping along edges.

    Heuristic Initialization:
    - Extract from static facts all edges between locations based on h80z8tvp predicates. The edges are directed:
      from location A to B if (h80z8tvp A B c) is given.
    - Precompute adjacency mapping for all locations (wlo7ezwc).
    - Extract the goal locations for each box from the goal facts (ldlu1rwo box loc).
    - Extract all boxes and all locations from the relevant facts.
    - This prepares data for fast heuristic calculation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current location of the token (unique loc such that zdufuwne loc holds).
    2. For each box:
       - Identify the current location of the box (ldlu1rwo box loc is unique in current state).
       - Identify the goal location for this box.
       - Compute the shortest path distance in the directed graph from the box's current location to its goal location.
         If no path exists, consider a large penalty cost, as box cannot be directly moved there.
    3. Also compute the shortest path distance from the token's current location to the box's current location.
       The token must travel to the box before it can be moved.
    4. The cost to move each box at least includes:
       - Moving token to box location.
       - Moving the box from its current location to its goal along shortest path edges.
       Each move of box and each move of token corresponds to actions in the domain.
    5. Sum these costs over all boxes.
    6. To avoid underestimating, we consider that moving the token and moving the box cannot be completely overlapped, so we sum these distances.
    7. Finally, since each movement step corresponds roughly to one action, this sum estimates the number of actions needed.
    8. If all boxes are at their goal locations and the token is anywhere, heuristic is 0 (goal reached).
    9. The heuristic is not admissible but is informative and efficiently computable.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract edges from static facts to build the directed graph between locations.
        # Graph: {location: set(of adjacent locations reachable from this location)}
        self.graph = {}

        for fact in self.static:
            # Match h80z8tvp ?loc1 ?loc2 ?label
            if match(fact, "h80z8tvp", "*", "*", "*"):
                parts = get_parts(fact)
                loc_from = parts[1]
                loc_to = parts[2]
                if loc_from not in self.graph:
                    self.graph[loc_from] = set()
                self.graph[loc_from].add(loc_to)

        # Extract all boxes and their goal locations from the goal facts.
        # Goal facts are of form: (ldlu1rwo box location)
        self.box_goals = {}
        for fact in self.goals:
            if match(fact, "ldlu1rwo", "*", "*"):
                parts = get_parts(fact)
                box = parts[1]
                loc = parts[2]
                self.box_goals[box] = loc

        # Extract all locations known in the graph keys and values, to get the locations set.
        self.locations = set(self.graph.keys())
        for adjacents in self.graph.values():
            self.locations.update(adjacents)

        # Large constant cost if no path exists
        self.no_path_cost = 1000

    def shortest_path_length(self, start, goal):
        """Compute shortest path length from start to goal in the directed graph using BFS.
        Returns the number of edges if path exists, else None."""
        if start == goal:
            return 0
        if start not in self.graph:
            return None
        from collections import deque

        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.graph.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # 1) Find the token location (unique location with (zdufuwne loc))
        token_loc = None
        for fact in state:
            if match(fact, "zdufuwne", "*"):
                token_loc = get_parts(fact)[1]
                break
        if token_loc is None:
            # If no token location found, heuristic returns high cost.
            return self.no_path_cost * len(self.box_goals)

        # 2) For each box, find current location from state:
        # The box location is identified by fact (ldlu1rwo box loc).
        box_current_locations = {}
        for fact in state:
            if match(fact, "ldlu1rwo", "*", "*"):
                parts = get_parts(fact)
                box_current_locations[parts[1]] = parts[2]

        # 3) If all boxes are at their goal locations, heuristic == 0 (goal reached)
        all_at_goal = True
        for box, goal_loc in self.box_goals.items():
            current_loc = box_current_locations.get(box)
            if current_loc != goal_loc:
                all_at_goal = False
                break

        if all_at_goal:
            return 0

        # 4) Sum heuristic costs for each box:
        # Cost for each box = distance(token -> box_loc) + distance(box_loc -> goal_loc)
        # If no path exists, add large penalty.

        total_cost = 0

        for box, goal_loc in self.box_goals.items():
            current_loc = box_current_locations.get(box)
            if current_loc is None:
                # Box location unknown, penalize
                total_cost += self.no_path_cost
                continue

            if current_loc == goal_loc:
                # No cost for this box
                continue

            dist_token_to_box = self.shortest_path_length(token_loc, current_loc)
            if dist_token_to_box is None:
                dist_token_to_box = self.no_path_cost

            dist_box_to_goal = self.shortest_path_length(current_loc, goal_loc)
            if dist_box_to_goal is None:
                dist_box_to_goal = self.no_path_cost

            total_cost += dist_token_to_box + dist_box_to_goal

        return total_cost
