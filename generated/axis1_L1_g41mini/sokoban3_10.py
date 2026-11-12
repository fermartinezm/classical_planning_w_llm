from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(zdufuwne loc_6_4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (kyjn6f1j).

    Summary:
    This heuristic estimates the remaining number of actions required to achieve the goal,
    which is to move boxes (type rullurva) onto specific locations (type wlo7ezwc).
    The main challenge is that boxes can only be moved by pushing them through adjacent 
    connected locations along edges defined by the predicate `h80z8tvp`, and the push 
    action x042b0fi moves the "zdufuwne" marker between locations.
    The heuristic estimates cost by calculating:
      1. The minimum number of pushes to move each box from its current location to its goal location
         along the undirected graph induced by `h80z8tvp`.
      2. Plus potentially one movement of the "zdufuwne" marker if it is not at the box's current location
         (the marker represents the current "agent" position required to push boxes).

    Assumptions:
    - Exactly one location has the "zdufuwne" fact true at any state, representing an agent or marker position.
    - Boxes must be pushed via connected locations defined by `h80z8tvp`.
    - Each push moves the box one step closer to its goal.
    - In some cases, the marker must move to the box before pushing.
    - Location connections represent a graph that is implicitly undirected—if `(h80z8tvp locA locB a6sst)` exists, the reverse direction also exists with some agent `a6sst'`.
    - The heuristic is not admissible (may overestimate) but aims to be informative and efficiently computable.
    - The heuristic value is 0 if and only if the goal condition (all boxes at their goal locations) is satisfied.

    Heuristic Initialization:
    - Extract from the static facts all edges between locations (wlo7ezwc) from `h80z8tvp`.
      These edges form a graph connecting locations.
    - Extract the goal locations of all boxes from the goal facts (`ldlu1rwo box loc`).
    - Prepare a graph adjacency dictionary for fast shortest-path lookups.
    - Record all push 'agents' (objects of type `sia5d33v`) for consideration but this is not further used.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current position of the "zdufuwne" marker, representing where the agent capable of pushing boxes stands.
    2. For each box:
       a. Find the current location of the box.
       b. Retrieve the box's goal location.
       c. Compute the shortest path length between current and goal locations on the graph of locations.
          This length corresponds to the minimal number of pushes needed (each push moves the box one edge).
       d. Also compute the shortest path length from the marker location to the box's current location, which estimates the number of moves the marker must make to be able to push.
          Marker moves do not correspond directly to actions but are roughly estimated to reflect accessibility cost.
       e. Sum these distances as cost components for that box.
    3. The overall heuristic is the sum of these estimated costs over all boxes.
    4. If no boxes remain or no distances are found, the heuristic returns 0 for goal states or at least 0 otherwise.
    5. The heuristic ensures a finite estimate or returns a large fallback cost if no path is found,
       preventing infinite values and ensuring the planner can always progress.

    This heuristic guides the planner to move the marker towards boxes and then push them along shortest paths
    toward their goal positions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the location graph, box goals, and push agents.

        Args:
            task (Task): The planning task with all static facts and goals.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract box goal locations from goals: facts of form (ldlu1rwo box loc)
        self.box_goals = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "ldlu1rwo":
                box, loc = parts[1], parts[2]
                self.box_goals[box] = loc

        # Extract undirected graph of locations from static facts (h80z8tvp)
        # The predicate is (h80z8tvp locA locB agent).
        # We consider edges between locA and locB as undirected for heuristic purposes.
        self.location_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "h80z8tvp":
                loc1, loc2 = parts[1], parts[2]
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # The set of all push agents (sia5d33v) is known but unused explicitly for the heuristic.

    def shortest_path_length(self, start, goal):
        """
        Compute shortest path length between start and goal locations in the location graph.
        Returns an integer length or None if no path exists.

        Uses a simple BFS because the graph is unweighted.

        Args:
            start (str): Starting location.
            goal (str): Goal location.

        Returns:
            int or None: Length of shortest path or None if unreachable.
        """
        if start == goal:
            return 0
        if start not in self.location_graph or goal not in self.location_graph:
            return None
        from collections import deque

        visited = set([start])
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.location_graph[current]:
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state node.

        Args:
            node (Node): The node representing the current search state with .state attribute.

        Returns:
            int: Estimated minimal number of actions required to reach the goal from this state.
        """
        state = node.state

        # If the goal is reached, heuristic = 0
        if self.goals <= state:
            return 0

        # Find the current "zdufuwne" marker location (exactly one should exist)
        marker_loc = None
        for fact in state:
            if match(fact, "zdufuwne", "*"):
                parts = get_parts(fact)
                marker_loc = parts[1]
                break

        if marker_loc is None:
            # Defensive fallback if no marker found: heuristic can't proceed meaningfully
            # Return 0 so search progresses (could be large but that risks pruning)
            return 0

        # Find current locations of all boxes (ldlu1rwo box loc)
        box_locations = {}
        for fact in state:
            if match(fact, "ldlu1rwo", "*", "*"):
                parts = get_parts(fact)
                box, loc = parts[1], parts[2]
                box_locations[box] = loc

        total_cost = 0

        # For each box with a goal, estimate cost as marker-to-box + box-to-goal distances
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locations.get(box)
            if current_loc is None:
                # Box location unknown in current state - pessimistic estimate
                # Assign a large cost (e.g. 50)
                total_cost += 50
                continue

            # Distance for marker to reach box location
            dist_marker_to_box = self.shortest_path_length(marker_loc, current_loc)
            if dist_marker_to_box is None:
                # No path from marker to box - large cost
                dist_marker_to_box = 50

            # Distance for box to reach goal
            dist_box_to_goal = self.shortest_path_length(current_loc, goal_loc)
            if dist_box_to_goal is None:
                # No path from box to goal - large cost
                dist_box_to_goal = 50

            # Each push/action roughly costs 1 step
            # Assume marker must first reach box, then push it along shortest path
            # The marker moves are not always explicit actions, but this provides a heuristic signal.
            # We sum them to reflect required efforts.
            cost = dist_marker_to_box + dist_box_to_goal
            total_cost += cost

        # Defensive fallback to ensure heuristic >= 0
        return max(int(total_cost), 0)
