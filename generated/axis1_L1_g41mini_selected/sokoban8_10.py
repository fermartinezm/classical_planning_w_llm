from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2 ...)".
    - `args`: The expected pattern, wildcards `*` allowed.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (kyjn6f1j).

    Summary:
    This heuristic estimates the number of required actions to transport boxes 
    to their goal locations using a network of wlo7ezwc-type locations and rullurva-type boxes.
    The central mechanic involves "zdufuwne" being true at exactly one location (the one currently holding "zdufuwne"),
    and "qsw2k0xh" marking certain reachable locations. The actions either move the "zdufuwne" marker between neighboring locations,
    or push a box from one location to another along connected locations, updating both the zdufuwne position and box position.

    Assumptions:
    - There is exactly one box being manipulated (typical in example files).
    - The "zdufuwne" predicate marks the agent or "position".
    - The "qsw2k0xh" predicate marks walkable/accessible locations.
    - The "h80z8tvp" predicate encodes edges labeled by a constant in sia5d33v, defining a directed graph.
    - Boxes have to be moved from current location to goal location along the edges.
    - Moves and pushes are only possible along edges where "h80z8tvp" is true.
    - Heuristic does not need to be admissible, but should be informative to minimize node expansions.

    Heuristic Initialization:
    - Extract the static graph from the "h80z8tvp" facts: build mapping from each location to neighbors reachable via any label.
    - Extract all box objects from the goals (the facts with predicate "ldlu1rwo").
    - Extract goal locations of each box.
    - Store the static walkable locations from "qsw2k0xh" facts for reference.

    Step-By-Step Thinking for Computing Heuristic:
    1. Locate the current "zdufuwne" position (agent position).
    2. Locate current positions of all boxes (from "ldlu1rwo" facts).
    3. If the current state satisfies all goals, heuristic value is 0.
    4. Compute shortest path from current agent position to each box that is not at its goal.
    5. Compute shortest path from each box's current position to its goal location.
    6. Because to move a box, the agent must be adjacent and then push, the cost heuristic is roughly the sum over all boxes of:
       (distance(agent -> box) + distance(box -> goal)).
       The agent must travel to the box first, then push it to the goal.
    7. If multiple boxes, take the sum of these costs.
    8. Use breadth-first search over the undirected graph ignoring edge labels and directions (since directions may be complex),
       treating edges as bidirectional for heuristic purposes to maintain admissibility and simplicity.
    9. If no path is found for any required movement, return a high heuristic (in practice, assume finite since solvable).
    10. The heuristic estimates move+push actions count, as each move or push corresponds roughly to one action.

    Note: The heuristic may underestimate real cost due to ignoring edge directions and labels in BFS,
    but it is efficiently computable and informative.
    """

    def __init__(self, task):
        # Store goals = set of goal facts
        self.goals = task.goals

        # Extract boxes and their goal positions from goal facts with predicate "ldlu1rwo"
        self.goal_box_positions = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "ldlu1rwo" and len(parts) == 3:
                box = parts[1]
                location = parts[2]
                self.goal_box_positions[box] = location

        # Extract static graph edges from "h80z8tvp" facts ignoring directions and labels for heuristic purposes.
        # We create an undirected adjacency map of locations.
        self.adjacency = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "h80z8tvp" and len(parts) == 4:
                src = parts[1]
                dst = parts[2]
                # Add undirected edges for heuristic (to simplify the graph)
                self.adjacency.setdefault(src, set()).add(dst)
                self.adjacency.setdefault(dst, set()).add(src)

        # Also store known walkable locations as per "qsw2k0xh"
        self.walkable_locations = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "qsw2k0xh" and len(parts) == 2:
                self.walkable_locations.add(parts[1])

    def _bfs_shortest_path_length(self, start, goal):
        """Compute shortest undirected distance between start and goal in adjacency graph.
        Return integer path length or None if no path."""
        if start == goal:
            return 0
        if start not in self.adjacency or goal not in self.adjacency:
            return None
        visited = set([start])
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            for neighbor in self.adjacency.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given search node's state.
        """

        state = node.state

        # Check if goal reached -> heuristic=0
        if self.goals <= state:
            return 0

        # Extract current agent position: unique loc with `zdufuwne`
        agent_locs = [get_parts(fact)[1] for fact in state if fact.startswith("(zdufuwne ")]
        if not agent_locs:
            # No agent position, assign high cost
            return 1000
        agent_pos = agent_locs[0]

        # Extract box positions: every fact with predicate "ldlu1rwo"
        box_positions = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ldlu1rwo" and len(parts) == 3:
                box_positions[parts[1]] = parts[2]

        total_cost = 0

        # For each box with a goal, if not at goal, estimate cost:
        # cost = distance agent->box + distance box->goal
        # If distances can't be found, use a large number to represent unreachable.
        for box, goal_loc in self.goal_box_positions.items():
            current_loc = box_positions.get(box, None)
            if current_loc is None:
                # Box location unknown, assume large cost
                total_cost += 1000
                continue
            if current_loc == goal_loc:
                # Box at goal, no cost
                continue

            dist_agent_box = self._bfs_shortest_path_length(agent_pos, current_loc)
            dist_box_goal = self._bfs_shortest_path_length(current_loc, goal_loc)

            if dist_agent_box is None or dist_box_goal is None:
                # No path found, consider large cost
                total_cost += 1000
                continue

            # Each step corresponds roughly to 1 action: either move of agent, or push of box.
            # The agent must reach the box first, then push the box to goal.
            # We sum them as a heuristic.
            total_cost += dist_agent_box + dist_box_goal

        return total_cost
