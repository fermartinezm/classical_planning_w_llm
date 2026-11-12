from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    Example: "(zdufuwne loc_1_1)" -> ["zdufuwne", "loc_1_1"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(zdufuwne loc_1_1)".
    - `args`: The expected pattern with wildcards '*'.
    - Returns True if fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the 'secret_domain' domain.

    Summary:
    The heuristic estimates how many moves are required to reach the goal state,
    which requires placing specific 'rullurva'-typed objects (boxes) at goal 'wlo7ezwc'-typed locations.
    It models the problem as transporting boxes across locations in a grid-like graph,
    navigating via edges annotated with "pemn4" objects, with current "zdufuwne" true at one location (agent position).
    The heuristic computes shortest distances over the location graph and estimates the minimal number of moves
    needed to bring all boxes to their goal locations, considering that the agent can move and push boxes
    along edges if certain conditions hold.

    Assumptions:
    - The problem involves an agent located at exactly one location marked by (zdufuwne ?loc).
    - Boxes (objects of type rullurva) must be at their goal locations (ldlu1rwo box loc).
    - The agent can move the 'zdufuwne' marker from one location to another by using action x042b0fi.
    - Boxes are moved by the yi0kp1zg action, which requires the agent, the box location, and intermediate adjacent connected locations.
    - Connections between locations are given by the static predicate (h80z8tvp loc1 loc2 pemn4).
    - Locations are connected both ways with predicates reflecting edges in a graph.
    - The heuristic assumes shortest paths in the graph and minimal box moves ignoring blocking conditions.

    Heuristic Initialization:
    - Extract all static connections between locations (edges) from static facts (h80z8tvp).
      Build an adjacency dictionary: loc -> set of adjacent locs.
    - Extract all rullurva objects and track their goal locations from goal conditions (ldlu1rwo box loc).
    - From currently given facts, identify the agent location (where (zdufuwne ?loc) holds).
    - Identify current location of each box from facts (ldlu1rwo box loc).
    - Precompute shortest path distances between all relevant locations using BFS for efficient heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the goal conditions hold in the current state; if yes, heuristic = 0.
    2. Determine agent current location by finding fact (zdufuwne ?loc).
    3. For each box:
       - Find its current location.
       - Find its goal location.
       - Compute shortest path length from box current location to goal location.
    4. For each box, estimate moves as:
       - The minimal cost to push the box from current to goal location (distance).
       - Plus the cost for the agent to arrive at the box location to push it, considering agent movement from its current location or after pushing previous boxes.
    5. Because we want an efficient heuristic, approximate agent travel cost separately:
       - Estimate shortest agent distance from its current position to each box.
       - Sum of all box pushing distances (ignoring possible conflicts).
    6. The heuristic returns the sum of minimal agent moves plus box moving moves.
    7. This heuristic may overestimate or underestimate the real cost but guides search effectively.

    This heuristic focuses on graph shortest paths abstraction ignoring complex constraints like multiple box interference
    or the "sia5d33v"-typed objects labels on edges ignored for heuristic simplicity.
    """

    def __init__(self, task):
        """
        Initialize by extracting graph (adjacency) from static h80z8tvp facts,
        box goal locations from task goals, and store for runtime heuristic computation.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract adjacency from static facts (h80z8tvp)
        self.adjacency = dict()  # loc -> set of adjacent locs
        for fact in self.static:
            if match(fact, "h80z8tvp", "*", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                self.adjacency.setdefault(loc1, set()).add(loc2)

        # Extract box goal locations from goal facts (ldlu1rwo box loc)
        self.box_goals = dict()  # box -> goal_loc
        for fact in self.goals:
            if match(fact, "ldlu1rwo", "*", "*"):
                parts = get_parts(fact)
                box, goal_loc = parts[1], parts[2]
                self.box_goals[box] = goal_loc

        # Memorize set of all boxes (rullurva objects) that appear in goals or initial
        self.all_boxes = set(self.box_goals.keys())

    def __call__(self, node):
        """
        Compute the heuristic estimate for the current node state.

        1. If all goal facts are satisfied, return 0.
        2. Identify current agent location (where zdufuwne holds).
        3. Identify current location of each box (ldlu1rwo box loc).
        4. For each box, compute shortest distance from current to goal location.
        5. Compute shortest distance from agent to each box to estimate approach cost.
        6. Heuristic = sum of box moving distances + minimal agent travel cost to push boxes.

        Uses BFS for shortest path computations.
        """

        state = node.state

        # If goal reached, heuristic = 0
        if self.goals <= state:
            return 0

        # Extract current agent location (zdufuwne ?loc)
        agent_loc = None
        for fact in state:
            if fact.startswith("(zdufuwne "):
                parts = get_parts(fact)
                agent_loc = parts[1]
                break
        if agent_loc is None:
            # No agent location found, heuristic very high (or could raise)
            return 1000000

        # Extract current box locations from state (ldlu1rwo box loc)
        box_locations = dict()
        for fact in state:
            if match(fact, "ldlu1rwo", "*", "*"):
                parts = get_parts(fact)
                box, loc = parts[1], parts[2]
                box_locations[box] = loc

        # For boxes not currently located, assign None (possibly off map? treat as high cost)
        for box in self.all_boxes:
            if box not in box_locations:
                box_locations[box] = None

        # Helper: BFS to compute shortest path distances from start to all reachable nodes
        def bfs(start):
            visited = {start: 0}
            queue = [start]
            idx = 0
            while idx < len(queue):
                current = queue[idx]
                idx += 1
                dist = visited[current]
                for neighbor in self.adjacency.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append(neighbor)
            return visited

        # Precompute distances from agent location
        dist_from_agent = bfs(agent_loc)

        # For heuristic, sum distances for each box:
        # distance from box curr_loc to goal_loc (box pushing cost),
        # plus minimal agent distance to box start location (approach cost).
        total_push_dist = 0
        min_agent_to_box_dist = 1000000
        boxes_considered = 0

        for box, goal_loc in self.box_goals.items():
            box_loc = box_locations.get(box)
            if box_loc is None:
                # Missing box position: large penalty
                return 1000000
            if box_loc == goal_loc:
                # Box already at goal - no cost
                continue

            # Distances from box location for pushing cost
            dist_from_box = bfs(box_loc)
            box_push_dist = dist_from_box.get(goal_loc, None)
            if box_push_dist is None:
                # Goal unreachable for this box from current location, very large heuristic
                return 1000000

            total_push_dist += box_push_dist

            # Agent must reach box location first to push it
            agent_dist_to_box = dist_from_agent.get(box_loc, None)
            if agent_dist_to_box is None:
                # Box unreachable by agent
                return 1000000
            if agent_dist_to_box < min_agent_to_box_dist:
                min_agent_to_box_dist = agent_dist_to_box
            boxes_considered += 1

        # If no boxes to move, heuristic is zero here (should be caught above by goal check)
        if boxes_considered == 0:
            return 0

        # Heuristic estimate: agent minimum move to reach closest box + sum of moves to push boxes to goals
        heuristic_value = min_agent_to_box_dist + total_push_dist

        return heuristic_value
