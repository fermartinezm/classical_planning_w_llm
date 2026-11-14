from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The fact as a string, e.g. "(predicate arg1 arg2)".
    - `args`: Pattern elements to match, supporting wildcards "*".
    - Returns True if the pattern matches, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (kyjn6f1j).

    Summary
    -------
    This heuristic estimates the minimal number of actions required to reach the goal state,
    where the goal is that certain objects of type `rullurva` are located at specific locations
    (facts of form (ldlu1rwo ?rullurva ?wlo7ezwc)). The domain models moving a "marked" location 
    (fact (zdufuwne ?loc)) and transferring that marker along paths defined by the predicate (h80z8tvp),
    while also transporting rullurva-typed objects using the second action. Transitions depend on 
    predicates like (qsw2k0xh) that mark valid locations.

    The heuristic uses shortest-path estimation on the static graph of locations joined by "h80z8tvp" edges.
    It accounts for the position of the "marker" (zdufuwne) and the locations of the rullurva objects to deliver,
    estimating how many moves and transports are needed. It is not guaranteed admissible but aims for good guidance.

    Assumptions
    -----------
    - The static facts (h80z8tvp) represent a bidirectional graph of "reachable" paths between locations.
    - The unique "marker" location (zdufuwne) behaves like a player or agent that moves.
    - Objects of type rullurva represent boxes that must be moved to their goal locations.
    - The preconditions of actions rely on matching transport edges carrying marker or moving a box together.
    - The goal is always a set of facts (ldlu1rwo box location) specifying desired box placements.
    - Locations have types that restrict usage and valid connections.
    - Moving the marker along edges costs one action (x042b0fi action).
    - Transporting a box to another location with the marker costs one action (yi0kp1zg action).

    Heuristic Initialization
    ------------------------
    - Extract all rullurva objects to track box goals.
    - Extract the goal mapping from rullurva objects to their goal locations.
    - Build a graph representing static connectivity between locations from h80z8tvp facts.
    - Cache validity facts (qsw2k0xh) per location for reachability checks.
    - Store the available costs: one move per edge, one transport per box move.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if all goal (ldlu1rwo) facts hold in the current state; if yes, heuristic is 0.
    2. Identify current marker location from (zdufuwne ?loc).
    3. Identify current locations of all boxes (rullurva objects) from (ldlu1rwo).
    4. For each box whose current location differs from goal:
       - Estimate cost to move marker to box's location (shortest path).
       - Estimate cost to transport box from its current location to goal location 
         (shortest path over reachable edges).
       - Sum these costs (one move per marker move edge, one transport per edge).
    5. Since marker must move between boxes, add estimated cost of moving the marker from one box to the next.
       A simple approximation: sum of individual box costs plus minimal marker travel 
       between boxes and final position at last delivery.
    6. Return total estimated cost as heuristic value, ensuring finite value for solvable states.

    Implementation notes
    --------------------
    - Use breadth-first search (BFS) on the static graph for shortest paths.
    - Consider edges undirected by adding both directions from h80z8tvp facts.
    - Count actions conservatively: 
      - marker move = 1 action per edge,
      - transport of box = 1 action per edge traversed (via yi0kp1zg).
    - Ignore complex ordering optimizations, relying on sum-of-distances approximation.
    """

    def __init__(self, task):
        """Initialize the heuristic with graph and goal information from the task."""
        self.goals = task.goals
        self.static = task.static

        # Extract all locations of type wlo7ezwc, represented in static or facts from h80z8tvp.
        # Build graph edges from h80z8tvp facts (bidirectional).
        self.graph = {}
        for fact in self.static:
            if match(fact, "h80z8tvp", "*", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                # Add edges both ways (undirected assumption)
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1)

        # Extract goal box locations from goals (ldlu1rwo box loc)
        self.box_goals = {}
        for goal_fact in self.goals:
            if match(goal_fact, "ldlu1rwo", "*", "*"):
                parts = get_parts(goal_fact)
                box, loc = parts[1], parts[2]
                self.box_goals[box] = loc

        # Store all boxes to move
        self.boxes = set(self.box_goals.keys())

        # Store valid (qsw2k0xh) locations for quick connectivity and verification
        self.valid_locations = set()
        for fact in task.initial_state | task.static:
            if match(fact, "qsw2k0xh", "*"):
                loc = get_parts(fact)[1]
                self.valid_locations.add(loc)

    def shortest_path_length(self, start, goal):
        """Compute shortest path length in the graph from start to goal, or None if unreachable."""
        if start == goal:
            return 0
        if start not in self.graph or goal not in self.graph:
            return None
        from collections import deque
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.graph.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor in self.valid_locations and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None

    def __call__(self, node):
        """
        Estimate the number of actions to reach the goal from the current state.

        Steps:
        1. Check if goals achieved → return 0.
        2. Find current marker location.
        3. Find current box locations.
        4. For each box not at goal:
           - Cost to move marker to box location.
           - Cost to move box to goal.
        5. Sum costs, add marker moves between boxes for delivery sequence approximation.
        """
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Extract marker location (zdufuwne loc)
        marker_loc = None
        for fact in state:
            if match(fact, "zdufuwne", "*"):
                marker_loc = get_parts(fact)[1]
                break
        if marker_loc is None:
            # Defensive fallback: no marker location found → large cost
            return 100000

        # Extract current box locations: ldlu1rwo box loc
        box_locations = {}
        for fact in state:
            if match(fact, "ldlu1rwo", "*", "*"):
                parts = get_parts(fact)
                box, loc = parts[1], parts[2]
                if box in self.boxes:
                    box_locations[box] = loc

        # Boxes that are missing location info (possibly all still in init or removed),
        # treat as far away for safety: assign None so heuristic is high but finite
        for box in self.boxes:
            if box not in box_locations:
                box_locations[box] = None

        # Compute costs:
        # Move marker to box location + transport box to goal location
        # Transporting box costs one action per edge
        # Moving marker costs one action per edge

        total_cost = 0
        current_marker_pos = marker_loc

        # We'll visit boxes in some order (greedy nearest box from current marker pos)
        boxes_to_deliver = [b for b in self.boxes if box_locations[b] != self.box_goals[b]]

        # If no boxes need delivery (all at goal), heuristic 0 (already checked goals)
        if not boxes_to_deliver:
            return 0

        # Simple nearest neighbor heuristic for visiting boxes
        undelivered = set(boxes_to_deliver)

        while undelivered:
            # Find closest box (consider path length marker->box loc)
            best_box = None
            best_dist = None
            best_box_transport_dist = None

            for box in undelivered:
                box_loc = box_locations[box]
                goal_loc = self.box_goals[box]

                if box_loc is None:
                    # Unknown location means large cost to deliver, skip for now
                    continue

                dist_to_box = self.shortest_path_length(current_marker_pos, box_loc)
                dist_box_to_goal = self.shortest_path_length(box_loc, goal_loc)
                if dist_to_box is None or dist_box_to_goal is None:
                    # Unreachable places increase heuristic drastically
                    continue

                # Cost estimate:
                # move marker to box: dist_to_box actions (x042b0fi)
                # move box to goal: dist_box_to_goal actions (yi0kp1zg)
                total_box_cost = dist_to_box + dist_box_to_goal

                if best_dist is None or total_box_cost < best_dist:
                    best_dist = total_box_cost
                    best_box = box
                    best_box_transport_dist = dist_box_to_goal

            # If no reachable box found, heuristic high but finite
            if best_box is None:
                return 100000

            # Add cost for marker moving to box location + transporting box to goal
            total_cost += best_dist

            # Update marker position to box's goal location (after transport)
            current_marker_pos = self.box_goals[best_box]
            undelivered.remove(best_box)

        return total_cost
