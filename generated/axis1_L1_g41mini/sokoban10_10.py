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
    Domain-dependent heuristic for the secret_domain.

    Summary:
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. The domain models moving a box on a grid-like
    topology where locations are connected with labeled edges. There are two
    main predicates describing connectivity: `h80z8tvp` encodes directed edges
    between locations with a label (from wlo7ezwc to wlo7ezwc using a sia5d33v).
    A box (rullurva) is located at some location (`ldlu1rwo`), and locations may
    be flagged as accessible (`qsw2k0xh`) or occupied by a box (`zdufuwne`).
    The key actions either move the "occupied" flag from one location to a connected one,
    or move the box along a path updating its location and also the status predicates.

    Assumptions:
    - The goal requires boxes to be at target locations (`ldlu1rwo` predicates).
    - A box can only move between locations connected by a chain of `h80z8tvp` edges
      with matching labels.
    - Only one box is moved at a time, and exactly one location is "occupied" at a time.
    - The heuristic ignores obstacles and other boxes for efficiency.
    - Movement cost is approximated by shortest path distance along edges ignoring labels,
      assuming that paths exist due to the static edges.
    - The heuristic does not need to be admissible but aims to closely estimate
      the number of moves required.

    Heuristic Initialization:
    - Extract all locations (`wlo7ezwc` type) from static facts.
    - Extract all static directed edges between locations from `(h80z8tvp from to label)` facts.
    - Build an undirected connectivity graph of locations (ignoring label directionality and label)
        since actions allow moving the "occupied" marker and box along edges.
    - Extract all goal box location assignments from goal facts `(ldlu1rwo box loc)`.
    - Extract all box objects.
    - Precompute shortest path distances between all locations using BFS for efficiency.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the current state is a goal state, return 0.
    2. Find the current location `pos` of the "occupied" spot (`zdufuwne ?loc`).
    3. For each box:
       - Determine the box's current location by `(ldlu1rwo box current_loc)`.
       - Determine the box's goal location from the extracted goals.
    4. Estimate the minimal cost to transport the box from current_loc to goal_loc.
       The cost includes:
       - Moving the "occupied" marker to a box's current location if needed,
       - Moving the box to the next location along the shortest path toward the goal,
       - Since each box move corresponds to at least one action and moving the
         occupied flag counts as action as well.
    5. Sum heuristics for all boxes, considering that only one box can be moved at once.
       Here we approximate cost as:
         distance from occupied_pos to box current location
         + distance from box current location to box goal location,
       for each box, and take the minimum such cost to select which box to move next.
    6. Add the number of boxes remaining to be moved (each box requires at least one move).
    7. Return the sum as the estimated heuristic.

    The heuristic is efficiently computable using precomputed shortest paths and
    works well to minimize expanded states by prioritizing moving boxes closer.
    """

    def __init__(self, task):
        self.goals = task.goals  # Goal facts (must be met)
        static_facts = task.static  # Static relations that do not change

        # Extract all locations of type wlo7ezwc from static facts (from edges h80z8tvp)
        locations = set()
        edges = []
        for fact in static_facts:
            if match(fact, "h80z8tvp", "*", "*", "*"):
                parts = get_parts(fact)
                _pred, from_loc, to_loc, _label = parts
                locations.add(from_loc)
                locations.add(to_loc)
                edges.append((from_loc, to_loc))

        self.locations = list(locations)

        # Build adjacency list graph of locations (undirected) ignoring label direction for heuristic
        self.adj = {loc: set() for loc in self.locations}
        for (a, b) in edges:
            self.adj[a].add(b)
            self.adj[b].add(a)

        # Extract goal locations per box from goal facts: (ldlu1rwo box location)
        self.goal_box_locations = {}
        for fact in self.goals:
            if match(fact, "ldlu1rwo", "*", "*"):
                parts = get_parts(fact)
                _, box, loc = parts
                self.goal_box_locations[box] = loc

        # Extract box objects from static facts or from goals (rullurva objects appear here)
        boxes = set()
        for fact in self.goals:
            if match(fact, "ldlu1rwo", "*", "*"):
                box = get_parts(fact)[1]
                boxes.add(box)
        self.boxes = boxes

        # Precompute all pairs shortest distances between locations for quick heuristic computation.
        # Use BFS for shortest path lengths.
        self.dist = {}
        for loc in self.locations:
            self.dist[loc] = self._bfs_distances(loc)

    def _bfs_distances(self, start):
        """Compute shortest distances from start location to all others in the graph"""
        distances = {start: 0}
        queue = [start]
        idx = 0
        while idx < len(queue):
            current = queue[idx]
            idx += 1
            current_dist = distances[current]
            for neigh in self.adj[current]:
                if neigh not in distances:
                    distances[neigh] = current_dist + 1
                    queue.append(neigh)
        return distances

    def __call__(self, node):
        state = node.state

        # Check if goal reached (heuristic = 0)
        if self.goals <= state:
            return 0

        # Find current location of occupied spot: (zdufuwne ?loc)
        occupied_locations = [get_parts(fact)[1] for fact in state if match(fact, "zdufuwne", "*")]
        # Assuming exactly one occupied location
        if not occupied_locations:
            # Defensive fallback, no occupied cell currently marked; heuristic high
            return float("inf")
        occupied_pos = occupied_locations[0]

        # Get current box locations from state: (ldlu1rwo box loc)
        box_current_positions = {}
        for fact in state:
            if match(fact, "ldlu1rwo", "*", "*"):
                parts = get_parts(fact)
                _, box, loc = parts
                if box in self.boxes:
                    box_current_positions[box] = loc

        # If some boxes missing in state, assign heuristic high cost to partial info
        if not box_current_positions:
            return float("inf")

        # Sum estimated cost as follows:
        # For each box not in goal location, compute:
        # cost = dist(occupied_pos, box_current_loc) + dist(box_current_loc, goal_loc)
        # Choose minimum such value and add number of boxes left (as a rough cost per box)
        boxes_to_move = [box for box in self.boxes if box_current_positions.get(box, None) != self.goal_box_locations.get(box, None)]
        if not boxes_to_move:
            # All boxes in goal
            return 0

        # Compute heuristic candidates for each box
        costs = []
        for box in boxes_to_move:
            current_loc = box_current_positions.get(box, None)
            goal_loc = self.goal_box_locations.get(box, None)

            if current_loc is None or goal_loc is None:
                # Missing info, assign high cost
                continue

            dist_occ_to_box = self.dist.get(occupied_pos, {}).get(current_loc, float("inf"))
            dist_box_to_goal = self.dist.get(current_loc, {}).get(goal_loc, float("inf"))

            cost = dist_occ_to_box + dist_box_to_goal
            costs.append(cost)

        if not costs:
            return float("inf")

        min_cost = min(costs)
        # Add number of boxes left to move as rough cost indicator (each box at least 1 move)
        return min_cost + len(boxes_to_move)
