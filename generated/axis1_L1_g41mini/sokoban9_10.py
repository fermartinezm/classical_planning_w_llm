from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(h80z8tvp loc_4_4 loc_4_5 gp6ny00g)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (kyjn6f1j).

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal 
    of moving boxes (rullurva) to their goal locations (ldlu1rwo predicates).
    The domain models some form of box-location movement through locations (wlo7ezwc)
    connected by typed edges with associated "weights" (cf. h80z8tvp predicate).
    The predicate zdufuwne marks which location currently holds the "agent" or "active marker".
    The heuristic thus focuses on estimating the minimal number of moves 
    to relocate each box from its current location to its goal location by moving 
    the "active" marker around locations.

    Assumptions
    -----------
    - The state encodes boxes positioned at specific locations via (ldlu1rwo box loc).
    - The goal is to achieve certain (ldlu1rwo box loc) facts.
    - zdufuwne marks exactly one location as "active" at any time.
    - Locations are connected by directional edges encoded in (h80z8tvp loc_from loc_to sia5d33v).
    - Predicate (qsw2k0xh loc) denotes "special" or "accessible" locations needed to move boxes onto.
    - Actions move the "active" mark or the box along these edges.
    - The heuristic ignores the actual edge weights (a6sst constants) since they do not affect action counts directly.
    - Moving the box requires navigating the "active" mark close enough so the box can be pushed/pulled along h80z8tvp edges.
    - The heuristic assumes shortest paths between relevant locations ignoring other constraints.

    Heuristic Initialization
    ------------------------
    - Extract the static graph of locations connected by edges (h80z8tvp), ignoring the actual "a6sst" constants.
    - Build adjacency maps for fast shortest path computation.
    - Index goal box positions from the goal facts.
    - Store the set of "accessible" locations (qsw2k0xh) to consider valid intermediate positions for the box and agent.
    - Identify all boxes in the instance from goal conditions.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the active location from the current state (the unique location with (zdufuwne loc)).
    2. Extract current box locations from (ldlu1rwo box loc) predicates in the state.
    3. For each box, find its goal location from the goals.
    4. Estimate the cost to move the box from current to goal location:
       - The box can only be moved along connected edges (h80z8tvp).
       - Movement of the box requires the active location to be adjacent to the box location on an accessible edge.
       - Therefore, estimate:
         a) The shortest path length to move the box from current to goal location over the graph restricted to accessible locations.
         b) The shortest path length for the active marker to get into position to move the box step-by-step.
    5. For each box, the heuristic is approximated by:
       cost_box = shortest path length from box current to goal +
                  minimal shortest path for agent from current active location 
                  to a location adjacent to box's start position to start moving it.
    6. Since the heuristic is greedy and need not be admissible, summing these costs per box suffices.
    7. The final heuristic value is the sum over all boxes of these costs, representing minimal estimated moves.

    Note:
    - If the state is a goal state, heuristic is 0.
    - The heuristic is finite as long as paths exist; if no path, heuristic returns a large number.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Parse static graph edges (h80z8tvp loc_from loc_to a6sst)
        self.graph_forward = {}  # loc_from -> set(loc_to)
        self.graph_backward = {}  # loc_to -> set(loc_from)
        self.accessible_locations = set()
        self.boxes = set()  # All boxes found from goal facts: rullurva objects

        # We consider accessible locations those appearing in qsw2k0xh predicate and in nodes of graph
        self.all_locations = set()
        # Map facts of predicate qsw2k0xh to identify accessible nodes
        for fact in task.static:
            if match(fact, "h80z8tvp", "*", "*", "*"):
                parts = get_parts(fact)
                loc_from = parts[1]
                loc_to = parts[2]
                self.all_locations.add(loc_from)
                self.all_locations.add(loc_to)
                # Build forward adjacency
                if loc_from not in self.graph_forward:
                    self.graph_forward[loc_from] = set()
                self.graph_forward[loc_from].add(loc_to)
                # Build backward adjacency
                if loc_to not in self.graph_backward:
                    self.graph_backward[loc_to] = set()
                self.graph_backward[loc_to].add(loc_from)

        for fact in task.static:
            if match(fact, "qsw2k0xh", "*"):
                loc = get_parts(fact)[1]
                self.accessible_locations.add(loc)

        # Boxes and goals: from goal facts ldlu1rwo box loc
        self.goal_positions = {}  # box -> loc
        for g in self.goals:
            if match(g, "ldlu1rwo", "*", "*"):
                parts = get_parts(g)
                box = parts[1]
                loc = parts[2]
                self.goal_positions[box] = loc
                self.boxes.add(box)

        # Precompute shortest paths between all locations and all accessible locations for efficiency
        # We'll use BFS since edges have uniform cost
        # Only consider locations in self.accessible_locations for box movement
        self.accessible_graph = {}
        for loc in self.all_locations:
            # Keep only neighbors that are also accessible locations (for box moves)
            if loc in self.graph_forward:
                self.accessible_graph[loc] = set(
                    n for n in self.graph_forward[loc] if n in self.accessible_locations
                )
            else:
                self.accessible_graph[loc] = set()

    def _bfs_shortest_path_length(self, graph, start, goal):
        """Breadth-first search to find shortest path length between start and goal in graph.
        Returns integer length or None if no path exists.
        """
        if start == goal:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            for neighbor in graph.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None

    def _all_distances_from(self, graph, start):
        """Return dict location->distance from start by BFS."""
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for nbr in graph.get(current, []):
                if nbr not in distances:
                    distances[nbr] = distances[current] + 1
                    queue.append(nbr)
        return distances

    def __call__(self, node):
        state = node.state

        # 1) If goal reached, heuristic is 0
        if self.goals <= state:
            return 0

        # 2) Identify active location: unique loc with (zdufuwne loc)
        active_locations = [get_parts(f)[1] for f in state if match(f, "zdufuwne", "*")]
        if len(active_locations) != 1:
            # Defensive fallback: if no or multiple active locations (should not occur), return large value
            return 100000
        active_loc = active_locations[0]

        # 3) Current box locations from (ldlu1rwo box loc)
        box_positions = {}
        for fact in state:
            if match(fact, "ldlu1rwo", "*", "*"):
                parts = get_parts(fact)
                box = parts[1]
                loc = parts[2]
                box_positions[box] = loc

        # 4) For each box, estimate effort to move from current location to goal location
        heuristic_value = 0

        # Precompute distances for the active location to all accessible locations (for agent moves)
        active_distances = self._all_distances_from(self.accessible_graph, active_loc)

        # For each box:
        for box in self.boxes:
            if box not in box_positions:
                # Box missing - treat as large cost
                heuristic_value += 1000
                continue

            start_loc = box_positions[box]
            goal_loc = self.goal_positions[box]

            # If box already at goal, cost zero for this box
            if start_loc == goal_loc:
                continue

            # Check box path distance in accessible graph: shortest path from start_loc to goal_loc
            box_dist = self._bfs_shortest_path_length(self.accessible_graph, start_loc, goal_loc)
            if box_dist is None:
                # No path for box -> unreachable goal
                return 100000

            # The agent must get adjacent to the box to push it.
            # Agent must move from active_loc to some location adjacent to start_loc in accessible graph
            # Adjacent locations are those with an edge to start_loc or from start_loc (as movement is directional)
            # We consider only accessible locations, as agent must stand on qsw2k0xh locations
            adjacent_locations = set()
            # backward neighbors (can agent come from these to start_loc)
            for loc in self.graph_backward.get(start_loc, []):
                if loc in self.accessible_locations:
                    adjacent_locations.add(loc)
            # forward neighbors - agent might push/pull from forward side also (be cautious)
            for loc in self.graph_forward.get(start_loc, []):
                if loc in self.accessible_locations:
                    adjacent_locations.add(loc)

            # Compute minimal agent distance to any adjacent location
            min_agent_dist = None
            for adj in adjacent_locations:
                d = active_distances.get(adj)
                if d is not None:
                    if min_agent_dist is None or d < min_agent_dist:
                        min_agent_dist = d

            if min_agent_dist is None:
                # Agent cannot get adjacent to the box start location
                return 100000

            # Estimated cost for this box is:
            # cost to get agent adjacent + cost to move box to goal (each move assumed 1 action)
            # Note: each box move requires agent positioning + actual moves; we simplify by summing distances
            heuristic_value += min_agent_dist + box_dist

            # Update active location to goal_loc for next box, because agent ends near last box moved
            # This is a heuristic approximation assuming sequential moves
            active_loc = goal_loc
            active_distances = self._all_distances_from(self.accessible_graph, active_loc)

        return heuristic_value
