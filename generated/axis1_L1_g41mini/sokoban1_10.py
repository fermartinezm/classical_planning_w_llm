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

    Summary
    -------
    This heuristic estimates the number of actions needed to move the box(es)
    (objects of type rullurva) from their current locations to their goal locations
    based on the shortest path on the graph of locations given by the adjacency relation
    encoded in the predicate `h80z8tvp`. Additionally, it considers the current position
    of the "agent" (the unique location marked with `zdufuwne`) and the `qsw2k0xh` predicate,
    which marks special locations relevant for movement feasibility. The heuristic returns 
    a sum of minimal required moves to push the box(es) closer to goal locations, using
    graph distance approximations to guide the search.

    Assumptions
    -----------
    - "zdufuwne ?location" indicates the current location of the "agent".
    - "ldlu1rwo ?box ?location" indicates the location of a box (rullurva object).
    - "qsw2k0xh ?location" indicates that this location is accessible or special for moving.
    - "h80z8tvp ?loc1 ?loc2 ?constant" encodes edges in a graph between locations.
    - The agent pushes boxes along paths formed by the adjacency predicate h80z8tvp.
    - Each box needs to be pushed from its current location to its goal location.
    - The heuristic is allowed to be non-admissible; it aims to be efficient and informative.

    Heuristic Initialization
    ------------------------
    - Extract all box names and their goal locations from the goal conditions (predicate ldlu1rwo).
    - Build a graph of locations, where edges are undirected connections derived from the static facts (h80z8tvp).
      Multiple edges between locations may exist but are treated simply as adjacency.
    - Extract the set of "reachable" locations (marked by qsw2k0xh facts).
    - Store the unique agent location (zdufuwne).
    - Precompute the graph to allow shortest path calculations efficiently at heuristic call time.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Identify the current location of the agent.
    2. For each box that has a goal, find:
       - The current location of the box (from ldlu1rwo facts).
       - The goal location of that box (from stored goals).
    3. For each box:
       - Compute the shortest distance from the box's current location to its goal location on the location graph.
       - Compute the shortest distance from the agent location to a location adjacent to the box, from which it can push the box toward the goal path.
       - Use an approximate formula to estimate minimal push moves as at least the distance for the box plus
         the cost for the agent to get close enough to the box.
    4. Sum these estimates for all boxes.
    5. Return zero if the goal is reached.
    6. Return a finite estimate otherwise. If any distance cannot be computed due to disconnected graph parts,
       return a large (=high) heuristic value.

    Note: The heuristic ignores the specifics of "a6sst" parameter in edges or actions, assuming the constant keys are irrelevant.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all boxes (rullurva type) that appear in goal locations.
        self.box_goals = {}  # box_name -> goal_location
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "ldlu1rwo" and len(parts) == 3:
                box, goal_loc = parts[1], parts[2]
                self.box_goals[box] = goal_loc

        # Extract all qsw2k0xh locations (reachable or special locations).
        self.qsw2k0xh_locations = {get_parts(fact)[1] for fact in task.static if match(fact, "qsw2k0xh", "*")}
        # Also include those in initial state, as the example shows qsw2k0xh are fluents? But domain suggests static? Take union:
        self.qsw2k0xh_locations.update({get_parts(fact)[1] for fact in task.initial_state if match(fact, "qsw2k0xh", "*")})

        # Extract graph edges from static facts "h80z8tvp locA locB constant"
        # We treat edges as undirected.
        self.graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "h80z8tvp" and len(parts) == 4:
                loc1, loc2 = parts[1], parts[2]
                # Add edge loc1 <-> loc2
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1)

        # The heuristic needs fast shortest path queries. We'll implement a BFS for shortest paths on-the-fly.

    def shortest_path_length(self, start, goal):
        """
        Compute shortest distance in undirected graph from start to goal.

        Returns integer distance or None if no path.
        """
        if start == goal:
            return 0
        if start not in self.graph or goal not in self.graph:
            return None

        from collections import deque

        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.graph.get(current, ()):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Extract agent location from predicate "zdufuwne ?loc"
        agent_loc = None
        for fact in state:
            if match(fact, "zdufuwne", "*"):
                agent_loc = get_parts(fact)[1]
                break
        if agent_loc is None:
            # No agent location known; assign a large heuristic.
            return 1000

        # Extract box current locations from "ldlu1rwo box loc"
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "ldlu1rwo":
                box, loc = parts[1], parts[2]
                if box in self.box_goals:
                    box_locations[box] = loc

        # If some boxes in goals have no current location info, assign a large heuristic.
        for box in self.box_goals:
            if box not in box_locations:
                return 1000

        # Check which locations are "pushable" by the agent: those with qsw2k0xh predicate true.
        # These represent reachable locations that allow moves/actions to be applied.

        # For each box, estimate minimal cost to push box from current location to goal location.
        # We sum over all boxes' individual estimates.

        total_estimate = 0

        # We consider that to push a box, the agent must move adjacent to the box's location.
        # Each step for pushing is approximated as one action.
        # Agent movements to reach pushing locations are considered similarly.

        # We approximate pushing cost as shortest path length from box's loc to goal loc
        # plus minimal distance for agent to reach an adjacent location to the box's current loc,
        # in order to start pushing.

        for box, goal_loc in self.box_goals.items():
            box_loc = box_locations[box]

            # If box is already at goal location, cost 0 for this box.
            if box_loc == goal_loc:
                continue

            # Compute shortest path box loc -> goal loc
            distance_box_to_goal = self.shortest_path_length(box_loc, goal_loc)
            if distance_box_to_goal is None:
                # No path, very hard to solve.
                return 1000

            # Find agent distance to a neighbor of the box location from which pushing can start.
            # That is, agent must be at a location adj to the box location (connected by an edge),
            # and that location must be reachable (qsw2k0xh true in state).

            neighbors_of_box = self.graph.get(box_loc, set())

            candidate_agent_positions = [loc for loc in neighbors_of_box if loc in self.qsw2k0xh_locations]

            if not candidate_agent_positions:
                # No reachable adjacent push position, heuristically very hard.
                return 1000

            distances_agent_to_neighbors = []
            for pos in candidate_agent_positions:
                dist = self.shortest_path_length(agent_loc, pos)
                if dist is not None:
                    distances_agent_to_neighbors.append(dist)
            if not distances_agent_to_neighbors:
                # Agent cannot reach any pushing position.
                return 1000

            min_agent_to_push_pos = min(distances_agent_to_neighbors)

            # Estimate cost for this box:
            # heuristic = distance to push box to goal + cost for agent to get to pushing position
            # Plus 1 to represent starting push action.

            cost_for_box = distance_box_to_goal + min_agent_to_push_pos + 1

            total_estimate += cost_for_box

        return total_estimate
