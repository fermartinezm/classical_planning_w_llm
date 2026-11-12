from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact string by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check whether a fact matches the given pattern with wildcards.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (kyjn6f1j).

    Summary:
    This heuristic estimates the number of actions needed to move boxes (rullurva) to their goal locations.
    The domain models movement of a "zdufuwne" entity (like a robot/agent) across locations of type wlo7ezwc,
    pushing or transporting boxes along connections labeled with "sia5d33v" types.
    The goal is that certain boxes (rullurva) are at given locations (wlo7ezwc), as specified by ldlu1rwo predicates.

    Assumptions:
    - Exactly one entity is "zdufuwne" at any given location (the agent).
    - Boxes can only be moved along edges indicated by the h80z8tvp predicate.
    - The predicates qsw2k0xh mark accessible locations where the agent/boxes can exist.
    - The agent moves by the action "x042b0fi" that moves itself between locations.
    - The agent can push (transports) boxes along chains of locations if connectivity and ownership conditions hold,
      as modeled by the "yi0kp1zg" action.
    - The heuristic is not necessarily admissible but should guide search efficiently.

    Heuristic Initialization:
    - Extract the goal assignments for each box from the goal facts (ldlu1rwo box location).
    - Extract static connectivity information between locations from h80z8tvp facts.
      Build a graph indexed by location nodes, with edges labeled by sia5d33v types.
    - Extract accessible locations (those with qsw2k0xh).
    - Extract the agent's current location from the unique zdufuwne fact.
    - Extract positions of boxes from ldlu1rwo facts in the state.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify for each box its current location and its goal location.
    2. Compute shortest path distances between locations on the static connectivity graph,
       ignoring edge labels (sia5d33v) because actions require exact matching but for heuristic estimate a simplified graph suffices.
    3. Estimate the minimal number of moves required to:
       a) Move the agent from its current location to the box's location.
       b) Push or transport the box along the shortest path from its current to its goal location.
    4. Since pushing a box is more costly than just moving the agent, count pushing steps with weight 2, moving steps with weight 1.
       Here pushing corresponds to box movement actions (yi0kp1zg).
    5. Sum these estimates for all boxes not yet at the goal.
    6. Return zero if all boxes are at their goal, because the state is goal.
    7. This heuristic is finite for all reachable states because the graph is connected from static facts and states are solvable.

    This heuristic ignores some constraints (like the required ownership and exact agent paths), but provides
    a good cost estimate to guide greedy best-first search.
    """

    def __init__(self, task):
        """
        Precompute static structures for heuristic:

        - Build location graph from (h80z8tvp loc1 loc2 type) facts for fast distance calculations.
        - Store goal locations for each box.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract all ldlu1rwo goals: (ldlu1rwo box location)
        self.goal_positions = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "ldlu1rwo":
                box, loc = parts[1], parts[2]
                self.goal_positions[box] = loc

        # Build undirected graph of locations as adjacency list based on h80z8tvp connecting edges.
        # We'll ignore edge labels (sia5d33v) for heuristic.
        self.graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "h80z8tvp":
                loc1, loc2 = parts[1], parts[2]
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1)

        # Precompute shortest paths between all pairs of locations needed or compute on demand via BFS.
        # We will implement BFS helper method to compute shortest distances.

    def shortest_path_length(self, start, goal):
        """
        Compute the shortest path length between two locations in the location graph using BFS.
        Returns an integer >= 0 or None if unreachable (shouldn't happen in solvable states).
        """
        if start == goal:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.graph.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None  # no path found

    def __call__(self, node):
        """
        Compute heuristic estimation for node.state.

        - Extract box current positions from ldlu1rwo facts.
        - Identify agent location from zdufuwne fact.
        - For each box not at goal, estimate cost as:
          cost(agent_to_box) + 2 * cost(box_to_goal),
          where moving agent on empty location costs 1 per step,
          pushing box costs 2 per step, roughly approximating the action costs.
        - Sum over all boxes to produce heuristic value.
        - Return 0 if all boxes at goal.
        """
        state = node.state

        # Extract current box locations: from ldlu1rwo facts
        box_positions = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ldlu1rwo":
                box_positions[parts[1]] = parts[2]

        # Extract agent location from unique zdufuwne fact (zdufuwne loc)
        # We assume exactly one such fact exists.
        agent_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "zdufuwne":
                agent_loc = parts[1]
                break
        if agent_loc is None:
            # Should not happen if state is valid; no agent location means infinite cost.
            return float("inf")

        total_cost = 0

        # If no boxes to move, heuristic is zero.
        if not self.goal_positions:
            return 0

        # For each box with goal position, estimate required moves.
        for box, goal_loc in self.goal_positions.items():
            box_loc = box_positions.get(box)
            # If box is already at goal, no cost.
            if box_loc == goal_loc:
                continue

            # Compute agent to box distance
            agent_to_box = self.shortest_path_length(agent_loc, box_loc)
            if agent_to_box is None:
                # No path from agent to box (unsolvable)
                return float("inf")

            # Compute box to goal distance
            box_to_goal = self.shortest_path_length(box_loc, goal_loc)
            if box_to_goal is None:
                # No path from box to goal (unsolvable)
                return float("inf")

            # Estimate: cost = agent travel cost + box pushing cost.
            # Agent moving alone: cost 1 per move (x042b0fi)
            # Pushing box: cost 2 per step (yi0kp1zg requires more conditions, push modeled higher cost)
            cost = agent_to_box + 2 * box_to_goal

            total_cost += cost

            # Update agent location to box's goal for next box,
            # assuming agent must move there after pushing this box.
            # This is a relaxation to tighten heuristic.
            agent_loc = goal_loc

        # If all boxes at goal, cost is zero.
        if total_cost == 0:
            return 0

        return total_cost
