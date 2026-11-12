from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
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
    This heuristic estimates the number of actions required to move all boxes (rullurva objects)
    from their current locations to their respective goal locations, considering the layout 
    defined by locations (wlo7ezwc) and connectivity edges labeled by the sia5d33v constants.
    It exploits the structure of the domain involving:
      - zdufuwne(?loc): denotes the current location of the agent (exactly one true at a time).
      - ldlu1rwo(?box ?loc): the box at a location.
      - qsw2k0xh(?loc): a special property ("qualified" locations).
      - h80z8tvp(?loc1 ?loc2 ?label): directed edges between locations with labels 
        in sia5d33v constants.

    The two actions:
      - x042b0fi: move agent's position between locations connected by edges with matching labels,
        requires the target to be qsw2k0xh, and changes agent position.
      - yi0kp1zg: pushes a box from one location to another along two consecutive edges 
        with the same label, transferring the box and agent, and modifying qsw2k0xh predicates.

    The heuristic estimates the effort needed for the agent to push all boxes to their goal locations 
    by computing shortest push-distances on a graph built from static edges and reachable qsw2k0xh 
    locations, taking into account the labels on edges and the constraints that pushing requires two 
    consecutive edges with the same label.

    Assumptions
    -----------
    - Exactly one agent location (zdufuwne) is true in each state.
    - The goal always requires each box (rullurva) to be at some specific location.
    - The static graph (h80z8tvp) is fixed and used for path computations.
    - Allowed push moves correspond to two-step paths where both edges share the same label.
    - Moving the agent alone costs 1 action, pushing a box costs 1 action.
      (We measure only pushing steps since moves without pushing are less crucial for goal progress.)
    - The heuristic is not guaranteed admissible but aims to minimize expanded nodes.

    Heuristic Initialization
    ------------------------
    On initialization, the heuristic:
    - Extracts all boxes, their goal locations (from ldlu1rwo(goal_box, goal_loc) in task.goals),
      and builds lookup maps.
    - Processes static facts (h80z8tvp) to construct:
      - A directed labeled multigraph of locations:
        location -> label -> set of neighbors (locations reachable with that label).
      - The set of all locations and the set of qsw2k0xh locations.
    - Precomputes all locations that are qsw2k0xh in the initial state for checking movable locations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify current agent location from zdufuwne predicate in the state.
    2. Identify current location of each box from ldlu1rwo predicates.
    3. Identify target (goal) location of each box from goals.
    4. Compute for each box the minimal "push-distance" from current to goal:
       - Using a specialized graph search on the directed labeled edges:
         - Pushing requires finding two consecutive edges with the *same* label to move box and agent forward.
         - The heuristic defines transitions by pairs (current_loc, next_loc) with a particular label.
         - The push step can be seen as a transition over two edges with identical label.
       - Compute shortest push-path length from the box's current location to its goal.
    5. Sum these minimal push-distances as the heuristic estimate.
    6. Since agent moves (x042b0fi) also require movements, add minimal agent steps from current agent location
       to positions adjacent to boxes to start pushing:
       - Estimate agent distance to each box's current location ignoring labels (simplified shortest path).
       - Add minimal distances of agent moving to boxes.
    7. Total heuristic returns sum of all push-distances plus minimal cost for agent positioning.
    8. If state is goal state, return 0.

    This heuristic balances accurate estimates of pushing steps (which are costly) and agent positioning 
    moves with efficient computation using preprocessed graphs.

    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.operators = task.operators

        # Set of all boxes (rullurva), inferred from goal facts and initial facts.
        self.boxes = set()
        self.goal_positions = {}

        # Extract goal box positions from goal ldlu1rwo facts: (ldlu1rwo ?box ?loc)
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "ldlu1rwo":
                box = parts[1]
                loc = parts[2]
                self.goal_positions[box] = loc
                self.boxes.add(box)

        # Extract known box set from goals (and maybe from static/initial)
        # Just ensure boxes set includes all known from goals.
        # Usually, boxes won't appear elsewhere forcibly.

        # Build graph: location: {label: set(neighbor_locations)}
        self.graph = {}
        self.locations = set()
        self.labels = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "h80z8tvp":
                src, dst, label = parts[1], parts[2], parts[3]
                self.locations.add(src)
                self.locations.add(dst)
                self.labels.add(label)
                if src not in self.graph:
                    self.graph[src] = {}
                if label not in self.graph[src]:
                    self.graph[src][label] = set()
                self.graph[src][label].add(dst)

        # Create undirected graph ignoring labels for agent movement over qsw2k0xh locations.
        # This enables computing minimal agent steps disregarding label constraints.
        self.agent_graph = {}
        for src in self.graph:
            for label, neighbors in self.graph[src].items():
                if src not in self.agent_graph:
                    self.agent_graph[src] = set()
                self.agent_graph[src].update(neighbors)
                # Also add reverse edges for undirected connectivity if appearing in static (check reverse edges exist)
                # Because push edges are directed, but agent can move in reverse via x042b0fi if edges exist.
                for nb in neighbors:
                    if nb not in self.agent_graph:
                        self.agent_graph[nb] = set()
                    self.agent_graph[nb].add(src)

        # We will also record which locations are qsw2k0xh in the initial state.
        # This is important since movement and pushing require qsw2k0xh state at locations.
        # As static facts do not convey this, this must be checked per state.

    def _get_agent_location(self, state):
        """Return the single location where zdufuwne(?loc) holds in state (agent's location)."""
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "zdufuwne":
                return parts[1]
        return None  # Should not happen if domain is consistent

    def _get_boxes_locations(self, state):
        """Return a dict {box: location} from ldlu1rwo facts in state."""
        box_locs = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ldlu1rwo":
                box, loc = parts[1], parts[2]
                if box in self.boxes:
                    box_locs[box] = loc
        return box_locs

    def _get_qsw2k0xh_locations(self, state):
        """Return set of locations currently with qsw2k0xh predicate true."""
        qsw2k0xh_locs = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "qsw2k0xh":
                qsw2k0xh_locs.add(parts[1])
        return qsw2k0xh_locs

    def _bfs_agent_distance(self, start, targets, qsw2k0xh_locs):
        """
        Compute minimal distance from start location to any location in targets.
        Movement allowed only via edges respecting that both ends are in qsw2k0xh locations,
        consistent with preconditions of x042b0fi (agent moves on qsw2k0xh).
        Uses undirected agent graph edges ignoring labels for efficiency.
        Returns distance in number of moves, or a large number if no path.
        """
        from collections import deque

        if start not in qsw2k0xh_locs:
            # Agent must be on a qsw2k0xh location for any move to be legal by domain,
            # so if not, movement distance is very large.
            return 1000000

        visited = set()
        queue = deque()
        queue.append((start, 0))
        visited.add(start)
        targets = set(targets)
        while queue:
            loc, dist = queue.popleft()
            if loc in targets:
                return dist
            for nb in self.agent_graph.get(loc, set()):
                if nb not in visited and nb in qsw2k0xh_locs:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return 1000000  # Not reachable

    def _min_push_distance(self, start, goal, qsw2k0xh_locs):
        """
        Compute shortest push distance from start to goal locations.

        Push movement rules:
        - A push moves a box from current location to a neighbor's neighbor location by two consecutive edges with the same label.
        - The agent must be at the intermediate location to perform the push.
        - Both start and goal locations must be in qsw2k0xh locations (push destination valid).
        
        Approach:
        - Model the state as pair (box_loc, agent_loc)
        - Agent loc must be adjacent to box_loc via an edge with label L, and box_loc has an outgoing edge with label L to dest.
        - The push moves box_loc to dest and agent_loc to box_loc.
        
        For heuristic efficiency:
        - We compute from start locations to the goal ignoring exact agent position initially.
        - We perform a search over box positions with implicit agent compatibility.
        - Return minimal number of pushes (actions).

        Returns:
            Integer push-distance, or a large number if no path.
        """
        from collections import deque

        if start == goal:
            return 0

        # For performance, we only track box positions and feasible agent positions ignoring full label variations.
        # State: (box_loc, agent_loc)
        # We start with all possible agent_loc adjacent to box_loc with some label that permits pushing forward.

        # Find all locations agent can stand to push box forward:
        def feasible_pushes(box_pos):
            """
            Yield (label, agent_pos, new_box_pos)
            for all possible pushes starting from box_pos.
            Conditions:
             - agent_pos in qsw2k0xh_locs
             - box_pos in qsw2k0xh_locs
             - There exists a label L, and locations agent_pos -> box_pos -> new_box_pos with label L
             - All locations involved in qsw2k0xh_locs
            """
            if box_pos not in self.graph:
                return

            for label, neighbors in self.graph[box_pos].items():
                for new_box_pos in neighbors:
                    if new_box_pos not in qsw2k0xh_locs:
                        continue
                    # We want agent to be adjacent to box_pos via an edge with label
                    # Find agent_pos where agent_pos -> box_pos with same label L
                    # Check reverse edges from box_pos
                    for agent_pos_candidate, edges in self.graph.items():
                        if label in edges and box_pos in edges[label]:
                            if agent_pos_candidate in qsw2k0xh_locs:
                                yield (label, agent_pos_candidate, new_box_pos)

        # Initialize queue with all (start, agent_pos) pairs where agent_pos adjacent to start with some label 
        # and all involved locations have qsw2k0xh.
        # If no such agent position, fallback to just pushing with agent on start (rough estimate).
        # To find initial agent positions:
        initial_states = set()
        for label, neighbors in self.graph.get(start, {}).items():
            for nb in neighbors:
                # Agent must be at neighbor of box, so look for agent_pos such that:
                # There is an edge agent_pos -> box_pos with label L for push.
                # So here agent_pos is "neighbor" of box_pos = start, but push requires two edges with same label.
                # We consider that for initial agent to push box from start, agent must currently be adjacent to box.

                # We check also reverse edges: agent -> box with label
                # So agent_pos is any location such that agent_pos -> start with label
                for agent_pos_candidate, edges in self.graph.items():
                    if label in edges and start in edges[label]:
                        # Check all involved qsw2k0xh
                        if agent_pos_candidate in qsw2k0xh_locs and start in qsw2k0xh_locs:
                            initial_states.add((start, agent_pos_candidate))

        # If no initial states found (no edges reverse to box start), just start with agent at box location
        if not initial_states:
            if start in qsw2k0xh_locs:
                initial_states.add((start, start))
            else:
                # No valid push possible if start not qsw2k0xh
                return 1000000

        visited = set(initial_states)
        queue = deque([(box_loc, agent_loc, 0) for (box_loc, agent_loc) in initial_states])

        while queue:
            box_pos, agent_pos, dist = queue.popleft()
            if box_pos == goal:
                return dist

            # Explore pushes from current state
            for label, new_agent_pos, new_box_pos in feasible_pushes(box_pos):
                # Push is possible if agent_pos == new_agent_pos in previous state?
                # Wait, the push moves box_pos -> new_box_pos, and agent moves box_pos <- agent_pos
                # Actually, after push:
                # New agent_pos = box_pos (agent moves onto box_pos)
                # So check if current agent_pos == old agent_pos and new_agent_pos corresponds to old agent_pos, etc.

                # The push action:
                # Preconditions:
                #   agent_pos adjacent to box_pos with label L (agent_pos -> box_pos label L)
                # Effects:
                #   box_pos -> new_box_pos with label L
                #   agent_pos moves onto box_pos (now agent_pos after push = box_pos)
                # So new agent_pos after push = box_pos
                # Therefore, to allow push, current agent_pos must be the agent_pos from yield,
                # and new agent_pos after push must be box_pos.

                # So only consider pushes where agent_pos matches the agent_pos candidate and new agent_pos = box_pos (current box location)
                if agent_pos == new_agent_pos:
                    new_state = (new_box_pos, box_pos)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_box_pos, box_pos, dist + 1))

        return 1000000  # No push path found

    def __call__(self, node):
        """
        Compute an estimate of the cost (number of actions) to reach the goal from the given node state.
        The heuristic sums minimal push distances of all boxes to their goal locations plus
        minimal agent moves to boxes to start pushing.

        Returns 0 if goal is reached.
        """
        state = node.state

        # Check goal condition fast: heuristic is 0 only in goal states
        if self.goals <= state:
            return 0

        # Extract current agent location
        agent_loc = self._get_agent_location(state)
        if agent_loc is None:
            # No agent location found, infinitely costly (invalid state)
            return 1000000

        # Extract current boxes locations
        boxes_locations = self._get_boxes_locations(state)
        if not boxes_locations:
            # No boxes found, possibly goal achieved or incorrect state
            return 0 if self.goals <= state else 1000000

        # Extract current qsw2k0xh locations
        qsw2k0xh_locs = self._get_qsw2k0xh_locations(state)

        # If agent location not in qsw2k0xh, agent can't move or push legally
        if agent_loc not in qsw2k0xh_locs:
            return 1000000

        # Compute minimal agent move to any box location (to start pushing)
        box_locs = list(boxes_locations.values())
        min_agent_to_box_dist = self._bfs_agent_distance(agent_loc, box_locs, qsw2k0xh_locs)

        # Sum push distances for all boxes from current to goal location
        total_push_distance = 0
        for box, box_loc in boxes_locations.items():
            goal_loc = self.goal_positions.get(box)
            if goal_loc is None:
                # No known goal for this box - skip or assume no cost
                continue
            if box_loc == goal_loc:
                continue  # box already at goal

            push_dist = self._min_push_distance(box_loc, goal_loc, qsw2k0xh_locs)
            total_push_distance += push_dist

        # Heuristic estimate = sum of pushes + minimal agent move to a box
        # We add 1 to min_agent_to_box_dist if > 0 to represent agent moves necessary to get ready to push
        agent_move_cost = min_agent_to_box_dist if min_agent_to_box_dist != 1000000 else 0

        heuristic_value = total_push_distance + agent_move_cost

        # Ensure heuristic is finite; if infinite, return large number
        if heuristic_value >= 1000000:
            heuristic_value = 1000000

        return heuristic_value
