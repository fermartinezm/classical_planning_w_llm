from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(wuqjn5r5 ip8xvrv6 z4076g0l)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (domain u2gh8ts5).

    Summary:
    This heuristic estimates the number of actions required to transform the current state into a goal state.
    The goal in this domain is to establish certain (wuqjn5r5 ?kukxix7x ?z1mgqxkc) facts as given by the goals.
    The domain actions revolve around:
      - Moving the special predicate (odqd6scj ?z1mgqxkc) between z1mgqxkc objects (like "tokens" or "positions").
      - Modifying predicates (ajcpzwcf ?z1mgqxkc) and (wuqjn5r5 ?kukxix7x ?z1mgqxkc).
      - The connectivity/static relations via predicate (k21e3pdq ?z1mgqxkc ?z1mgqxkc ?yrvdqfln).

    Assumptions:
    - The primary "location" that holds the (odqd6scj ...) predicate is unique at any time: exactly one object of type z1mgqxkc has (odqd6scj).
    - The goal requires specific (wuqjn5r5 ...) facts to hold.
    - The (k21e3pdq ...) predicate encodes a static graph or connectivity with typed edges labeled with yrvdqfln constants.
    - Actions modify these predicates in a way that "moving" odqd6scj along edges, toggling ajcpzwcf marks, and moving wuqjn5r5 from one z1mgqxkc to another.
    - The cost to "move" odqd6scj from one node to another adjacent node is 1 (using action gww2bqr6).
    - The action yaz3jy7e modifies a "marked" node to enable moving wuqjn5r5 objects between nodes but requires specific ajcpzwcf and wuqjn5r5 conditions and also toggles odqd6scj.

    Heuristic Initialization:
    - Extract goal (wuqjn5r5 ?kukxix7x ?z1mgqxkc) facts from the goals.
    - Extract static connectivity edges from (k21e3pdq ?z1mgqxkc ?z1mgqxkc ?yrvdqfln) and store adjacency maps ignoring the edge labels, since actions rely on edges without costs.
    - Extract all z1mgqxkc objects (nodes).
    - Extract all kukxix7x objects (agents) appearing in goals.
    - Since the domain manipulates locations of (wuqjn5r5 ?kukxix7x ?z1mgqxkc) and (odqd6scj ?z1mgqxkc) "marker", the heuristic estimates the cost to:
        a) Move the odqd6scj marker close to the location of the w-r5 wants to reach or control.
        b) Perform the necessary toggling of ajcpzwcf marks to enable moving wuqjn5r5 to their goal locations.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify current positions of:
        - The odqd6scj marker (unique node that has odqd6scj true).
        - Each kukxix7x object’s current position via wuqjn5r5 facts.
        - Which nodes currently have ajcpzwcf true.

    2. For each goal (wuqjn5r5 agent node), check:
        - Is the agent already at the goal node? If yes, no cost for it.
        - Otherwise, estimate cost to move it there.
          The only way to move an agent is by using action yaz3jy7e, which:
            - Requires ajcpzwcf at the target node (where movement happens),
            - Requires odqd6scj at the source node,
            - Moves the marker odqd6scj to the target node,
            - Moves wuqjn5r5 from source node to target node,
            - Toggles ajcpzwcf on source and target nodes.

        - If ajcpzwcf is missing at the node to which movement is desired, it must be prepared by toggling (which is done by yaz3jy7e itself on prior steps).
        This suggests a trade off between moving odqd6scj marker and toggling ajcpzwcf marks.

    3. To model a heuristic estimate, assume:
        - The odqd6scj marker can move cheaply along the static graph edges to any required node with cost equal to the shortest path length.
        - Moving a wuqjn5r5 agent from node A to node B costs at least 1 (one execution of yaz3jy7e).
        - Since yaz3jy7e changes ajcpzwcf marks, toggling might cost additional moves.
        - Simplify by estimating the cost as:
            For each agent not at the goal:
              - Distance from the marker position to the agent's current node (to prepare movement)
              - Plus distance from agent's current node to goal node (assuming one yaz3jy7e action to move)
              - Plus 1 for the action cost.
            This considers the marker moving first near the agent, then moving the agent.

    4. The heuristic sums the costs for all agents not at their goal positions.

    5. Since ajcpzwcf toggling is part of yaz3jy7e and depends on node positions, and marker moves, this heuristic underestimates the extra toggling but is efficient.

    6. The heuristic returns 0 only if all goal wuqjn5r5 facts are met.

    This heuristic may not be admissible but serves as a practical guide for greedy best-first search.

    """

    def __init__(self, task):
        """Initialize by extracting static adjacency and goal positions."""
        self.goals = task.goals

        # Extract static adjacency (undirected edges) ignoring yrvdqfln labels.
        self.adjacency = dict()  # node -> set of neighbor nodes
        self.nodes = set()
        for fact in task.static:
            if match(fact, "k21e3pdq", "*", "*", "*"):
                parts = get_parts(fact)
                from_node, to_node = parts[1], parts[2]
                self.nodes.add(from_node)
                self.nodes.add(to_node)
                self.adjacency.setdefault(from_node, set()).add(to_node)
                self.adjacency.setdefault(to_node, set()).add(from_node)

        # Extract kukxix7x agents that appear in goal wuqjn5r5 facts, and their goal nodes
        self.goal_positions = dict()  # agent -> node
        for goal in self.goals:
            if match(goal, "wuqjn5r5", "*", "*"):
                parts = get_parts(goal)
                agent = parts[1]
                node = parts[2]
                self.goal_positions[agent] = node

        # Initialize for quick BFS searches
        self.nodes = list(self.nodes)

    def shortest_path_length(self, start, targets):
        """Compute shortest path length from start node to any node in targets set using BFS over adjacency."""
        if start in targets:
            return 0
        from collections import deque

        visited = {start}
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    if neighbor in targets:
                        return dist + 1
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # No path found
        return None

    def __call__(self, node):
        """Estimate the number of actions required to reach a goal state from the current state."""
        state = node.state

        # If goal reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Identify current odqd6scj location (only one expected)
        odqd_nodes = [get_parts(fact)[1] for fact in state if match(fact, "odqd6scj", "*")]
        if not odqd_nodes:
            # No marker position found, extremely bad/invalid state, return high heuristic
            return 1000
        odqd_node = odqd_nodes[0]

        # Map agents to current nodes by wuqjn5r5 facts
        agent_positions = dict()
        for fact in state:
            if match(fact, "wuqjn5r5", "*", "*"):
                parts = get_parts(fact)
                agent = parts[1]
                node_ = parts[2]
                agent_positions[agent] = node_

        # Find ajcpzwcf nodes (could be needed for estimations, but ignoring detailed toggling cost)
        ajcpzwcf_nodes = {get_parts(f)[1] for f in state if match(f, "ajcpzwcf", "*")}

        total_cost = 0

        for agent, goal_node in self.goal_positions.items():
            current_node = agent_positions.get(agent, None)
            if current_node is None:
                # Agent not found in state? High cost penalization.
                total_cost += 10
                continue

            if current_node == goal_node:
                # Already at goal, no cost
                continue

            # Cost part 1: Move marker odqd to the agent's position to enable movement
            dist_marker_to_agent = self.shortest_path_length(odqd_node, {current_node})
            if dist_marker_to_agent is None:
                # If marker cannot reach current node, large cost penalty
                dist_marker_to_agent = 10

            # Cost part 2: Move agent from current node to goal node
            dist_agent_to_goal = self.shortest_path_length(current_node, {goal_node})
            if dist_agent_to_goal is None:
                # If agent cannot reach goal node (also network disconnected), high penalty
                dist_agent_to_goal = 10

            # Cost part 3: Estimate minimal number of actions
            # Each move of marker or agent along an edge costs at least one action (gww2bqr6 or yaz3jy7e),
            # plus 1 for the actual transfer action yaz3jy7e,
            # toggling ajcpzwcf costs are absorbed into action cost as approximation.

            # Since in domain:
            # - Moving odqd marker along an edge is one action.
            # - Moving agent from node A to neighbor node B is one yaz3jy7e action, requires marker on source.
            # To move agent more than one edge away, multiple yaz3jy7e actions needed (assuming agent single-hop moves).
            # Here we approximate agent movement cost by its graph distance + 1 (for toggling and marker movement).

            # We heuristically count:
            #  cost_marker_move + cost_agent_moves + 1 (for toggling / finalization)

            # The heuristic will be: distance marker->agent + distance agent->goal + 1
            agent_cost = dist_marker_to_agent + dist_agent_to_goal + 1

            total_cost += agent_cost

        # Heuristic is finite and zero only when goal facts all hold.
        return total_cost
