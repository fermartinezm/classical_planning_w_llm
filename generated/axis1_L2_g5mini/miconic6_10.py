from collections import deque
from heuristics.heuristic_base import Heuristic

UNREACHABLE_COST = 10 ** 6


def get_parts(fact):
    """Return list of tokens for a fact string like '(pred a b)' -> ['pred', 'a', 'b']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the secret_domain (unkhrp2l) planning domain. In this domain there is a single
    mobile marker (represented by predicate `v0yy74n7 <loc>`) that can move along locations connected
    by the static relation `pvhp8m0p <loc1> <loc2>`. Agents of type `jxhpkqr5` have:
      - a pickup fact `pxch7pi8 <agent> <pickup_loc>` (item to be "picked" at pickup_loc),
      - a static target mapping `pahbmn6t <agent> <target_loc>`,
      - an intermediate predicate `eacj8ftl <agent>` (agent is "picked up"/ready),
      - and a goal `guby8wxu <agent>` (delivered).

    The typical sequence to achieve `guby8wxu agent` is:
      1. Move the marker to the pickup location where `pxch7pi8 agent pickup_loc` holds,
         apply `makpz2tc` (cost 1) which removes `pxch7pi8` and sets `eacj8ftl agent`.
      2. Move the marker to the agent's target location (given by static `pahbmn6t agent target_loc`)
         and apply `jg64qouf` (cost 1) which removes `eacj8ftl` and sets `guby8wxu agent`.

    This heuristic estimates the number of actions (moves and the two operator applications)
    required for all agents that appear in the goal (i.e., need to achieve `guby8wxu`),
    assuming the single marker performs actions sequentially. It is not admissible: it uses a
    greedy sequencing of tasks (nearest next-action selection) to estimate a realistic cost,
    which typically reduces node expansions for greedy best-first search.

    Assumptions
    -----------
    - Movement along `pvhp8m0p` is bidirectional (both movement actions in the domain allow travel
      along the static relation in each direction), so the static pvhp8m0p facts form an undirected graph.
    - Each agent in the goal must either already have `guby8wxu agent` in the state, or it must have
      either `eacj8ftl agent` (already picked up) or `pxch7pi8 agent <loc>` (an available pickup).
      If an agent lacks pxch7pi8 and eacj8ftl and is not yet at goal, the problem is unsolvable;
      in that case the heuristic returns a large finite cost.
    - Costs counted:
        - Each move along the graph from one location to another costs equal to the number of edges
          on the shortest path (each action jt9wnwgs/v90zoaro counts as 1).
        - Each application of makpz2tc or jg64qouf counts as 1 action.
    - The heuristic is designed to be efficient: distances are computed by BFS on the static graph
      and cached for repeated queries. The heuristic then simulates a greedy sequence of immediate
      actions (nearest pickup or nearest delivery) until all goal agents are delivered.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
    - Records the goal facts for quick goal checks.
    - Extracts the static mapping `pahbmn6t agent target_loc` into a dictionary agent -> target_loc.
    - Builds an undirected adjacency map of locations from static facts of the form
      `pvhp8m0p loc1 loc2` to support shortest-path BFS queries.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Quick goal check: if all goal facts are already satisfied in the given state return 0.
    2. Parse the current state to obtain:
       - the set of agents already with `guby8wxu` (delivered),
       - which agents have `eacj8ftl` (picked up, pending delivery),
       - for each agent the pickup location `pxch7pi8 agent pickup_loc` (if present),
       - the current marker location (the unique fact `v0yy74n7 <loc>`).
    3. For every agent required by the goal but not yet delivered:
       - If the agent has `eacj8ftl` then it only needs a delivery action at its static target.
       - Else if the agent has a `pxch7pi8` pickup fact then it needs both pickup and delivery.
       - Else the agent is unreachable (no pickup and not already picked) -> return a large cost.
    4. Compute shortest-path distances on the static location graph. Distances between arbitrary
       relevant nodes (current marker loc, pickup locs, and delivery locs) are computed by BFS
       (with caching per source) and considered infinite if nodes are disconnected.
    5. Simulate a greedy schedule for the marker:
       - Start at the marker's current location.
       - Repeatedly select the immediate next action with minimal immediate cost:
         * For any agent not yet picked: cost = distance(current_pos, pickup_loc) + 1 (makpz2tc).
         * For any agent already picked: cost = distance(current_pos, deliver_loc) + 1 (jg64qouf).
       - Execute that action in the simulated world: add the movement cost + 1 to total, update
         current_pos (to pickup_loc or deliver_loc), and update the agent's status (picked/delivered).
       - Continue until all goal agents are delivered.
    6. If any required travel distance is unreachable (disconnected) the heuristic returns a
       large finite cost; otherwise it returns the accumulated simulated action cost.

    This greedy nearest-next-action scheduling provides a more informed estimate than simply
    summing independent per-agent minimal costs (it accounts for travel reuse) while remaining
    fast to compute.
    """

    def __init__(self, task):
        # store goals (set of fact strings)
        self.goals = task.goals
        # parse static facts: build adjacency from pvhp8m0p and mapping for pahbmn6t
        self.adj = {}  # location -> set(neighbor_locs)
        self.agent_target = {}  # agent -> target_loc

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "pvhp8m0p" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)
            elif parts[0] == "pahbmn6t" and len(parts) >= 3:
                agent, target = parts[1], parts[2]
                # assume one static target per agent (domain static mapping)
                self.agent_target[agent] = target

        # cache for BFS distances: src -> {dest: distance}
        self._dist_cache = {}

    def _bfs_distances(self, src):
        """Return dict mapping each reachable node to its shortest distance from src (edges cost 1)."""
        if src in self._dist_cache:
            return self._dist_cache[src]
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._dist_cache[src] = dist
        return dist

    def _distance(self, a, b):
        """Return shortest-path distance between a and b (integer). If unreachable return None."""
        if a == b:
            return 0
        dmap = self._bfs_distances(a)
        return dmap.get(b, None)

    def __call__(self, node):
        state = node.state

        # 1) If all goal facts already true -> heuristic 0
        if self.goals <= state:
            return 0

        # Gather current marker location (v0yy74n7 <loc>), agent statuses and pickups.
        marker_loc = None
        picked_agents = set()  # agents with eacj8ftl
        delivered_agents = set()  # agents with guby8wxu
        pickup_loc_of = {}  # agent -> pickup_loc (from pxch7pi8)

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "v0yy74n7" and len(parts) >= 2:
                marker_loc = parts[1]
            elif pred == "eacj8ftl" and len(parts) >= 2:
                picked_agents.add(parts[1])
            elif pred == "guby8wxu" and len(parts) >= 2:
                delivered_agents.add(parts[1])
            elif pred == "pxch7pi8" and len(parts) >= 3:
                agent, loc = parts[1], parts[2]
                # store pickup location (domain likely has at most one)
                pickup_loc_of[agent] = loc

        # If marker missing in state, heuristic cannot simulate movement reliably; return large cost.
        if marker_loc is None:
            return UNREACHABLE_COST

        # Determine which agents are required by goals and which remain to be delivered.
        required_agents = []
        for goal_fact in self.goals:
            gparts = get_parts(goal_fact)
            if gparts[0] == "guby8wxu" and len(gparts) >= 2:
                required_agents.append(gparts[1])

        # Build tasks: for each required agent not yet delivered, decide needed steps.
        tasks = {}
        for agent in required_agents:
            if agent in delivered_agents:
                continue  # already done
            # target must exist in static mapping
            if agent not in self.agent_target:
                # if no static target mapping -> unreachable
                return UNREACHABLE_COST
            target = self.agent_target[agent]
            if agent in picked_agents:
                # no pickup needed, only delivery
                tasks[agent] = {"picked": True, "pickup_loc": None, "target": target}
            elif agent in pickup_loc_of:
                tasks[agent] = {"picked": False, "pickup_loc": pickup_loc_of[agent], "target": target}
            else:
                # no pickup available and not already picked -> unreachable
                return UNREACHABLE_COST

        # If no pending tasks, return 0 (should have been caught by goals check, but safe)
        if not tasks:
            return 0

        # Precheck connectivity: ensure all relevant locations are in adjacency or are isolated nodes known.
        relevant_locs = {marker_loc}
        for t in tasks.values():
            if t["pickup_loc"]:
                relevant_locs.add(t["pickup_loc"])
            relevant_locs.add(t["target"])

        # If any relevant loc not in adjacency dict, it's isolated but may still be reachable if it's same node.
        # BFS distance will yield only self-distance; if there are edges missing and needed travel >0 -> unreachable.
        # We'll detect unreachable during distance queries.

        # Greedy simulation: repeatedly execute the nearest immediate action (pickup or delivery)
        current_pos = marker_loc
        total_cost = 0

        # Local copy of tasks modifiable during simulation
        pending = {agent: dict(info) for agent, info in tasks.items()}

        while pending:
            best_agent = None
            best_action = None  # "pickup" or "deliver"
            best_cost = None
            best_move_dist = None
            # Evaluate immediate actions
            for agent, info in pending.items():
                if not info["picked"]:
                    pickup = info["pickup_loc"]
                    d = self._distance(current_pos, pickup)
                    if d is None:
                        # unreachable route to pickup
                        return UNREACHABLE_COST
                    cost_here = d + 1  # move + makpz2tc
                    if best_cost is None or cost_here < best_cost:
                        best_cost = cost_here
                        best_agent = agent
                        best_action = "pickup"
                        best_move_dist = d
                else:
                    target = info["target"]
                    d = self._distance(current_pos, target)
                    if d is None:
                        # unreachable route to delivery
                        return UNREACHABLE_COST
                    cost_here = d + 1  # move + jg64qouf
                    if best_cost is None or cost_here < best_cost:
                        best_cost = cost_here
                        best_agent = agent
                        best_action = "deliver"
                        best_move_dist = d

            # Apply the chosen immediate action in simulation
            if best_agent is None:
                # something went wrong; safeguard
                return UNREACHABLE_COST

            # Add movement + operator application cost
            total_cost += best_move_dist + 1

            if best_action == "pickup":
                # move to pickup loc and do makpz2tc; mark picked
                current_pos = pending[best_agent]["pickup_loc"]
                pending[best_agent]["picked"] = True
                # pickup_loc no longer needed
                pending[best_agent]["pickup_loc"] = None
            else:
                # deliver: move to target and do jg64qouf; remove task
                current_pos = pending[best_agent]["target"]
                del pending[best_agent]

            # Continue until all tasks processed

            # Safety cap: if total_cost grows beyond UNREACHABLE_COST, stop and return large cost
            if total_cost >= UNREACHABLE_COST:
                return UNREACHABLE_COST

        # Return the simulated greedy total cost (integer)
        return int(total_cost)