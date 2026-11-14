from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 10 ** 6


def get_parts(fact):
    """Return the token list of a fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (unkhrp2l) domain.

    This heuristic estimates the number of actions required to achieve all
    goal facts of the form (guby8wxu <agent>). In this domain an agent
    reaches its goal when:
      1) There exists a static fact (pahbmn6t <agent> <loc>) giving the
         agent-specific target location.
      2) The robot is at some location and the operator jg64qouf is applied
         when (eacj8ftl <agent>) is true and the robot is at the agent's
         target location; jg64qouf consumes eacj8ftl and produces guby8wxu.
      3) eacj8ftl is produced by makpz2tc when the robot is at the location
         where (pxch7pi8 <agent> <loc_px>) holds (the operator converts
         pxch7pi8 into eacj8ftl).
      4) The robot moves between locations along facts (pvhp8m0p <a> <b>).
         There are two movement operators in the domain (jt9wnwgs and
         v90zoaro); together they allow movement along pairs given by
         pvhp8m0p in either direction. Each movement is one action.

    Assumptions
    -----------
    - The planner represents facts as strings like "(predicate arg1 arg2)".
    - Static connectivity facts are given by (pvhp8m0p loc1 loc2) and are
      treated as undirected links for movement (movement operators allow
      traveling along either ordering).
    - For each goal agent there exists a static (pahbmn6t agent loc_goal).
      If such static info is missing the goal is treated as unreachable.
    - If an agent neither has eacj8ftl nor a pxch7pi8 fact in the current
      state, there is no way to produce eacj8ftl and the agent is treated
      as unreachable.
    - The heuristic is not required to be admissible; it sums the individual
      minimal costs for each goal agent starting from the robot's current
      location. This may over-estimate because the same robot trip may
      serve multiple agents, but provides a strong domain-specific guidance.

    Heuristic Initialization
    ------------------------
    - Extract and store the set of goal facts (strings).
    - From the static facts:
        * Build the location adjacency graph from (pvhp8m0p loc1 loc2).
          Since movement operators allow either orientation, adjacency is
          made bidirectional (both loc1->loc2 and loc2->loc1).
        * Extract the mapping agent -> target location from (pahbmn6t agent loc).
    - Precompute shortest distances (in number of movement actions) between
      all pairs of locations using BFS from every location present in the graph.
      Distances that are unreachable are represented by a large finite value.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goals are already satisfied in the state, return 0.
    2. Parse the state to obtain:
         - robot_loc: the unique location x such that (v0yy74n7 x) is true.
         - which agents currently have eacj8ftl (a set).
         - for every agent, whether there's a pxch7pi8 fact and the associated
           location px_loc (mapping agent -> px_loc).
         - which guby8wxu facts are already true (but goals handle that).
    3. For each goal (guby8wxu <agent>) that is not yet true in the state,
       compute the minimal action cost assuming independent servicing starting
       from the current robot location:
         - If the agent already has eacj8ftl:
             * cost = distance(robot_loc, loc_goal) + 1
               (the +1 is the application of jg64qouf).
         - Else if the agent has a pxch7pi8 at px_loc:
             * cost = distance(robot_loc, px_loc)       # move to px_loc
                      + 1                               # makpz2tc (pxch->eacj8ftl)
                      + distance(px_loc, loc_goal)      # move to target
                      + 1                               # jg64qouf (eacj8ftl->guby8wxu)
             * Note: if px_loc == loc_goal the second travel distance is zero,
               and both operators can be applied consecutively at the same location.
         - Else:
             * The agent cannot be advanced toward its goal in the current
               state (no pxch7pi8 and no eacj8ftl) => treat as unreachable and
               return a large heuristic (INF) to steer search away.
    4. Sum the costs for all unsatisfied goal agents and return the sum.
       This sum is finite if every needed distance exists in the connectivity
       graph and agents are serviceable; otherwise INF is returned.

    The heuristic is designed to be quick to compute (BFS precomputation on
    the static graph, then simple lookups per node) and to provide strong
    guidance by accounting for movement costs and necessary operator steps.
    """

    def __init__(self, task):
        # Store goals (set of strings like "(guby8wxu ps0y2bwz)").
        self.goals = task.goals
        # Static facts (frozenset of strings).
        static_facts = task.static

        # Build undirected adjacency for locations from (pvhp8m0p a b).
        self.adj = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "pvhp8m0p":
                a, b = parts[1], parts[2]
                locations.add(a)
                locations.add(b)
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

        # Extract agent -> required target location from (pahbmn6t agent loc).
        self.agent_target = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "pahbmn6t":
                agent, loc = parts[1], parts[2]
                self.agent_target[agent] = loc
                locations.add(loc)

        # Ensure every location present in adjacency dict.
        for loc in locations:
            self.adj.setdefault(loc, set())

        # Precompute shortest movement distances between locations using BFS.
        # distances[src][dst] = minimal number of movement actions to go src->dst.
        self.distances = {}
        for src in self.adj:
            dist = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            self.distances[src] = dist

    def __call__(self, node):
        state = node.state

        # If the node already satisfies all goals, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse state facts.
        robot_loc = None
        agents_with_eacj = set()
        agent_px = {}  # agent -> px location (from pxch7pi8 facts)
        present_guby = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "v0yy74n7":  # robot at location
                robot_loc = parts[1]
            elif pred == "eacj8ftl":
                agents_with_eacj.add(parts[1])
            elif pred == "pxch7pi8":
                agent_px[parts[1]] = parts[2]
            elif pred == "guby8wxu":
                present_guby.add(parts[1])

        if robot_loc is None:
            # No robot location in the state -> cannot move; return large cost.
            return INF

        total_cost = 0

        # For each goal agent, estimate minimal actions required.
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] != "guby8wxu":
                # Only handle guby8wxu goals (domain uses that form).
                continue
            agent = parts[1]

            # If already attained in state, no cost.
            if f"(guby8wxu {agent})" in state or agent in present_guby:
                continue

            # Target location must be known (static). If missing -> unreachable.
            if agent not in self.agent_target:
                return INF
            target_loc = self.agent_target[agent]

            # If agent already has eacj8ftl, only need to move robot to target and apply jg64qouf.
            if agent in agents_with_eacj:
                # distance from robot to target
                d = self.distances.get(robot_loc, {}).get(target_loc, INF)
                if d >= INF:
                    return INF
                total_cost += d + 1  # +1 for jg64qouf
                continue

            # Else agent needs pxch7pi8 to be converted first.
            if agent not in agent_px:
                # No way to create eacj8ftl for this agent in current state.
                return INF
            px_loc = agent_px[agent]

            # Distances: robot -> px_loc, px_loc -> target_loc
            d1 = self.distances.get(robot_loc, {}).get(px_loc, INF)
            d2 = self.distances.get(px_loc, {}).get(target_loc, INF)
            if d1 >= INF or d2 >= INF:
                return INF

            # Actions:
            # - d1 moves to px_loc
            # - 1 makpz2tc to create eacj8ftl
            # - d2 moves to target_loc
            # - 1 jg64qouf to produce guby8wxu
            total_cost += d1 + 1 + d2 + 1

        return total_cost if total_cost < INF else INF