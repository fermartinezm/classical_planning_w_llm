from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

INF = 10 ** 6


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its components."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (unkhrp2l) domain.

    The domain contains a movable token (v0yy74n7 at some location) that can be
    moved along connections (pvhp8m0p) between locations. For each goal object
    (an element of type jxhpkqr5) we must achieve predicate (guby8wxu OBJ).
    Producing guby8wxu for an object OBJ requires applying action jg64qouf,
    which needs (eacj8ftl OBJ) and the token to be at a location L such that
    (pahbmn6t OBJ L) holds. The predicate (eacj8ftl OBJ) can be produced by
    applying action makpz2tc at a location L' where the dynamic fact
    (pxch7pi8 OBJ L') currently holds and the token is at L'.

    Thus, to get guby8wxu for an agent OBJ we generally need to:
      - (if not already present) visit its pxch location L_px and run makpz2tc
        (cost: moves to L_px + 1 action), producing eacj8ftl; then
      - visit some pahbmn6t location L_pa (there may be multiple in static info)
        and run jg64qouf (cost: moves to L_pa + 1 action).

    The heuristic estimates the minimal number of token moves plus the number
    of makpz2tc / jg64qouf actions required to achieve all remaining goal
    guby8wxu predicates. It computes a greedy route that at every step goes
    to the nearest needed action location from the current token location.
    This is not admissible but is informative and efficient to compute.

    Assumptions
    -----------
    - Movement of the token is possible along pvhp8m0p edges. For planning
      we treat pvhp8m0p edges as undirected for movement because the two
      movement actions together allow traversing an edge in either direction
      if the corresponding pvhp8m0p fact exists in either orientation.
    - Each agent that must reach guby8wxu will have at least one pahbmn6t
      location in the static facts; otherwise the agent cannot be completed.
    - The dynamic fact pxch7pi8 (agent -> location) indicates where makpz2tc
      can be applied for that agent. If neither eacj8ftl nor pxch7pi8 for an
      agent exists in the current state, the agent is considered unreachable
      by this heuristic and a large penalty is returned.
    - All action costs are counted as 1; token moves (jt9wnwgs/v90zoaro) are
      counted per edge traversed.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - The set of goal agents that require guby8wxu (self.goal_agents).
    - For each agent, the set of its pahbmn6t locations from the static facts
      (self.agent_pa_locations).
    - The movement graph extracted from static pvhp8m0p facts as an adjacency
      mapping (self.adjacency). Edges are added in both directions for BFS.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal guby8wxu facts already hold in the state -> return 0.
    2. Find the current token location by locating the v0yy74n7 fact in the
       state. If none exists -> return a large penalty (INF).
    3. For each goal agent that is not yet guby8wxu in the state:
       - If eacj8ftl agent is present in the state:
           -> The agent needs a single visit to any pahbmn6t location + 1 action.
       - Else if pxch7pi8 agent L_px is present in the state:
           -> The agent needs a visit to L_px + 1 action, then a visit to one
              of the agent's pahbmn6t locations + 1 action.
       - Else:
           -> The agent appears unreachable under the domain model; return INF.
    4. Compute an approximate minimal total by simulating a greedy route:
       - Start at the current token location.
       - Repeatedly pick the next required step (either a PX or a PA step for
         some agent) that is nearest (shortest path in the adjacency graph)
         from the current token location.
       - Add the travel distance (number of moves) to reach that location and
         add 1 for executing the corresponding action (makpz2tc or jg64qouf).
       - If a PX step is executed, mark the agent as now having eacj8ftl so
         its next step becomes a PA step; the location chosen for that PA
         step will be the currently nearest pahbmn6t location when the PA
         step gets selected.
    5. Return the accumulated total cost as the heuristic estimate.

    This greedy route gives a reasonable, efficiently computable estimate of
    the number of actions needed to reach the goal from the given state.
    """

    def __init__(self, task):
        # Store goals (set of strings) and extract the agent names mentioned
        self.goals = task.goals
        # Extract goal agents that must have (guby8wxu AGENT)
        goal_agents = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "guby8wxu" and len(parts) >= 2:
                goal_agents.add(parts[1])
        self.goal_agents = goal_agents

        # Static facts: collect pvhp8m0p edges and pahbmn6t mappings
        self.adjacency = defaultdict(set)  # location -> set(neighbor locations)
        self.agent_pa_locations = defaultdict(set)  # agent -> set(locations)
        for fact in task.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "pvhp8m0p" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                # add both directions as movement is possible when pvhp exists in either orientation
                self.adjacency[a].add(b)
                self.adjacency[b].add(a)
            elif pred == "pahbmn6t" and len(parts) >= 3:
                agent, loc = parts[1], parts[2]
                self.agent_pa_locations[agent].add(loc)

    def bfs_distances(self, start):
        """
        BFS to compute shortest distances (in number of moves) from start to all
        reachable locations in the static adjacency graph.
        Returns a dict location -> distance.
        """
        distances = {}
        if start not in self.adjacency:
            # Still include the node itself with distance 0 (isolated location).
            distances[start] = 0
            return distances
        q = deque([start])
        distances[start] = 0
        while q:
            u = q.popleft()
            du = distances[u]
            for v in self.adjacency[u]:
                if v not in distances:
                    distances[v] = du + 1
                    q.append(v)
        return distances

    def __call__(self, node):
        state = node.state

        # 1) If goal already reached, heuristic is 0
        # Check all goal guby8wxu facts are present
        all_goals_present = True
        for agent in self.goal_agents:
            if f"(guby8wxu {agent})" not in state:
                all_goals_present = False
                break
        if all_goals_present:
            return 0

        # 2) Find token location
        token_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "v0yy74n7" and len(parts) >= 2:
                token_loc = parts[1]
                break
        if token_loc is None:
            return INF

        # 3) Build per-agent step sequences
        agents_steps = {}  # agent -> list of steps, where step is dict {'type':'PX'/'PA', 'loc':... or None}
        for agent in self.goal_agents:
            # Skip already satisfied agents
            if f"(guby8wxu {agent})" in state:
                continue

            # check if eacj8ftl present
            has_eac = f"(eacj8ftl {agent})" in state

            # find pxch7pi8 location for agent in current state (could be absent)
            px_loc = None
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "pxch7pi8" and len(parts) >= 3 and parts[1] == agent:
                    px_loc = parts[2]
                    break

            # find possible pahbmn6t (PA) locations from static info
            pa_locs = self.agent_pa_locations.get(agent, set())

            # If agent lacks eac and also has no px in state, unreachable by planner actions
            if (not has_eac) and (px_loc is None):
                return INF

            if not pa_locs:
                # No pahbmn6t locations specified in static facts -> unreachable target
                return INF

            steps = []
            if not has_eac:
                # need PX step first at px_loc (must exist)
                steps.append({"type": "PX", "loc": px_loc})
                # then PA step; we do not fix which loc now; pick nearest when executing
                steps.append({"type": "PA", "locs": set(pa_locs)})
            else:
                # only PA step needed
                steps.append({"type": "PA", "locs": set(pa_locs)})

            agents_steps[agent] = steps

        # If no agents to handle (all goals satisfied), return 0
        if not agents_steps:
            return 0

        # 4) Greedy route simulation: iteratively go to nearest next required step
        total_cost = 0
        current_loc = token_loc

        # Precompute BFS distances from current_loc; will recompute when current changes
        distances_cache = {}

        def get_distance(src, dst):
            # Return shortest distance between src and dst in movement graph (number of moves).
            if src == dst:
                return 0
            if src not in distances_cache:
                distances_cache[src] = self.bfs_distances(src)
            dmap = distances_cache[src]
            return dmap.get(dst, INF)

        # Continue until all agents' steps are done
        while True:
            # gather candidate next steps: for each agent, take its next step
            candidates = []
            for agent, steps in agents_steps.items():
                if not steps:
                    continue
                next_step = steps[0]
                if next_step["type"] == "PX":
                    loc = next_step["loc"]
                    dist = get_distance(current_loc, loc)
                    candidates.append((dist, agent, "PX", loc))
                else:  # "PA"
                    # choose nearest PA location for this agent from current_loc
                    pa_locs = next_step["locs"]
                    best_loc = None
                    best_dist = INF
                    for loc in pa_locs:
                        dist = get_distance(current_loc, loc)
                        if dist < best_dist:
                            best_dist = dist
                            best_loc = loc
                    candidates.append((best_dist, agent, "PA", best_loc))

            if not candidates:
                break  # all done

            # pick candidate with minimal travel distance; tie-break by agent name
            candidates.sort(key=lambda x: (x[0], x[1]))
            dist, agent, step_type, step_loc = candidates[0]

            if dist >= INF:
                return INF

            # travel cost + action cost (1)
            total_cost += dist + 1

            # update current location
            current_loc = step_loc

            # Invalidate BFS cache for new source (we'll lazily compute when needed)
            # We keep existing cache entries; distances_cache keyed by source.

            # Pop the finished step and handle updates
            finished_step = agents_steps[agent].pop(0)
            if finished_step["type"] == "PX":
                # executing makpz2tc produces eacj8ftl and removes pxch7pi8 dynamically.
                # Next step for this agent must be a PA step, which is already present.
                # Nothing else to do here.
                pass
            else:
                # finished PA step by executing jg64qouf => agent achieved guby8wxu
                # Remove all remaining steps (should be none), and mark agent as done.
                agents_steps[agent] = []

            # If all agents have no remaining steps, break
            if all(len(s) == 0 for s in agents_steps.values()):
                break

        return total_cost