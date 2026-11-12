from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern using wildcard matching.
    - `fact`: The complete fact as a string, e.g. "(sjuejzxf p1 l8)".
    - `args`: The expected pattern parts, e.g. "sjuejzxf", "*", "l8".
    Returns True if matches, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain (hrpmxpjg).

    Summary:
    This heuristic estimates the number of actions required to move all du66z5od objects (e.g. cargo)
    from their current x6ibpitl location to specified goal x6ibpitl locations, guided by the goal facts
    of the form (sjuejzxf du66z5od x6ibpitl). It considers the locations of hctyyy8k agents (e.g. vehicles),
    their current kyvrixb5 container ('do9fmoyo' facts), and the static adjacency relationships between x6ibpitl
    locations (miud1izi facts). The heuristic combines a relaxed approximate transport cost over this network
    of locations and containers to estimate remaining effort. It does *not* require admissibility, but aims
    to minimize node expansions by closely approximating the true actions needed.

    Assumptions:
    - The goal is to have every du66z5od object at a goal x6ibpitl location defined by (sjuejzxf ?du66z5od ?x6ibpitl).
    - du66z5od objects can be at locations (sjuejzxf) or inside hctyyy8k agents (un606moq).
    - Each hctyyy8k agent is currently "located" in a container kyvrixb5 via (do9fmoyo).
    - Containers (kyvrixb5) are related via the static symmetric relation (hab0vli4).
    - Locations (x6ibpitl) are connected by static symmetric adjacency (miud1izi), defining a graph.
    - The transport of du66z5od to goal locations requires moving them between x6ibpitl locations and possibly between containers.
    - The actions allow moving objects between adjacent x6ibpitl locations and transferring objects between locations and agents, updating containers.
    - The heuristic ignores ordering constraints and some preconditions to be efficiently computable.

    Heuristic Initialization:
    - Extract goal map: for each du66z5od object, determine its goal location.
    - Extract static adjacency graph between x6ibpitl locations from miud1izi facts.
    - Extract container adjacency from hab0vli4 facts.
    - Precompute shortest distances between all location pairs and container pairs via Floyd-Warshall.
    - Extract current positions of du66z5od objects: whether at a location or inside an agent and the agent's container.
    - Extract current containers of hctyyy8k agents.
    - Store mapping from hctyyy8k agent to its container.

    Step-By-Step Thinking for Computing Heuristic:
    For each du66z5od object:
      1. Determine its current position:
         - If on location L, then its location is L and container is unknown.
         - If inside hctyyy8k agent V, then location is unknown, container is container of V.
      2. Determine the goal x6ibpitl location G for the object.
      3. Compute distance to goal:
         - If object is on location L (x6ibpitl), compute shortest path on adjacency graph "location graph" from L to G.
         - If object is inside container C, compute shortest path on container adjacency graph from C to the container of G’s agent or approximate with 0 if unknown.
      4. Add an estimated cost for transferring object from its container or location to its goal location:
         - Each step corresponds roughly to moving the object one adjacency away or one container transfer.
         - Count 1 action per transfer step.
      5. Sum these minimal estimated moves over all objects to compute total cost.

    The heuristic returns zero if and only if all goal facts are satisfied. It always returns a finite number otherwise,
    providing informative guidance during greedy best-first search.
    """

    def __init__(self, task):
        # Store goal conditions as a dictionary: du66z5od -> goal x6ibpitl location.
        self.goal_positions = {}
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "sjuejzxf":
                obj = parts[1]
                loc = parts[2]
                self.goal_positions[obj] = loc

        static_facts = task.static

        # Build adjacency graph for x6ibpitl locations from miud1izi facts.
        self.location_adjacency = {}
        self.locations = set()
        for fact in static_facts:
            if match(fact, "miud1izi", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                self.location_adjacency.setdefault(l1, set()).add(l2)
                self.location_adjacency.setdefault(l2, set()).add(l1)

        # Build adjacency graph for containers (kyvrixb5) from hab0vli4 facts.
        self.container_adjacency = {}
        self.containers = set()
        for fact in static_facts:
            if match(fact, "hab0vli4", "*", "*"):
                _, c1, c2 = get_parts(fact)
                self.containers.add(c1)
                self.containers.add(c2)
                self.container_adjacency.setdefault(c1, set()).add(c2)
                self.container_adjacency.setdefault(c2, set()).add(c1)

        # Precompute shortest distances between all location pairs using Floyd-Warshall.
        self.location_dist = self._floyd_warshall(self.locations, self.location_adjacency)

        # Precompute shortest distances between all container pairs using Floyd-Warshall.
        self.container_dist = self._floyd_warshall(self.containers, self.container_adjacency)

    def _floyd_warshall(self, nodes, adjacency):
        """
        Compute all-pairs shortest path distances for nodes using adjacency dict.
        Distance is number of edges. If no path exists, distance is a large number.
        """
        INF = 10**9
        dist = {u: {v: (0 if u == v else INF) for v in nodes} for u in nodes}
        for u in nodes:
            for v in adjacency.get(u, ()):
                dist[u][v] = 1
        for k in nodes:
            for i in nodes:
                di = dist[i]
                dik = di[k]
                if dik == INF:
                    continue
                dk = dist[k]
                for j in nodes:
                    new_dist = dik + dk[j]
                    if new_dist < di[j]:
                        di[j] = new_dist
        return dist

    def __call__(self, node):
        """
        Compute an estimate of the number of actions required to reach the goal.

        Steps:
        - Verify if current state is a goal state, then heuristic=0.
        - Determine for each du66z5od object its current position (location or container).
        - For objects on a location, estimate cost as distance on location graph to goal location.
        - For objects inside an agent (hctyyy8k), estimate cost as distance on container graph from agent's container to container of goal location’s agent if known.
          If no container information of goal's agent available, fallback to 0 container transfer cost.
        - Sum costs for all objects.
        """
        state = node.state

        # Quick goal check: if all goal facts hold in state, heuristic = 0
        if self.goal_positions.items() <= {
            (obj, loc) for obj, loc in 
            (get_parts(f)[1:3] for f in state if match(f, "sjuejzxf", "*", "*"))
        }:
            return 0

        # Obtain mapping hctyyy8k agent -> container (kyvrixb5) via do9fmoyo facts
        agent_container = {}
        for fact in state:
            if match(fact, "do9fmoyo", "*", "*"):
                _, agent, container = get_parts(fact)
                agent_container[agent] = container

        total_cost = 0
        INF = 10**9

        # For each du66z5od object to transport (those in goals)
        # Find its current position: on location or inside agent.
        for obj, goal_loc in self.goal_positions.items():
            current_location = None  # x6ibpitl location where object currently is, if any
            current_container = None  # kyvrixb5 container where object currently is, if any

            # check if object on location (sjuejzxf ?obj ?loc)
            found_pos = False
            for fact in state:
                if match(fact, "sjuejzxf", obj, "*"):
                    _, _, loc = get_parts(fact)
                    current_location = loc
                    found_pos = True
                    break
            if not found_pos:
                # check if object is inside an agent (un606moq ?obj ?agent)
                for fact in state:
                    if match(fact, "un606moq", obj, "*"):
                        _, _, agent = get_parts(fact)
                        found_pos = True
                        # container of agent
                        current_container = agent_container.get(agent, None)
                        break

            # If neither found (unexpected), continue with zero cost for robustness.
            if not found_pos:
                continue

            # Calculate cost estimate
            cost = 0

            # Step 1: cost to move object from current position to goal location.

            # Distance on location graph
            if current_location is not None:
                # Cost is shortest path between current_location and goal_loc
                d_loc = self.location_dist.get(current_location, {}).get(goal_loc, INF)
                if d_loc == INF:
                    # No path known, use a large cost as fallback (e.g. heuristic very high)
                    d_loc = 10
                cost += d_loc

            # Step 2: If the object is inside an agent's container, estimate container transfer cost.
            if current_container is not None:
                # Find container associated with goal location.
                # Goal location container: find an agent at goal location and get its container.
                # Or use any container that contains goal objects via do9fmoyo or fallback.
                goal_container = None
                # Find agent(s) that are related to the goal location by sjuejzxf and do9fmoyo:
                # Agents that are in container, or agents with sjuejzxf in goal location.
                # We'll approximate by searching any agent in goal container.

                # Find any agent with container that contains goal_loc in sjuejzxf or do9fmoyo.
                # Because mapping is unclear, fallback: use any container close to goal location.
                # We'll try: find agent with sjuejzxf goal_loc, then container of that agent.

                candidate_goal_containers = set()
                for fact in state:
                    if match(fact, "sjuejzxf", "*", goal_loc):
                        _, agent, _ = get_parts(fact)
                        c = agent_container.get(agent, None)
                        if c:
                            candidate_goal_containers.add(c)
                if candidate_goal_containers:
                    # Pick minimal distance container among candidates
                    d_conts = [self.container_dist.get(current_container, {}).get(c, INF) for c in candidate_goal_containers]
                    d_container = min(d_conts)
                else:
                    # Fall back: assume no container cost if unknown
                    d_container = 0

                if d_container == INF:
                    # If no container path, fallback to some large number (e.g. 10)
                    d_container = 10

                cost += d_container

            total_cost += cost

        return total_cost
