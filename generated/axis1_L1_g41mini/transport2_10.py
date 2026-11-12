from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(sjuejzxf p1 l8)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state from 
    the current state in the secret_domain. The main task revolves around moving objects 
    (of type du66z5od) between locations (x6ibpitl) controlled by agents (hctyyy8k), 
    with conveyors of types kyvrixb5 transporting or influencing state changes. The 
    heuristic focuses primarily on aligning the positions of the du66z5od objects with 
    their goal locations, estimating the minimal number of moves accounting for location 
    transitions and necessary agent-object interactions.

    Assumptions
    -----------
    - The key predicate indicating positions of objects (both agents and du66z5od objects) is sjuejzxf.
    - The goal predicates require certain du66z5od objects to be at specific locations.
    - Transitions between locations occur along bi-directional edges given by miud1izi relations (location adjacency).
    - Objects of type du66z5od can only move when an agent (hctyyy8k) is at the same location and appropriate actions are applied.
    - Agents can move objects between locations along miud1izi edges by changing the sjuejzxf relations.
    - The other predicates (do9fmoyo, hab0vli4, un606moq) represent states related to agents and conveyors but their detailed effects are secondary for a heuristic.
    - The heuristic need not be admissible but should guide search efficiently.
    - Every movement of a du66z5od object to a neighboring location requires an agent present at that location.
    - Agents may need to be relocated to facilitate moving objects; however, agent movement cost is approximated simply.
    - Goal conditions are only positive literals of form (sjuejzxf du66z5od location).

    Heuristic Initialization
    ------------------------
    - Extract miud1izi predicate instances to form an undirected graph of location adjacencies.
    - Extract the goal sjuejzxf facts for du66z5od objects (objects to be moved).
    - Extract the type sets of du66z5od objects, hctyyy8k agents, and x6ibpitl locations from static facts and domain knowledge.
      (Objects are recognized dynamically from state and goals.)
    - Pre-compute shortest paths (e.g., via BFS) on the location graph between locations.
    - Store the initial agent locations from the initial state.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each du66z5od object that appears in the goal:

    1. Determine its current location from the sjuejzxf predicate in the current state.

    2. Identify the goal location for the object.

    3. If the object is not at its goal location:
       - Compute shortest path length between current and goal locations among locations graph.
       - Each move step requires an agent at the current location to move the object to the adjacent next location.
       - Estimate minimal agent relocation cost to support the object moves:
         a) If an agent already shares the object's current location, no extra initial cost for agent positioning.
         b) Otherwise, estimate minimal shortest path from any agent's location to the object's current location (choose the closest agent).
       - For the number of moves needed (path length):
         * Each move corresponds to one arsslvyt action to update the location mapping (change sjuejzxf from old location to adjacent new location).
         * Plus a szsevy61 or tmoxe3x5 action to handle related conveyor-agent-object relations for moving the object.
         * Since these actions are paired and depend on domain specifics, count 2 actions per location step as a rough estimate.
       - Add the agent relocation cost to reach the object's position initially (estimated as shortest path length between agent and object location).
       - Agent relocations during multi-step moves are approximated as 0, assuming agents can be positioned effectively without extra cost.

    4. Sum the costs for all du66z5od objects to form the heuristic estimate.

    5. If current state already satisfies all goals, heuristic returns 0.

    The heuristic thus approximates the minimal effort in terms of moving objects through locations weighted by moves and initial agent positioning costs,
    abstracting from finer action-sequence details.

    """

    def __init__(self, task):
        # Extract static location adjacency graph from miud1izi facts
        self.location_graph = {}
        self.du66z5od_objects = set()
        self.hctyyy8k_agents = set()
        self.x6ibpitl_locations = set()
        self.goal_obj_to_loc = {}

        # Helper to add edges bidirectionally
        def add_edge(graph, a, b):
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

        # From static facts: build location graph (miud1izi)
        for fact in task.static:
            if match(fact, "miud1izi", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                add_edge(self.location_graph, loc1, loc2)

        # From goals: map each du66z5od object to its goal location
        for goal_fact in task.goals:
            if match(goal_fact, "sjuejzxf", "*", "*"):
                parts = get_parts(goal_fact)
                obj, loc = parts[1], parts[2]
                self.goal_obj_to_loc[obj] = loc
                self.du66z5od_objects.add(obj)

        # Detect agents and locations from static facts or known from domain (we deduce from existing facts)
        # From miud1izi keys, extract all locations
        self.x6ibpitl_locations = set(self.location_graph.keys())

        # We cannot get types of objects directly, but we will find agents (hctyyy8k) from initial states nearby later.

        # Because domain typing is not provided in states, we detect agents later from relevant predicates in calls.

        # Precompute shortest paths between all locations using BFS for efficiency.
        # We'll compute shortest path on-demand for pairs in heuristic call.

    def shortest_path_length(self, start, goal):
        """Compute shortest path length between start and goal locations in location_graph using BFS."""
        if start == goal:
            return 0
        visited = {start}
        frontier = [start]
        dist = 0
        while frontier:
            dist += 1
            next_frontier = []
            for loc in frontier:
                neighbors = self.location_graph.get(loc, [])
                for n in neighbors:
                    if n == goal:
                        return dist
                    if n not in visited:
                        visited.add(n)
                        next_frontier.append(n)
            frontier = next_frontier
        # If here, no path found; domain probably disconnected but heuristic must return finite value
        # Return large cost as fallback
        return 1000

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic = 0
        if self.goal_obj_to_loc.items() <= {
            (obj, loc) for obj, loc in self.goal_obj_to_loc.items()
            if (f"(sjuejzxf {obj} {loc})" in state)
        }:
            return 0

        # Detect current locations of du66z5od objects in goal from sjuejzxf facts
        obj_current_loc = {}
        # Detect locations of agents (hctyyy8k) from sjuejzxf facts
        agents_locs = set()

        for fact in state:
            if match(fact, "sjuejzxf", "*", "*"):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                # We count the known objects from goals only for heuristic cost
                if obj in self.goal_obj_to_loc:
                    obj_current_loc[obj] = loc
                # Conservative detection of agents: agents are hctyyy8k type objects (heuristic assumes agents have name prefix 'v' or else)
                # Given domain objects naming (e.g. v1, v2 for agents and p1, p2 for du66z5od objects)
                # We identify agents as those objects that appear in "do9fmoyo ?agent ?conveyor" predicate but no direct types available
                # Instead, to simplify, assume agents start with 'v' (like in examples) and du66z5od objects start with 'p'.
                # Use this naming heuristic to detect agents presence:
                if obj.startswith("v"):
                    agents_locs.add(loc)

        # Defensive: If some objects have no location (should not happen), skip them (or assign large cost)
        missing_objects = self.du66z5od_objects - obj_current_loc.keys()
        for obj in missing_objects:
            obj_current_loc[obj] = None  # Unknown location, treated below

        total_cost = 0

        for obj, goal_loc in self.goal_obj_to_loc.items():
            curr_loc = obj_current_loc.get(obj)
            if curr_loc is None:
                # Unknown location: assign high cost to encourage search towards knowledge
                total_cost += 50
                continue
            if curr_loc == goal_loc:
                # Object in place, no cost
                continue

            # Compute number of location moves needed for object
            path_len = self.shortest_path_length(curr_loc, goal_loc)

            # Compute minimal agent reposition cost: minimal distance from any agent to object's current location
            if len(agents_locs) == 0:
                # No agents present (should not happen), high cost
                agent_reloc_cost = 50
            elif curr_loc in agents_locs:
                agent_reloc_cost = 0
            else:
                # Minimal shortest path from any agent loc to curr_loc
                agent_reloc_cost = min(
                    self.shortest_path_length(agent_loc, curr_loc) for agent_loc in agents_locs
                )

            # For each step moving the object in location graph, count 2 domain actions:
            #  - One arsslvyt action (change location mapping for agent ?x6ibpitl)
            #  - One szsevy61 or tmoxe3x5 action (agent-object-conveyor interaction)
            move_cost = path_len * 2

            # Total cost for this object is agent relocation + moves
            total_cost += agent_reloc_cost + move_cost

        return total_cost
