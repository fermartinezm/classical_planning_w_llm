from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(kut2ksnk p1 f6)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to reach the goal state from a given
    state by analyzing the partial progress of nxbd1hc3 objects (entities) through a two-stage
    transformation process tracked by predicates f2xo7hb6 and a7sb56om, and the spatial position
    of t4rdxh32 objects flagged by omggbn9z. It incorporates movement costs between t4rdxh32 
    locations connected by acovodrf.

    Assumptions
    -----------
    - Each nxbd1hc3 object must reach goal condition (a7sb56om) for the goal state.
    - A two-step "transformation" applies to each nxbd1hc3 from neither f2xo7hb6 nor a7sb56om,
      to f2xo7hb6 true, to a7sb56om true.
    - Actions diwlmgg9 and y7nik9q2 represent the transitions for nxbd1hc3 objects,
      dependent on the presence of omggbn9z on some t4rdxh32 object.
    - The omggbn9z predicate marks the current location(s) of the "agent" or "control point".
    - The domain allows moving the omggbn9z marker between connected t4rdxh32 locations via qs8pdrqj and i0veo5kv.
    - Movement costs between locations follow the shortest path along acovodrf edges.
    - The heuristic is not necessarily admissible, but aims for accuracy and efficiency.

    Heuristic Initialization
    ------------------------
    - Extract the set of nxbd1hc3 objects from static facts known via kut2ksnk / klw11kjf predicates.
    - Extract all t4rdxh32 locations.
    - Build an undirected graph of locations using acovodrf static facts to compute shortest paths.
    - From goals, identify nxbd1hc3 objects which must satisfy (a7sb56om).
    
    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which nxbd1hc3 objects still need to be fully "processed" to reach the goal
       (i.e., do not have (a7sb56om)).
    2. For each such nxbd1hc3 object:
       a. If it lacks (f2xo7hb6), it needs to apply diwlmgg9 first (requires the nxbd1hc3 located on
          some t4rdxh32 location where (kut2ksnk) holds, and the agent needs to be there).
       b. After (f2xo7hb6) holds but no (a7sb56om), it needs to apply y7nik9q2 (agent and object
          also co-located on t4rdxh32 location where (klw11kjf) holds).
    3. The agent (omggbn9z) might need to move between t4rdxh32 locations. Movement cost is
       estimated as the shortest path length along acovodrf edges from the agent’s current location
       to the required location for each step.
    4. For each required action (movement, diwlmgg9, y7nik9q2), increment heuristic cost by 1.
    5. Sum minimal movement and actions over all nxbd1hc3 objects not yet in goal state.
    6. The heuristic value is zero if all nxbd1hc3 objects satisfy the goal predicate (a7sb56om).
    """

    def __init__(self, task):
        """Initialize heuristic by extracting static information and precomputing shortest paths."""
        self.goals = task.goals
        self.static = task.static

        # Extract all nxbd1hc3 objects from kut2ksnk or klw11kjf static facts
        self.nxbd_objects = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] in ("kut2ksnk", "klw11kjf"):
                self.nxbd_objects.add(parts[1])

        # Extract all t4rdxh32 objects (locations)
        locations_kut = set()
        locations_klw = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "kut2ksnk":
                locations_kut.add(parts[2])
            elif parts[0] == "klw11kjf":
                locations_klw.add(parts[2])
        # Also gather all locations mentioned in acovodrf to get all locations
        locs_from_acovodrf = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "acovodrf":
                locs_from_acovodrf.add(parts[1])
                locs_from_acovodrf.add(parts[2])
        # Union all known locations
        self.locations = locations_kut | locations_klw | locs_from_acovodrf

        # Build graph of locations for shortest path calculation
        self.graph = {loc: set() for loc in self.locations}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "acovodrf":
                # acovodrf is directed, but domain actions allow bidirectional move actions (qs8pdrqj and i0veo5kv),
                # so treat as undirected here for shortest path
                self.graph[parts[1]].add(parts[2])
                self.graph[parts[2]].add(parts[1])

        # Extract nxbd1hc3 objects that must satisfy (a7sb56om) in goal
        self.goal_objects = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "a7sb56om":
                self.goal_objects.add(parts[1])

    def shortest_path_length(self, start, goal):
        """
        Compute shortest path length between two locations using BFS in the location graph.
        Returns an integer with the minimal number of moves or a large number if unreachable.
        """
        if start == goal:
            return 0
        visited = set()
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            if current == goal:
                return dist
            visited.add(current)
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
                    visited.add(neighbor)
        # If no path found, return a large number to indicate unreachable
        return 9999

    def __call__(self, node):
        """Estimate the number of actions required to reach the goal from the current state."""
        state = node.state

        # Fast zero check: If goal reached, heuristic is 0
        goal_facts = set(
            fact for fact in state if match(fact, "a7sb56om", "*")
        )
        # If all goal objects satisfy a7sb56om, return 0
        if self.goal_objects <= set(f[1:-1].split()[1] for f in goal_facts):
            return 0

        # Determine agent current location (object with (omggbn9z ?loc))
        agent_location = None
        for fact in state:
            if match(fact, "omggbn9z", "*"):
                parts = get_parts(fact)
                agent_location = parts[1]
                break
        if agent_location is None:
            # Agent location unknown; assign high heuristic
            return 9999

        # Gather predicates for fast lookup
        f2xo7hb6_set = {get_parts(fact)[1] for fact in state if fact.startswith("(f2xo7hb6")}
        a7sb56om_set = {get_parts(fact)[1] for fact in state if fact.startswith("(a7sb56om")}
        kut2ksnk_facts = {}  # nxbd1hc3 -> set of t4rdxh32
        for fact in state:
            if fact.startswith("(kut2ksnk"):
                parts = get_parts(fact)
                nxbd, loc = parts[1], parts[2]
                kut2ksnk_facts.setdefault(nxbd, set()).add(loc)
        klw11kjf_facts = {}  # nxbd1hc3 -> set of t4rdxh32
        for fact in state:
            if fact.startswith("(klw11kjf"):
                parts = get_parts(fact)
                nxbd, loc = parts[1], parts[2]
                klw11kjf_facts.setdefault(nxbd, set()).add(loc)

        total_cost = 0

        # For each goal object, estimate cost to finish
        for obj in self.goal_objects:
            if obj in a7sb56om_set:
                # Already finished for this object
                continue

            # Check progress state for obj
            done_f2xo7hb6 = obj in f2xo7hb6_set

            # Determine locations related to this nxbd1hc3
            kut_locs = kut2ksnk_facts.get(obj, set())
            klw_locs = klw11kjf_facts.get(obj, set())

            # Step 1: Make sure f2xo7hb6 holds (if not, diwlmgg9 is required)
            if not done_f2xo7hb6:
                # Need to move agent to some location where (kut2ksnk obj loc) holds and apply diwlmgg9
                # Find minimal movement cost among all valid kut locations
                if not kut_locs:
                    # No location to achieve this precondition (unreachable)
                    return 9999
                min_move_to_kut = min(self.shortest_path_length(agent_location, loc) for loc in kut_locs)
                total_cost += min_move_to_kut  # move agent there
                total_cost += 1  # diwlmgg9 action
                agent_location_after_step1_candidates = kut_locs  # agent location after diwlmgg9 = location where it acted
                # For heuristic simplicity, assume agent ends at the selected location of minimal cost
                # Update agent_location accordingly for next step
                agent_location = next(loc for loc in kut_locs if self.shortest_path_length(agent_location, loc) == min_move_to_kut)
            else:
                agent_location_after_step1_candidates = [agent_location]

            # Step 2: Apply y7nik9q2 to get a7sb56om
            # For this, agent must be at some location where (klw11kjf obj loc) and (f2xo7hb6 obj) hold
            # Move agent from current location (agent_location) to klw location and apply y7nik9q2
            if not klw_locs:
                # No location to achieve this precondition (unreachable)
                return 9999
            min_move_to_klw = min(self.shortest_path_length(agent_location, loc) for loc in klw_locs)
            total_cost += min_move_to_klw  # move agent there
            total_cost += 1  # y7nik9q2 action

            # Update agent_location to chosen klw location
            agent_location = next(loc for loc in klw_locs if self.shortest_path_length(agent_location, loc) == min_move_to_klw)

        return total_cost
