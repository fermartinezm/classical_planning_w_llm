from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (unkhrp2l).

    Summary:
    The heuristic estimates the number of actions required to transform all jxhpkqr5-type objects 
    from the initial states (where they satisfy predicates pxch7pi8 and eacj8ftl) into goal states 
    (where they satisfy guby8wxu). The primary cost comes from:
     1) Moving the 'v0yy74n7' predicate between ql6b2h31-type objects along allowed paths (defined by pvhp8m0p)
     2) Transforming jxhpkqr5 objects' predicates from pxch7pi8 to eacj8ftl to guby8wxu via actions.
    
    Assumptions:
     - Each jxhpkqr5 object must reach a state with (guby8wxu ?obj).
     - The predicate (v0yy74n7 ?loc) represents a token that moves between ql6b2h31 objects.
     - Movement between ql6b2h31 objects is only possible along edges defined by predicate (pvhp8m0p).
     - jxhpkqr5 objects interact with ql6b2h31 objects to update their predicates pxch7pi8 -> eacj8ftl -> guby8wxu.
     - Actions that move the token or update object predicates cost 1 each.
     - The heuristic does not need to be admissible but should provide a rough count of the minimal steps.
    
    Heuristic Initialization:
     - Extract static paths pvhp8m0p between ql6b2h31 objects as a bidirectional graph.
     - Extract goal jxhpkqr5 objects that must satisfy (guby8wxu ?obj).
     - Extract initial predicates pxch7pi8 (associations between jxhpkqr5 and ql6b2h31).
     - Extract initial predicates eacj8ftl and guby8wxu from the current state.
     - Extract which ql6b2h31 object currently has (v0yy74n7 ?loc).
    
    Step-By-Step Thinking for Computing Heuristic:
     1) For each jxhpkqr5 object that must reach (guby8wxu ?obj) in goal:
       - If already (guby8wxu ?obj), cost = 0 for that object.
       - Else, determine how far along the predicate chain the object is:
         a) If (eacj8ftl ?obj), it needs one action (jg64qouf) and its ql6b2h31 location.
         b) If (pxch7pi8 ?obj ?loc), it must first do makpz2tc to gain (eacj8ftl),
            then jg64qouf to reach (guby8wxu).
         c) Else, it has not started and must do a sequence from pxch7pi8 state.
       - Identify the target ql6b2h31 location for interaction based on the predicate the object currently has.
     
     2) Calculate token moves:
       - The token (v0yy74n7) must move from its current location to the required ql6b2h31 location 
         where the object interaction occurs.
       - Use shortest distance on the pvhp8m0p graph to estimate moves.
       - Movement of the token costs one action per edge traversal.

     3) Sum for all jxhpkqr5:
       (token moves to required location) + (
        number of predicate-changing actions needed for the object:
        1 or 2 actions depending on whether object is at pxch7pi8 or eacj8ftl).

     4) Approximate token moves sequentially for all jxhpkqr5 objects not yet at goal,
        assuming token must move individually for each.

     5) Return total estimated cost as heuristic value.

    This heuristic captures the main costs in moving the token and updating object predicates,
    providing guidance for greedy best-first search.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build adjacency for token movement graph from pvhp8m0p static facts (bidirectional).
        self.token_graph = dict()  # {loc: set(connected_locs)}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pvhp8m0p" and len(parts) == 3:
                a, b = parts[1], parts[2]
                self.token_graph.setdefault(a, set()).add(b)
                self.token_graph.setdefault(b, set()).add(a)

        # Extract goal jxhpkqr5 objects required to satisfy (guby8wxu ?obj)
        self.goal_objects = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "guby8wxu" and len(parts) == 2:
                self.goal_objects.add(parts[1])

    def shortest_path_length(self, start, goal):
        """Compute shortest path length on token graph from start to goal using BFS. Return a large number if unreachable."""
        if start == goal:
            return 0
        from collections import deque

        visited = set()
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            if node == goal:
                return dist
            visited.add(node)
            for neighbor in self.token_graph.get(node, ()):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return 1000  # Large penalty for unreachable

    def __call__(self, node):
        state = node.state

        # Extract current mappings from the state:
        # pxch7pi8 ?obj ?loc
        pxch7pi8_map = {}  # obj -> loc
        # eacj8ftl ?obj
        eacj8ftl_set = set()
        # guby8wxu ?obj
        guby8wxu_set = set()
        # v0yy74n7 ?loc (position of token)
        token_location = None

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "pxch7pi8" and len(parts) == 3:
                pxch7pi8_map[parts[1]] = parts[2]
            elif pred == "eacj8ftl" and len(parts) == 2:
                eacj8ftl_set.add(parts[1])
            elif pred == "guby8wxu" and len(parts) == 2:
                guby8wxu_set.add(parts[1])
            elif pred == "v0yy74n7" and len(parts) == 2:
                token_location = parts[1]

        # If token location unknown (not in state), assign None and handle conservatively
        if token_location is None:
            # Arbitrary large penalty: no token location means heuristic can't compute moves.
            token_location = None

        # We will estimate cost for each goal object:
        # Cost = steps to move token to interaction location +
        #        number of object predicate-transition actions needed (1 or 2)
        total_cost = 0
        current_token_pos = token_location

        for obj in self.goal_objects:
            # If object already satisfies goal, cost 0 for this object
            if obj in guby8wxu_set:
                continue

            # Determine object's current predicate state:
            if obj in eacj8ftl_set:
                # Object is in middle state, needs jg64qouf to get to guby8wxu.
                # Interaction occurs at the location from which eacj8ftl is obtained,
                # which corresponds to the ql6b2h31 location associated with (pahbmn6t ?obj ?loc).
                # We find that location from static facts.
                int_loc = None
                for fact in self.static:
                    parts = get_parts(fact)
                    if parts[0] == "pahbmn6t" and parts[1] == obj:
                        int_loc = parts[2]
                        break
                if int_loc is None:
                    # If unknown interaction location, fallback to no moves (penalty)
                    int_loc = current_token_pos
                moves = 0 if current_token_pos is None else self.shortest_path_length(current_token_pos, int_loc)
                total_cost += moves + 1  # 1 action to apply jg64qouf
                current_token_pos = int_loc

            elif obj in pxch7pi8_map:
                # Object in initial state, linked to some ql6b2h31 location.
                # Needs:
                #   - moving token to pxch7pi8 location,
                #   - makpz2tc (1 action) to get eacj8ftl,
                #   - jg64qouf (1 action) to get guby8wxu.
                int_loc = pxch7pi8_map[obj]
                moves = 0 if current_token_pos is None else self.shortest_path_length(current_token_pos, int_loc)
                total_cost += moves + 2  # 2 actions to transition predicates
                current_token_pos = int_loc

            else:
                # Object neither in eacj8ftl nor pxch7pi8: likely not started yet or unreachable state.
                # We treat similarly to pxch7pi8 case but with large penalty for unknown loc.
                # Try to find a pahbmn6t location as interaction point (fallback)
                int_loc = None
                for fact in self.static:
                    parts = get_parts(fact)
                    if parts[0] == "pahbmn6t" and parts[1] == obj:
                        int_loc = parts[2]
                        break
                if int_loc is None:
                    # fallback: no known interaction location; large penalty
                    moves = 100
                else:
                    moves = 0 if current_token_pos is None else self.shortest_path_length(current_token_pos, int_loc)
                total_cost += moves + 2  # cost of two predicate transitions
                current_token_pos = int_loc if int_loc is not None else current_token_pos

        return total_cost
