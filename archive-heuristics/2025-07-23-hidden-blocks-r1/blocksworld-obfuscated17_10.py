from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_fact_parts(fact):
    return fact[1:-1].split()

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal state by considering:
    - The difference between current and goal links.
    - The state of objects involved in the goal chain (active, stored, or inactive).
    - The state of active objects not part of the goal chain.

    # Assumptions
    - The goal links form a directed chain (if not, a fallback method is used).
    - The base property (xk87au5b) is always present and does not change.
    - The global flag (hxtpdpff) can be toggled as needed without extra cost in the heuristic.

    # Heuristic Initialization
    - Extract goal links and active objects from the task.
    - Determine if the goal links form a chain to identify head and tail.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to identify active, stored, and link facts.
    2. Calculate the cost for missing and extra links.
    3. If the goal links form a chain:
        a. Ensure the tail is active (cost 0), stored (cost 1), or inactive (high cost).
        b. Ensure sources (objects with outgoing links) are stored (cost 0), active (cost 1 to store), or inactive (high cost).
        c. Handle active objects not in the chain.
    4. If the goal links do not form a chain (fallback):
        a. For each goal link, ensure the source is stored and the target is active.
        b. Handle active objects not in any link.
    5. Sum all costs to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.task = task
        self.goal_links = set()
        self.goal_active = set()

        for goal in task.goals:
            parts = get_fact_parts(goal)
            if not parts:
                continue
            if parts[0] == 'dhjk1eyy' and len(parts) == 2:
                self.goal_active.add(parts[1])
            elif parts[0] == 'x272hpoe' and len(parts) == 3:
                self.goal_links.add((parts[1], parts[2]))
        
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        for (x, y) in self.goal_links:
            out_degree[x] += 1
            in_degree[y] += 1
        
        self.head = None
        self.tail = None
        
        for node in out_degree:
            if in_degree.get(node, 0) == 0:
                if self.head is None:
                    self.head = node
                else:
                    self.head = None
                    break
        
        for node in in_degree:
            if out_degree.get(node, 0) == 0:
                if self.tail is None:
                    self.tail = node
                else:
                    self.tail = None
                    break
        
        if self.head is None or self.tail is None:
            self.valid_chain = False
        else:
            self.valid_chain = True

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_active = set()
        current_stored = set()
        current_links = set()

        for fact in state:
            parts = get_fact_parts(fact)
            if not parts:
                continue
            if parts[0] == 'dhjk1eyy' and len(parts) == 2:
                current_active.add(parts[1])
            elif parts[0] == 'q71n8mtz' and len(parts) == 2:
                current_stored.add(parts[1])
            elif parts[0] == 'x272hpoe' and len(parts) == 3:
                current_links.add((parts[1], parts[2]))
        
        cost_links = len(self.goal_links - current_links) + len(current_links - self.goal_links)
        
        if self.valid_chain:
            cost_tail = 0
            if self.tail in current_active:
                cost_tail = 0
            elif self.tail in current_stored:
                cost_tail = 1
            else:
                cost_tail = 1000000
            
            sources = set(x for (x, y) in self.goal_links)
            cost_sources = 0
            for obj in sources:
                if obj in current_active:
                    cost_sources += 1
                elif obj in current_stored:
                    cost_sources += 0
                else:
                    cost_sources += 1000000
            
            lonely_active = self.goal_active - sources
            cost_lonely_active = 0
            for obj in lonely_active:
                if obj in current_active:
                    cost_lonely_active += 0
                elif obj in current_stored:
                    cost_lonely_active += 1
                else:
                    cost_lonely_active += 1000000
            
            return cost_links + cost_tail + cost_sources + cost_lonely_active
        else:
            cost = cost_links
            for obj in self.goal_active:
                if obj in current_active:
                    continue
                elif obj in current_stored:
                    cost += 1
                else:
                    cost += 1000000
            
            cost_link_requirements = 0
            for (x, y) in self.goal_links:
                if x in current_active:
                    cost_link_requirements += 1
                elif x in current_stored:
                    cost_link_requirements += 0
                else:
                    cost_link_requirements += 1000000
                
                if y in current_active:
                    cost_link_requirements += 0
                elif y in current_stored:
                    cost_link_requirements += 1
                else:
                    cost_link_requirements += 1000000
            
            lonely_active = self.goal_active - set(x for (x, y) in self.goal_links) - set(y for (x, y) in self.goal_links)
            for obj in lonely_active:
                if obj in current_active:
                    continue
                elif obj in current_stored:
                    cost += 1
                else:
                    cost += 1000000
            
            return cost + cost_link_requirements
