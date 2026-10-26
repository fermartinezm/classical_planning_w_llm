from heuristics.heuristic_base import Heuristic

def parse_fact(fact_str):
    if fact_str.startswith('(') and fact_str.endswith(')'):
        content = fact_str[1:-1].strip()
        if not content:
            return None, []
        parts = content.split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args
    return None, []

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.
    
    Summary:
    This heuristic estimates the number of actions required to achieve the goal by considering:
    - The number of missing links in the goal chain.
    - The number of missing 'xk87au5b' predicates on objects specified in the goal.
    - The number of goal chain head objects not in the 'q71n8mtz' state.
    
    Assumptions:
    - The goal involves a single chain of 'x272hpoe' relations.
    - The chain is linear and has one head (with 'dhjk1eyy') and one tail.
    - The global flag 'hxtpdpff' is managed by actions and does not require explicit counting.
    
    Heuristic Initialization:
    - Extracts goal conditions: 'dhjk1eyy', 'xk87au5b', and 'x272hpoe' relations.
    - Constructs the goal chain by identifying the head (with 'dhjk1eyy') and following 'x272hpoe' links.
    - Identifies goal heads (all chain objects except the tail) and objects requiring 'xk87au5b'.
    
    Step-By-Step Thinking for Computing Heuristic:
    1. For the current state, parse predicates to identify present 'x272hpoe' links, 'xk87au5b', and 'q71n8mtz'.
    2. Compute missing links: goal links not in the current state.
    3. Compute missing 'xk87au5b': objects specified in the goal lacking 'xk87au5b' in the current state.
    4. Compute missing 'q71n8mtz' heads: goal heads not in 'q71n8mtz' state.
    5. Sum these counts to estimate the total actions needed.
    """
    
    def __init__(self, task):
        self.goal_links = set()
        self.goal_xk87au5b = set()
        self.goal_dhjk1eyy = set()
        
        for goal in task.goals:
            pred, args = parse_fact(goal)
            if pred == 'x272hpoe' and len(args) == 2:
                self.goal_links.add((args[0], args[1]))
            elif pred == 'xk87au5b' and len(args) == 1:
                self.goal_xk87au5b.add(args[0])
            elif pred == 'dhjk1eyy' and len(args) == 1:
                self.goal_dhjk1eyy.add(args[0])
        
        outgoing = {}
        for (fr, to) in self.goal_links:
            outgoing[fr] = to
        
        self.chain = []
        self.goal_heads = set()
        candidate_heads = [head for head in self.goal_dhjk1eyy if head not in outgoing.values()]
        if len(candidate_heads) == 1:
            head = candidate_heads[0]
            self.chain = [head]
            current = head
            while current in outgoing:
                next_obj = outgoing[current]
                self.chain.append(next_obj)
                current = next_obj
            if len(self.chain) > 1:
                self.goal_heads = set(self.chain[:-1])
            else:
                self.goal_heads = set()
        else:
            self.chain = None
            self.goal_heads = set(candidate_heads)
    
    def __call__(self, node):
        state = node.state
        current_links = set()
        current_xk87au5b = set()
        current_q71n8mtz = set()
        
        for fact in state:
            pred, args = parse_fact(fact)
            if pred == 'x272hpoe' and len(args) == 2:
                current_links.add((args[0], args[1]))
            elif pred == 'xk87au5b' and len(args) == 1:
                current_xk87au5b.add(args[0])
            elif pred == 'q71n8mtz' and len(args) == 1:
                current_q71n8mtz.add(args[0])
        
        if self.chain is not None:
            goal_links_set = set()
            for i in range(len(self.chain) - 1):
                goal_links_set.add((self.chain[i], self.chain[i+1]))
            missing_links = len(goal_links_set - current_links)
        else:
            missing_links = len(self.goal_links - current_links)
        
        missing_xk87au5b = len(self.goal_xk87au5b - current_xk87au5b)
        missing_q71n8mtz_heads = len(self.goal_heads - current_q71n8mtz)
        
        return missing_links + missing_xk87au5b + missing_q71n8mtz_heads
