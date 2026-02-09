from datetime import datetime

class TokenTracker:
    def __init__(self, budget_limit=10.0):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_log = []
        self.budget_limit = budget_limit
        
    def log_call(self, operation, input_tokens, output_tokens, cost):
        """Log each API call"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        
        self.call_log.append({
            'operation': operation,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost,
            'timestamp': datetime.now(),
            'total_cost_so_far': self.total_cost
        })
        
        # Alert if approaching budget
        if self.total_cost > self.budget_limit * 0.8:
            print(f"\n⚠️  WARNING: Using 80% of budget! ${self.total_cost:.4f}/${self.budget_limit}")
        elif self.total_cost > self.budget_limit * 0.95:
            print(f"\n🚨 CRITICAL: Using 95% of budget! ${self.total_cost:.4f}/${self.budget_limit}")
    
    def get_summary(self):
        """Print nice summary"""
        print("\n" + "="*70)
        print("TOKEN USAGE SUMMARY")
        print("="*70)
        print(f"Total Input Tokens:  {self.total_input_tokens:,}")
        print(f"Total Output Tokens: {self.total_output_tokens:,}")
        print(f"Total Cost:          ${self.total_cost:.4f}")
        print(f"Budget Remaining:    ${self.budget_limit - self.total_cost:.4f}")
        pct = ((self.budget_limit - self.total_cost) / self.budget_limit * 100)
        print(f"Remaining Budget:    {pct:.1f}%")
        print("="*70 + "\n")
    
    def save_report(self, filename='tokens_report.txt'):
        """Save report for submission"""
        with open(filename, 'w') as f:
            f.write("TOKEN USAGE REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Input Tokens:  {self.total_input_tokens:,}\n")
            f.write(f"Total Output Tokens: {self.total_output_tokens:,}\n")
            f.write(f"Total Cost:          ${self.total_cost:.4f}\n")
            f.write(f"Budget:              ${self.budget_limit:.2f}\n")
            f.write(f"Budget Remaining:    ${self.budget_limit - self.total_cost:.4f}\n\n")
            
            f.write("BREAKDOWN BY OPERATION:\n")
            f.write("-"*70 + "\n")
            
            # Group by operation
            by_op = {}
            for call in self.call_log:
                op = call['operation']
                if op not in by_op:
                    by_op[op] = {'calls': 0, 'tokens': 0, 'cost': 0.0}
                by_op[op]['calls'] += 1
                by_op[op]['tokens'] += call['input_tokens'] + call['output_tokens']
                by_op[op]['cost'] += call['cost']
            
            for op, stats in sorted(by_op.items(), key=lambda x: x[1]['cost'], reverse=True):
                f.write(f"\n{op}:\n")
                f.write(f"  Calls: {stats['calls']}\n")
                f.write(f"  Tokens: {stats['tokens']:,}\n")
                f.write(f"  Cost: ${stats['cost']:.4f}\n")
            
            f.write("\n\nDETAILED LOG:\n")
            f.write("-"*70 + "\n")
            for call in self.call_log:
                ts = call['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{ts} | {call['operation']:20} | In: {call['input_tokens']:6} | Out: {call['output_tokens']:6} | Cost: ${call['cost']:.6f} | Total: ${call['total_cost_so_far']:.4f}\n")

# Global tracker
tracker = TokenTracker()
