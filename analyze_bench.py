import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

multi = pd.read_csv('bench_multi.csv')
single = pd.read_csv('bench_single.csv')

avg_multi = multi['ms_per_step'].mean()
avg_single = single['ms_per_step'].mean()
speedup = avg_single / avg_multi

# Summary dataframe
summary = pd.DataFrame([
    {"mode":"Single-core","threads":int(single['threads'].iloc[0]),"avg_ms_per_step":avg_single},
    {"mode":"Multi-core","threads":int(multi['threads'].iloc[0]),"avg_ms_per_step":avg_multi},
])
summary['speedup_vs_single'] = summary.loc[0,'avg_ms_per_step'] / summary['avg_ms_per_step']

# Write summary CSV and friendly text
summary.to_csv('bench_summary.csv', index=False)
with open('bench_summary.txt','w') as f:
    f.write(f"Single-core avg ms/step: {avg_single:.3f}\n")
    f.write(f"Multi-core  avg ms/step: {avg_multi:.3f}\n")
    f.write(f"Speedup: {speedup:.2f}x\n")

# Print a markdown table for your PDF/report
print("| Mode | Threads | Avg ms/step | Speedup vs Single |")
print("|------|---------|-------------|-------------------|")
for _,row in summary.iterrows():
    print(f"| {row['mode']} | {int(row['threads'])} | {row['avg_ms_per_step']:.3f} | {row['speedup_vs_single']:.2f}x |")

# Time-series plot
plt.figure(figsize=(8,5))
plt.plot(single['step'], single['ms_per_step'], label=f"Single-core ({avg_single:.2f} ms/step)")
plt.plot(multi['step'],  multi['ms_per_step'],  label=f"Multi-core ({avg_multi:.2f} ms/step)")
plt.xlabel('Step'); plt.ylabel('Milliseconds per step')
plt.title('Physics Benchmark: Single vs Multi-core (ms/step)')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=180)

# Speed bar chart
plt.figure(figsize=(6,5))
plt.bar(['Single','Multi'], [avg_single, avg_multi])
plt.ylabel('Avg ms/step')
plt.title(f'Average Time per Step (Speedup {speedup:.2f}×)')
plt.tight_layout()
plt.savefig('performance_speed_bar.png', dpi=180)

print("Wrote: bench_summary.csv, bench_summary.txt, performance_comparison.png, performance_speed_bar.png")
