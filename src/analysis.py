# src/analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_analysis_report(scores_df, output_path='analysis.md'):
    """Generate markdown report with visualizations"""
    print("Generating analysis report...")
    
    # Create bins
    bins = list(range(0, 1100, 100))
    labels = [f"{i}-{i+99}" for i in range(0, 1000, 100)]
    scores_df['score_range'] = pd.cut(scores_df['credit_score'], bins=bins, labels=labels)
    
    # Generate plots
    plt.figure(figsize=(10, 6))
    sns.histplot(data=scores_df, x='credit_score', bins=20)
    plt.title('Credit Score Distribution')
    plt.savefig('src/outputs/score_distribution.png')
    plt.close()
    
    # Generate markdown
    with open(output_path, 'w') as f:
        f.write("# Aave V2 Wallet Credit Score Analysis\n\n")
        f.write("## Score Distribution\n")
        f.write("![Score Distribution](src/outputs/score_distribution.png)\n\n")
        
        f.write("## Score Range Statistics\n")
        stats = scores_df.groupby('score_range').size().reset_index(name='count')
        f.write(stats.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Interpretation\n")
        f.write("- Wallets with scores < 300 show high-risk behavior (frequent liquidations, abnormal patterns)\n")
        f.write("- Scores 300-600 indicate moderate risk with some concerning patterns\n")
        f.write("- Scores 600-800 represent typical protocol users\n")
        f.write("- Scores > 800 indicate highly reliable, conservative usage patterns\n")