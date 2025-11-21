#!/usr/bin/env python3
"""
Job Architecture Data Explorer
Visualize and explore the job title dataset
"""

import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

def load_data():
    """Load the job architecture data"""
    data_dir = Path(".")
    
    print("Loading data...")
    with open(data_dir / "job_graph.json", 'r') as f:
        graph_data = json.load(f)
    
    with open(data_dir / "statistics.json", 'r') as f:
        stats = json.load(f)
    
    print(f"✓ Loaded {len(graph_data['job_lookup']):,} jobs")
    
    return graph_data, stats

def create_level_distribution(graph_data):
    """Visualize level distribution"""
    jobs = graph_data['job_lookup'].values()
    levels = [job['level'] for job in jobs]
    
    level_names = {
        0: "Intern",
        1: "Junior",
        2: "Mid-Level",
        3: "Senior",
        4: "Staff/Mgr",
        5: "Sr Manager",
        6: "Director",
        7: "VP",
        8: "SVP",
        9: "C-Suite"
    }
    
    level_counts = Counter(levels)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    levels_sorted = sorted(level_counts.keys())
    counts = [level_counts[l] for l in levels_sorted]
    labels = [f"L{l}: {level_names.get(l, str(l))}" for l in levels_sorted]
    
    ax1.bar(range(len(counts)), counts, color='steelblue')
    ax1.set_xticks(range(len(counts)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Number of Job Titles')
    ax1.set_title('Job Titles by Organizational Level')
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart
    ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribution of Job Titles Across Levels')
    
    plt.tight_layout()
    plt.savefig('level_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: level_distribution.png")
    plt.close()

def create_family_distribution(graph_data):
    """Visualize family distribution"""
    jobs = graph_data['job_lookup'].values()
    families = [job['family'] for job in jobs]
    
    family_counts = Counter(families)
    sorted_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    families_list = [f[0] for f in sorted_families]
    counts = [f[1] for f in sorted_families]
    
    bars = ax.barh(range(len(families_list)), counts, color='coral')
    ax.set_yticks(range(len(families_list)))
    ax.set_yticklabels(families_list)
    ax.set_xlabel('Number of Job Titles')
    ax.set_title('Job Titles by Family', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 50, i, f'{count:,}', va='center')
    
    plt.tight_layout()
    plt.savefig('family_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: family_distribution.png")
    plt.close()

def create_family_level_heatmap(graph_data):
    """Create heatmap of families vs levels"""
    jobs = graph_data['job_lookup'].values()
    
    # Create DataFrame
    data = [(job['family'], job['level']) for job in jobs]
    df = pd.DataFrame(data, columns=['Family', 'Level'])
    
    # Create pivot table
    heatmap_data = df.groupby(['Family', 'Level']).size().unstack(fill_value=0)
    
    # Sort families by total count
    family_totals = heatmap_data.sum(axis=1).sort_values(ascending=False)
    heatmap_data = heatmap_data.loc[family_totals.index]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Number of Jobs'}, ax=ax)
    
    ax.set_title('Job Distribution: Family vs Level', fontsize=14, fontweight='bold')
    ax.set_xlabel('Organizational Level', fontsize=12)
    ax.set_ylabel('Job Family', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('family_level_heatmap.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: family_level_heatmap.png")
    plt.close()

def create_soc_category_distribution(graph_data, top_n=20):
    """Show top SOC categories"""
    jobs = graph_data['job_lookup'].values()
    soc_categories = [job['soc_category'] for job in jobs]
    
    soc_counts = Counter(soc_categories)
    top_categories = soc_counts.most_common(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    categories = [c[0] for c in top_categories]
    counts = [c[1] for c in top_categories]
    
    bars = ax.barh(range(len(categories)), counts, color='teal')
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel('Number of Job Titles')
    ax.set_title(f'Top {top_n} SOC Categories by Job Count', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 2, i, f'{count}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('soc_categories.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: soc_categories.png")
    plt.close()

def create_summary_stats(graph_data, stats):
    """Create a summary statistics visualization"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Job Architecture System - Overview', fontsize=16, fontweight='bold')
    
    # Stats boxes
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    stats_text = f"""
    Total Unique Job Titles: {stats['total_jobs']:,}
    Searchable Title Variations: {stats['total_searchable_titles']:,}
    Job Families: {len(stats['families'])}
    Organizational Levels: {len(stats['levels'])}
    SOC Categories: {stats['soc_categories']}
    """
    
    ax1.text(0.5, 0.5, stats_text, ha='center', va='center', 
             fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Level distribution mini chart
    ax2 = fig.add_subplot(gs[1, 0])
    jobs = graph_data['job_lookup'].values()
    levels = [job['level'] for job in jobs]
    level_counts = Counter(levels)
    ax2.bar(sorted(level_counts.keys()), [level_counts[k] for k in sorted(level_counts.keys())])
    ax2.set_title('Jobs by Level')
    ax2.set_xlabel('Level')
    ax2.set_ylabel('Count')
    
    # Family distribution mini chart
    ax3 = fig.add_subplot(gs[1, 1])
    families = [job['family'] for job in jobs]
    family_counts = Counter(families)
    top_families = dict(family_counts.most_common(8))
    ax3.barh(range(len(top_families)), list(top_families.values()))
    ax3.set_yticks(range(len(top_families)))
    ax3.set_yticklabels(list(top_families.keys()), fontsize=9)
    ax3.set_title('Top 8 Families')
    ax3.invert_yaxis()
    
    # Sample jobs table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    sample_jobs = list(graph_data['job_lookup'].values())[:10]
    table_data = [[j['title'][:40], j['level'], j['family']] for j in sample_jobs]
    
    table = ax4.table(cellText=table_data, 
                     colLabels=['Job Title', 'Level', 'Family'],
                     cellLoc='left',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    ax4.set_title('Sample Job Titles', fontsize=12, pad=20)
    
    plt.savefig('summary_dashboard.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: summary_dashboard.png")
    plt.close()

def export_data_samples(graph_data):
    """Export sample data to CSV"""
    jobs = list(graph_data['job_lookup'].values())
    
    # Sample by level
    samples_by_level = {}
    for level in range(10):
        level_jobs = [j for j in jobs if j['level'] == level]
        samples_by_level[level] = level_jobs[:10]
    
    data = []
    for level, job_list in samples_by_level.items():
        for job in job_list:
            data.append({
                'Level': level,
                'Title': job['title'],
                'Family': job['family'],
                'SOC_Category': job['soc_category']
            })
    
    df = pd.DataFrame(data)
    df.to_csv('job_samples_by_level.csv', index=False)
    print("✓ Saved: job_samples_by_level.csv")
    
    # Export family summary
    family_summary = []
    for family in set(j['family'] for j in jobs):
        family_jobs = [j for j in jobs if j['family'] == family]
        level_dist = Counter(j['level'] for j in family_jobs)
        family_summary.append({
            'Family': family,
            'Total_Jobs': len(family_jobs),
            'Min_Level': min(j['level'] for j in family_jobs),
            'Max_Level': max(j['level'] for j in family_jobs),
            'Avg_Level': sum(j['level'] for j in family_jobs) / len(family_jobs)
        })
    
    df_summary = pd.DataFrame(family_summary).sort_values('Total_Jobs', ascending=False)
    df_summary.to_csv('family_summary.csv', index=False)
    print("✓ Saved: family_summary.csv")

def main():
    """Run all visualizations"""
    print("="*60)
    print("Job Architecture Data Explorer")
    print("="*60)
    print()
    
    try:
        graph_data, stats = load_data()
        
        print("\nCreating visualizations...")
        create_summary_stats(graph_data, stats)
        create_level_distribution(graph_data)
        create_family_distribution(graph_data)
        create_family_level_heatmap(graph_data)
        create_soc_category_distribution(graph_data)
        
        print("\nExporting data samples...")
        export_data_samples(graph_data)
        
        print("\n" + "="*60)
        print("✅ All visualizations created successfully!")
        print("="*60)
        print("\nGenerated files:")
        print("  📊 summary_dashboard.png")
        print("  📊 level_distribution.png")
        print("  📊 family_distribution.png")
        print("  📊 family_level_heatmap.png")
        print("  📊 soc_categories.png")
        print("  📄 job_samples_by_level.csv")
        print("  📄 family_summary.csv")
        print()
        
    except FileNotFoundError:
        print("\n❌ Error: job_architecture_data directory not found!")
        print("Please run the Jupyter notebook first to generate the data.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
