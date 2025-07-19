import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Set up the output directory
output_dir = '../output/NumberOfRacesQuestion'

def extract_average_from_csv(filepath):
    """Extract the average time difference from the last row of a CSV file."""
    try:
        df = pd.read_csv(filepath)
        if not df.empty:
            # Get the last row which contains the average
            last_row = df.iloc[-1]
            return last_row['time_diff_seconds']
        else:
            return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def create_broader_category_visualizations():
    """Create visualizations for the original broader categories (2-3, 4-5, 6+)."""
    print("Creating broader category visualizations...")
    
    # Data structure to store averages
    data = {
        '2023': {'M': {}, 'F': {}},
        '2024': {'M': {}, 'F': {}}
    }
    
    # Extract averages from the broader category files
    for year in ['2023', '2024']:
        for gender in ['M', 'F']:
            gender_label = 'male' if gender == 'M' else 'female'
            
            # 2-3 races
            file_2_3 = os.path.join(output_dir, f'{gender_label}_2_3_time_diff_{year}.csv')
            avg_2_3 = extract_average_from_csv(file_2_3)
            if avg_2_3 is not None:
                data[year][gender]['2-3'] = avg_2_3
            
            # 4-5 races
            file_4_5 = os.path.join(output_dir, f'{gender_label}_4_5_time_diff_{year}.csv')
            avg_4_5 = extract_average_from_csv(file_4_5)
            if avg_4_5 is not None:
                data[year][gender]['4-5'] = avg_4_5
            
            # 6+ races
            file_6_plus = os.path.join(output_dir, f'{gender_label}_6_plus_time_diff_{year}.csv')
            avg_6_plus = extract_average_from_csv(file_6_plus)
            if avg_6_plus is not None:
                data[year][gender]['6+'] = avg_6_plus
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Average Time Difference by Race Count (Broader Categories)', fontsize=16)
    
    categories = ['2-3', '4-5', '6+']
    colors = {'M': 'blue', 'F': 'red'}
    gender_labels = {'M': 'Male', 'F': 'Female'}
    
    for i, year in enumerate(['2023', '2024']):
        for j, gender in enumerate(['M', 'F']):
            ax = axes[i, j]
            
            values = []
            for cat in categories:
                if cat in data[year][gender]:
                    values.append(data[year][gender][cat])
                else:
                    values.append(0)
            
            bars = ax.bar(categories, values, color=colors[gender], alpha=0.7)
            ax.set_title(f'{gender_labels[gender]} Athletes - {year}')
            ax.set_xlabel('Number of Races')
            ax.set_ylabel('Average Time Difference (seconds)')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'broader_categories_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Broader category visualization saved!")

def create_granular_category_visualizations():
    """Create visualizations for the granular categories (2, 3, 4, 5, 6 races)."""
    print("Creating granular category visualizations...")
    
    # Data structure to store averages
    data = {
        '2023': {'M': {}, 'F': {}},
        '2024': {'M': {}, 'F': {}}
    }
    
    # Extract averages from the granular category files
    for year in ['2023', '2024']:
        for gender in ['M', 'F']:
            gender_label = 'male' if gender == 'M' else 'female'
            
            for race_count in ['2', '3', '4', '5', '6']:
                file_path = os.path.join(output_dir, f'{gender_label}_{race_count}_time_diff_{year}.csv')
                avg = extract_average_from_csv(file_path)
                if avg is not None:
                    data[year][gender][race_count] = avg
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Average Time Difference by Race Count (Granular Categories)', fontsize=16)
    
    categories = ['2', '3', '4', '5', '6']
    colors = {'M': 'blue', 'F': 'red'}
    gender_labels = {'M': 'Male', 'F': 'Female'}
    
    for i, year in enumerate(['2023', '2024']):
        for j, gender in enumerate(['M', 'F']):
            ax = axes[i, j]
            
            values = []
            for cat in categories:
                if cat in data[year][gender]:
                    values.append(data[year][gender][cat])
                else:
                    values.append(0)
            
            bars = ax.bar(categories, values, color=colors[gender], alpha=0.7)
            ax.set_title(f'{gender_labels[gender]} Athletes - {year}')
            ax.set_xlabel('Number of Races')
            ax.set_ylabel('Average Time Difference (seconds)')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'granular_categories_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Granular category visualization saved!")

def create_combined_visualization():
    """Create a combined visualization showing both years and genders together."""
    print("Creating combined visualization...")
    
    # Extract data for both categories
    broader_data = {'2023': {'M': {}, 'F': {}}, '2024': {'M': {}, 'F': {}}}
    granular_data = {'2023': {'M': {}, 'F': {}}, '2024': {'M': {}, 'F': {}}}
    
    # Broader categories
    for year in ['2023', '2024']:
        for gender in ['M', 'F']:
            gender_label = 'male' if gender == 'M' else 'female'
            for cat in ['2_3', '4_5', '6_plus']:
                file_path = os.path.join(output_dir, f'{gender_label}_{cat}_time_diff_{year}.csv')
                avg = extract_average_from_csv(file_path)
                if avg is not None:
                    broader_data[year][gender][cat] = avg
    
    # Granular categories
    for year in ['2023', '2024']:
        for gender in ['M', 'F']:
            gender_label = 'male' if gender == 'M' else 'female'
            for race_count in ['2', '3', '4', '5', '6']:
                file_path = os.path.join(output_dir, f'{gender_label}_{race_count}_time_diff_{year}.csv')
                avg = extract_average_from_csv(file_path)
                if avg is not None:
                    granular_data[year][gender][race_count] = avg
    
    # Create combined plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Broader categories
    broader_cats = ['2-3', '4-5', '6+']
    x_pos = range(len(broader_cats))
    width = 0.35
    
    for i, year in enumerate(['2023', '2024']):
        ax = ax1 if i == 0 else ax2
        
        male_vals = [broader_data[year]['M'].get(cat.replace('-', '_').replace('+', '_plus'), 0) for cat in broader_cats]
        female_vals = [broader_data[year]['F'].get(cat.replace('-', '_').replace('+', '_plus'), 0) for cat in broader_cats]
        
        bars1 = ax.bar([x - width/2 for x in x_pos], male_vals, width, label='Male', color='blue', alpha=0.7)
        bars2 = ax.bar([x + width/2 for x in x_pos], female_vals, width, label='Female', color='red', alpha=0.7)
        
        ax.set_title(f'Broader Categories - {year}')
        ax.set_xlabel('Number of Races')
        ax.set_ylabel('Average Time Difference (seconds)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(broader_cats)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                           f'{height:.1f}', ha='center', va='bottom')
    
    # Granular categories
    granular_cats = ['2', '3', '4', '5', '6']
    x_pos = range(len(granular_cats))
    
    for i, year in enumerate(['2023', '2024']):
        ax = ax3 if i == 0 else ax4
        
        male_vals = [granular_data[year]['M'].get(cat, 0) for cat in granular_cats]
        female_vals = [granular_data[year]['F'].get(cat, 0) for cat in granular_cats]
        
        bars1 = ax.bar([x - width/2 for x in x_pos], male_vals, width, label='Male', color='blue', alpha=0.7)
        bars2 = ax.bar([x + width/2 for x in x_pos], female_vals, width, label='Female', color='red', alpha=0.7)
        
        ax.set_title(f'Granular Categories - {year}')
        ax.set_xlabel('Number of Races')
        ax.set_ylabel('Average Time Difference (seconds)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(granular_cats)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                           f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Combined visualization saved!")

def main():
    print("Creating visualizations for average time differences by race count...")
    
    # Create all visualizations
    create_broader_category_visualizations()
    create_granular_category_visualizations()
    create_combined_visualization()
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main() 