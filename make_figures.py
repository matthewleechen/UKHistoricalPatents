"""
Make and save plots for all figures in the paper 
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd
from collections import Counter
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random
from math import radians, sin, cos, sqrt, atan2
from utils import load_base_datasets, load_kpst_scores, process_patent_inventors

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)



def load_data():
    
    # Load base datasets
    dataset_all_years, dataset_all_entities = load_base_datasets()
    
    # Process entities into dataframe
    entities_df = process_patent_inventors(dataset_all_entities)
    
    # Load KPST scores
    kpst_scores = load_kpst_scores()
    
    return dataset_all_years, dataset_all_entities, entities_df, kpst_scores



def generate_figure_2(dataset_all_years, entities_df):
    """
    Generate patent counts and inventor cohorts plots.
    """
    # Get patent counts
    patents_by_year = pd.Series(Counter(dataset_all_years['train']['year'])).sort_index()
    
    # Get inventor counts
    first_patents = entities_df.groupby('inventor_id')['year'].min()
    new_inventor_counts = first_patents.value_counts().sort_index()
    
    # Get subset for post-1800 analysis
    patents_1800 = patents_by_year[patents_by_year.index >= 1800]
    inventors_1800 = new_inventor_counts[new_inventor_counts.index >= 1800]
    
    # 1. Patent counts in levels
    plt.figure(figsize=(10, 6))
    plt.bar(patents_by_year.index, patents_by_year.values, width=0.8, color='#E24A33')
    
    plt.axvline(x=1852, color='black', linestyle='--', alpha=1)
    plt.axvline(x=1883, color='black', linestyle='--', alpha=1)
    
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Number of Patents', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/patent_counts_levels.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Patent counts in log (1800-1899)
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.bar(patents_1800.index, patents_1800.values, width=0.8, color='#E24A33')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    
    plt.axvline(x=1852, color='black', linestyle='--', alpha=1)
    plt.axvline(x=1883, color='black', linestyle='--', alpha=1)
    
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Log Number of Patents', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/patent_counts_log.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Inventor cohorts in levels
    plt.figure(figsize=(10, 6))
    plt.bar(new_inventor_counts.index, new_inventor_counts.values, width=0.8, color='#348ABD')
    
    plt.axvline(x=1852, color='black', linestyle='--', alpha=1)
    plt.axvline(x=1883, color='black', linestyle='--', alpha=1)
    
    plt.xlabel('Year of First Patent', fontsize=10)
    plt.ylabel('Number of Inventors', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/inventor_cohorts_levels.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Inventor cohorts in log (1800-1899)
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.bar(inventors_1800.index, inventors_1800.values, width=0.8, color='#348ABD')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    
    plt.axvline(x=1852, color='black', linestyle='--', alpha=1)
    plt.axvline(x=1883, color='black', linestyle='--', alpha=1)
    
    plt.xlabel('Year of First Patent', fontsize=10)
    plt.ylabel('Log Number of Inventors', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/inventor_cohorts_log.png", dpi=300, bbox_inches='tight')
    plt.close()



def generate_figure_3(entities_df):
    """
    Generate co-invention plots.
    """
    # Count inventors per patent
    patents_by_inventors = entities_df.groupby(['year', 'patent_id'])['inventor_id'].nunique()
    
    # Mark co-invented patents (more than 1 inventor)
    coinvented = patents_by_inventors > 1
    
    # Calculate number and share of co-invented patents per year
    coinvention_count = coinvented.groupby('year').sum()
    coinvention_share = coinvented.groupby('year').mean()

    # Plot 1: Number of co-invented patents
    plt.figure(figsize=(10, 6))
    plt.bar(coinvention_count.index, coinvention_count.values,
            width=0.8, color='#E24A33')
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Number of Co-invented Patents', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/coinvention_count.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Share of co-invented patents
    plt.figure(figsize=(10, 6))
    plt.bar(coinvention_share.index, coinvention_share.values * 100,
            width=0.8, color='#348ABD')
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Share of Co-invented Patents (%)', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/coinvention_share.png", dpi=300, bbox_inches='tight')
    plt.close()



def generate_figure_4(dataset_all_entities):
    """
    Generate communicated patents plots.
    """
    # Count all patents and communicated patents per year
    total_patents = Counter(dataset_all_entities['train']['year'])
    communicated_patents = Counter()
    
    # Check for COMM entities in each patent's front_page_entities
    for row in dataset_all_entities['train']:
        if any(entity['class'] == 'COMM' for entity in row['front_page_entities']):
            communicated_patents[row['year']] += 1
    
    # Convert to pandas Series and sort by year
    total_patents_series = pd.Series(total_patents).sort_index()
    communicated_patents_series = pd.Series(communicated_patents).sort_index()
    
    # Calculate share for each year
    communicated_share = pd.Series({
        year: communicated_patents[year] / total_patents[year]
        for year in total_patents.keys()
    }).sort_index()

    # Plot 1: Number of communicated patents
    plt.figure(figsize=(10, 6))
    plt.bar(communicated_patents_series.index, communicated_patents_series.values,
            width=0.8, color='#E24A33')
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Number of Communicated Patents', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/communicated_count.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Share of communicated patents
    plt.figure(figsize=(10, 6))
    plt.bar(communicated_share.index, communicated_share.values * 100,
            width=0.8, color='#348ABD')
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Share of Communicated Patents (%)', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/communicated_share.png", dpi=300, bbox_inches='tight')
    plt.close()



def generate_figure_5(dataset_all_years):
    """
    Generate patent classes plots.
    """
    # Extract only 'year' and 'predicted_BPO_classes' columns
    df_years = pd.DataFrame({
        'year': dataset_all_years['train']['year'],
        'predicted_BPO_classes': dataset_all_years['train']['predicted_BPO_classes']
    })

    year_subsets = {
        "1617-1750": (1617, 1750),
        "1751-1800": (1751, 1800),
        "1801-1850": (1801, 1850),
        "1851-1899": (1851, 1899),
        "Full Period": (1617, 1899),
    }

    # Process each subset and save plots
    for subset_name, (start_year, end_year) in year_subsets.items():
        # Filter df for the current subset
        subset_df = df_years[
            (df_years["year"] >= start_year) & 
            (df_years["year"] <= end_year)
        ]

        # Explode the 'predicted_BPO_classes' lists into separate rows
        subset_exploded = subset_df.explode("predicted_BPO_classes")

        # Count the frequency of each class and get top 10
        class_counts = subset_exploded["predicted_BPO_classes"].value_counts()
        top10 = class_counts.head(10)

        # Calculate percentage share for each top 10 class
        total_top10 = top10.sum()
        percentage_share = (top10 / total_top10) * 100

        # Create a horizontal bar plot
        plt.figure(figsize=(8, 6))
        sns.set_theme(style="white")
        ax = sns.barplot(
            x=percentage_share.values,
            y=percentage_share.index,
            color='#348ABD',
            orient='h'
        )

        # Customize the plot
        ax.set_title(f"Top 10 Patent Classes ({subset_name})", fontsize=14)
        ax.set_xlabel("Percentage of Patents (%)", fontsize=12)
        ax.set_ylabel("Patent Class", fontsize=12)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Adjust y-axis labels
        ax.tick_params(axis='y', labelsize=10)

        # Annotate bars with percentage values
        for i, (percent, label) in enumerate(zip(percentage_share.values, percentage_share.index)):
            ax.text(percent + 0.5, i, f"{percent:.1f}%", va='center', fontsize=10)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"output/top10_patent_classes_{subset_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()



def generate_figure_6(dataset_all_entities):
    """
    Generate geographic distribution plots.
    """
    # Load and crop Europe map from LeakyMirror ("https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson")
    europe = gpd.read_file('temp/europe.geojson')
    uk_bounds = {'minx': -11, 'miny': 49, 'maxx': 2, 'maxy': 61} # approx. UK bounding box coords
    europe_cropped = europe.cx[uk_bounds['minx']:uk_bounds['maxx'],
                             uk_bounds['miny']:uk_bounds['maxy']]

    # Process patent coordinates
    patent_coords = []
    for row in dataset_all_entities['train']:
        coords = [
            (ent['longitude'], ent['latitude'])
            for ent in row['front_page_entities']
            if (ent.get('class') == 'ADD' and
                ent.get('longitude') is not None and 
                ent.get('latitude') is not None)
        ]
        if coords:
            lon, lat = random.choice(coords)
            if (uk_bounds['minx'] <= lon <= uk_bounds['maxx'] and
                uk_bounds['miny'] <= lat <= uk_bounds['maxy']):
                patent_coords.append((row['year'], lon, lat))

    df = pd.DataFrame(patent_coords, columns=['year', 'longitude', 'latitude'])

    # Generate plots for each time period
    time_periods = [(1617, 1750), (1751, 1800), (1801, 1850), (1851, 1899)]
    
    for start, end in time_periods:
        
        # Filter data for period
        period_data = df[(df['year'] >= start) & (df['year'] <= end)]
        
        if period_data.empty:
            continue

        # Group locations
        locations = period_data.groupby(['longitude', 'latitude']).size().reset_index(name='count')
        
        if locations.empty:
            continue

        # Create plot
        max_count = locations['count'].max()
        sizes = np.sqrt(locations['count'].values / max_count) * 2000
        
        fig, ax = plt.subplots(figsize=(10, 15))
        europe_cropped.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=1)
        ax.scatter(locations['longitude'].values, locations['latitude'].values,
                  s=sizes, c='red', alpha=0.3, zorder=2)
        
        ax.set_xlim(uk_bounds['minx'], uk_bounds['maxx'])
        ax.set_ylim(uk_bounds['miny'], uk_bounds['maxy'])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(f'Patent Locations {start}-{end}')
        
        plt.savefig(f'output/uk_patents_{start}_{end}.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()



def generate_figure_7(dataset_all_entities):
    """
    Generate inventor location plots.
    """

    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points.
        """
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371  # Radius of earth in kilometers
        return c * r

    # Initialize dictionaries for inventor data
    inventor_locations = {}  # store sets of location pairs
    first_years = {}  # store first patent year
    max_distances = {}  # store maximum distance between locations

    # Process each patent
    for row in dataset_all_entities['train']:
        year = row['year']
        
        # Find all ADD entities and their coordinates
        address_coords = {}
        for entity in row['front_page_entities']:
            if (entity['class'] == 'ADD' and 
                'person_id' in entity and 
                entity['person_id'] is not None and
                'latitude' in entity and 
                entity['latitude'] is not None and 
                'longitude' in entity and 
                entity['longitude'] is not None):
                
                for pid in entity['person_id']:
                    if pid not in address_coords:
                        address_coords[pid] = set()
                    address_coords[pid].add((entity['latitude'], entity['longitude']))
        
        # Process inventors
        for entity in row['front_page_entities']:
            if (entity['class'] == 'PER' and 
                'inventor_id' in entity and 
                entity['inventor_id'] is not None and 
                'person_id' in entity and 
                entity['person_id']):
                
                inventor_id = entity['inventor_id']
                person_id = entity['person_id'][0]
                
                if person_id in address_coords:
                    if inventor_id not in inventor_locations:
                        inventor_locations[inventor_id] = set()
                        first_years[inventor_id] = year
                    else:
                        first_years[inventor_id] = min(first_years[inventor_id], year)
                    
                    inventor_locations[inventor_id].update(address_coords[person_id])

    # Calculate maximum distances
    for inventor_id, locations in inventor_locations.items():
        locations_list = list(locations)
        if len(locations_list) > 1:
            max_dist = 0
            for i in range(len(locations_list)):
                for j in range(i + 1, len(locations_list)):
                    lat1, lon1 = locations_list[i]
                    lat2, lon2 = locations_list[j]
                    dist = haversine_distance(lat1, lon1, lat2, lon2)
                    max_dist = max(max_dist, dist)
            max_distances[inventor_id] = max_dist

    # Calculate yearly statistics
    year_location_stats = {}
    year_distance_stats = {}

    for inventor_id, locations in inventor_locations.items():
        first_year = first_years[inventor_id]
        
        if first_year not in year_location_stats:
            year_location_stats[first_year] = []
        year_location_stats[first_year].append(len(locations))
        
        if inventor_id in max_distances:
            if first_year not in year_distance_stats:
                year_distance_stats[first_year] = []
            year_distance_stats[first_year].append(max_distances[inventor_id])

    # Convert to averages and create time series
    yearly_avg_locations = pd.Series({
        year: np.mean(counts) for year, counts in year_location_stats.items()
    }).sort_index()

    yearly_avg_distances = pd.Series({
        year: np.mean(distances) for year, distances in year_distance_stats.items()
    }).sort_index()

    # Create and save first plot
    plt.figure(figsize=(10, 6))
    plt.bar(yearly_avg_locations.index, yearly_avg_locations.values, 
            width=0.8, color='#E24A33')
    plt.xlabel('Year of First Patent', fontsize=10)
    plt.ylabel('Average Number of Locations', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/avg_num_locations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create and save second plot
    plt.figure(figsize=(10, 6))
    plt.bar(yearly_avg_distances.index, yearly_avg_distances.values, 
            width=0.8, color='#348ABD')
    plt.xlabel('Year of First Patent', fontsize=10)
    plt.ylabel('Average Maximum Distance (km)', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/avg_max_distances.png', dpi=300, bbox_inches='tight')
    plt.close()



def generate_figure_8(kpst_scores):
    """
    Generate breakthrough scores plots.
    """
    def process_dataset(dataset):
        df = pd.DataFrame(dataset['train'])
        df = df[df['backward_similarity'] != 0]
        df = df[df['forward_similarity'] != 0]
        
        percentiles = df.groupby('year')['breakthrough_score'].agg([
            ('median', 'median'),
            ('75th', lambda x: x.quantile(0.75)),
            ('90th', lambda x: x.quantile(0.90)),
            ('95th', lambda x: x.quantile(0.95))
        ]).reset_index()
        
        return percentiles

    # Set style
    plt.style.use('ggplot')
    sns.set_palette("husl")

    # Create and save plots for each horizon
    for i, (horizon, dataset) in enumerate(kpst_scores.items()):
        percentiles = process_dataset(dataset)
        
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # Plot each percentile
        ax.plot(percentiles['year'], percentiles['median'],
                label='Median', linewidth=2)
        ax.plot(percentiles['year'], percentiles['75th'],
                label='75th percentile', linewidth=2)
        ax.plot(percentiles['year'], percentiles['90th'],
                label='90th percentile', linewidth=2)
        ax.plot(percentiles['year'], percentiles['95th'],
                label='95th percentile', linewidth=2)
        
        ax.set_title(f'Breakthrough Scores Over Time ({horizon})')
        ax.set_xlabel('Year')
        ax.set_ylabel('Breakthrough Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'output/breakthrough_scores_{horizon}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()



def generate_figure_9(kpst_scores):
    """
    Generate population-normalized quality plots.
    """
    # Load population data
    population = {}
    # DOWNLOAD DATA FROM: https://ourworldindata.org/grapher/population-of-england-millennium?time=1700..latest
    # Bank of England Millenium of Macroeconomic Data project has populations going back to 1700
    with open('temp/population-of-england-millennium.csv') as f:
        headers = f.readline().strip().split(',')
        year_idx = headers.index('Year')
        pop_idx = headers.index('Population (England)')
        for line in f:
            data = line.strip().split(',')
            population[int(data[year_idx])] = int(data[pop_idx])

    def process_dataset(dataset, population, forward_horizon):
        """Process individual dataset and create plot."""
        # Convert to pandas for residualization
        df = pd.DataFrame({
            'year': dataset['train']['year'],
            'score': dataset['train']['breakthrough_score']
        })
        
        # Residualize using statsmodels
        results = smf.ols(formula='score ~ C(year)', data=df).fit(cov_type='HC1')
        residuals = results.resid
        
        # Calculate threshold for top 10%
        high_threshold = np.percentile(residuals, 90)
        
        # Count high quality patents by year
        yearly_counts = {}
        for year, resid in zip(df['year'], residuals):
            if year not in yearly_counts:
                yearly_counts[year] = 0
            if resid >= high_threshold:
                yearly_counts[year] += 1
        
        # Normalize by population and create time series
        years = sorted(yearly_counts.keys())
        high_quality = []
        for year in years:
            if year in population:
                pop = population[year]
                high_quality.append(yearly_counts[year] / pop * 1000)
            else:
                high_quality.append(None)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(years, high_quality)
        plt.title(f'High Quality Patents per 1000 people ({forward_horizon})')
        plt.xlabel('Year')
        plt.ylabel('Number of patents / 1000 people')
        plt.grid(True)
        plt.savefig(f'output/high_quality_{forward_horizon}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Process each dataset
    for horizon, dataset in kpst_scores.items():
        process_dataset(dataset, population, horizon)



def generate_figure_10(kpst_scores):
    """
    Generate reform analysis plots.
    """
    def analyze_quality(dataset_dict, reform_year, quality_col='breakthrough_score', window=5):
        """Analyze quality distribution split at median."""
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset_dict['train'])
        
        # Residualize using statsmodels
        formula = f"{quality_col} ~ C(year)"
        model = sm.OLS.from_formula(formula, data=df)
        results = model.fit()
        df['residualized_quality'] = results.resid
        
        # Filter to reform window
        mask = (df['year'] >= reform_year - window) & (df['year'] <= reform_year + window)
        reform_df = df[mask].copy()
        
        # Split at median (calculated on full sample)
        median_quality = df['residualized_quality'].median()
        reform_df['above_median'] = reform_df['residualized_quality'] > median_quality
        
        # Calculate shares by year
        yearly_counts = reform_df.groupby(['year', 'above_median']).size().unstack()
        total_patents = yearly_counts.sum(axis=1)
        shares_df = yearly_counts.div(total_patents, axis=0) * 100
        
        # Rename columns for clarity
        shares_df.columns = ['Below Median', 'Above Median']
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(shares_df.index, shares_df['Above Median'],
                label='Above Median Quality',
                color='black',
                marker='o',
                linewidth=2)
        plt.plot(shares_df.index, shares_df['Below Median'],
                label='Below Median Quality',
                color='lightgray',
                marker='o',
                linewidth=2)
        plt.axvline(x=reform_year, color='red', linestyle='--',
                   label='Reform Year')
        plt.title(f'Breakthrough Scores around {reform_year}')
        plt.xlabel('Year')
        plt.ylabel('Share of Patents (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(f'output/patent_quality_median_{reform_year}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return shares_df

    # Analyze for both reform years using FH10-BH5 dataset
    residuals_1852 = analyze_quality(kpst_scores['fh10'], 1852, 
                                   quality_col='breakthrough_score')
    residuals_1883 = analyze_quality(kpst_scores['fh10'], 1883, 
                                   quality_col='breakthrough_score')



def main():

    # Load all required data
    dataset_all_years, dataset_all_entities, entities_df, kpst_scores = load_data()
    
    # Generate all figures
    generate_figure_2(dataset_all_years, entities_df)
    generate_figure_3(entities_df)
    generate_figure_4(dataset_all_entities)
    generate_figure_5(dataset_all_years)
    generate_figure_6(dataset_all_entities)
    generate_figure_7(dataset_all_entities)
    generate_figure_8(kpst_scores)
    generate_figure_9(kpst_scores)
    generate_figure_10(kpst_scores)

if __name__ == "__main__":
    main()