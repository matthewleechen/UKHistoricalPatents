"""
Generate main figures for the 300 Years of British Patents paper.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd
from collections import Counter
import random
import re
from utils import load_base_datasets, load_kpst_scores, process_patent_inventors, standardize_occupation

# set seeds
random.seed(42)
np.random.seed(42)



def load_data():
    """load required datasets"""
    dataset_all_years, dataset_all_entities = load_base_datasets()
    entities_df = process_patent_inventors(dataset_all_entities)
    kpst_scores = load_kpst_scores()
    return dataset_all_years, dataset_all_entities, entities_df, kpst_scores



def generate_figure_2(dataset_all_years, entities_df):
    """create figure 2: patents and inventors over time"""
    # get patent counts
    patents_by_year = pd.Series(Counter(dataset_all_years['train']['year'])).sort_index()
    
    # get inventor counts
    first_patents = entities_df.groupby('inventor_id')['year'].min()
    new_inventor_counts = first_patents.value_counts().sort_index()
    
    # subset post-1800
    patents_1800 = patents_by_year[patents_by_year.index >= 1800]
    inventors_1800 = new_inventor_counts[new_inventor_counts.index >= 1800]
    
    # plot patent counts (levels)
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

    # plot patent counts (log)
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

    # plot inventor cohorts (levels) 
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

    # plot inventor cohorts (log)
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





def generate_table_1(entities_df):
    """create table 1: top occupations by period"""
    # get first patent year and standardize occupation
    first_patents = entities_df.groupby('inventor_id').agg({
        'year': 'min',
        'occupation': lambda x: pd.Series.mode(x)[0] if len(pd.Series.mode(x)) == 1 else x.iloc[0]
    }).reset_index()
    
    # standardize occupations
    first_patents['occupation'] = first_patents['occupation'].apply(standardize_occupation)
    
    # define periods
    periods = [
        (1617, 1750, '1617-1750'),
        (1751, 1800, '1751-1800'),
        (1801, 1850, '1801-1850'),
        (1851, 1899, '1851-1899')
    ]
    
    # process each period
    results = []
    for start_year, end_year, period_name in periods:
        period_data = first_patents[
            (first_patents['year'] >= start_year) &
            (first_patents['year'] <= end_year)
        ]
        total_inventors = len(period_data)
        occupation_counts = period_data['occupation'].value_counts()
        
        # get top 10
        for occ in occupation_counts.head(10).index:
            count = occupation_counts[occ]
            results.append({
                'period': period_name,
                'occupation': occ.title(),
                'count': count,
                'percentage': count/total_inventors * 100,
                'total': total_inventors
            })
    
    # create latex table
    latex_content = r'''\begin{table}[H]
\centering
\renewcommand{\arraystretch}{.65}
\caption{Top 10 Occupations}
\vspace{5mm}
\begin{tabular}{llrrr}
\toprule
\textbf{Period} & \textbf{Occupation} & \textbf{Count} & \textbf{Percentage} & \textbf{Total Inventors} \\
\midrule'''

    current_period = None
    for row in results:
        if current_period != row['period']:
            if current_period is not None:
                latex_content += r'\midrule' + '\n'
            current_period = row['period']
            period_str = row['period']
        else:
            period_str = ''
        
        line = f"""{period_str:<9} & {row['occupation']:<20} & {row['count']:>5,d} & {row['percentage']:>6.2f}\\% & {row['total']:,d} \\\\"""
        latex_content += '\n' + line
    
    latex_content += r'''
\bottomrule
\end{tabular}
\caption*{\textit{Note}: Table displays the top 10 most frequent occupations in each of four periods: 1617-1750, 1751-1800, 1801-1850, and 1851-1899. Occupations are standardized using a series of rules: we drop plural mentions of occupations, and group together common abbreviations for `gentleman' and `esquire'. We then take the modal occupation of inventors (or if there does not exist a mode for a particular inventor, we assign a random occupation). We display the count (number of inventors in that period with the given occupation), the percentage of patents associated with that occupation, and the total number of inventors in the period.}
\label{tab:occupations_total}
\end{table}'''

    # save table
    with open('output/occupations_total.tex', 'w') as f:
        f.write(latex_content)





def generate_figure_3(dataset_all_years):
    """create figure 3: patent classes analysis"""
    # prepare data
    df_years = pd.DataFrame({
        'year': dataset_all_years['train']['year'],
        'predicted_BPO_classes': dataset_all_years['train']['predicted_BPO_classes']
    })

    # define time periods
    year_subsets = {
        "1617-1750": (1617, 1750),
        "1751-1800": (1751, 1800),
        "1801-1850": (1801, 1850),
        "1851-1899": (1851, 1899),
        "Full Period": (1617, 1899),
    }

    for subset_name, (start_year, end_year) in year_subsets.items():
        # filter and process data
        subset_df = df_years[
            (df_years["year"] >= start_year) & 
            (df_years["year"] <= end_year)
        ]
        subset_exploded = subset_df.explode("predicted_BPO_classes")
        class_counts = subset_exploded["predicted_BPO_classes"].value_counts()
        top10 = class_counts.head(10)
        total_top10 = top10.sum()
        percentage_share = (top10 / total_top10) * 100

        # create plot
        plt.figure(figsize=(8, 6))
        sns.set_theme(style="white")
        ax = sns.barplot(
            x=percentage_share.values,
            y=percentage_share.index,
            color='#348ABD',
            orient='h'
        )

        ax.set_title(f"Top 10 Patent Classes ({subset_name})", fontsize=14)
        ax.set_xlabel("Percentage of Patents (%)", fontsize=12)
        ax.set_ylabel("Patent Class", fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=10)

        for i, (percent, label) in enumerate(zip(percentage_share.values, percentage_share.index)):
            ax.text(percent + 0.5, i, f"{percent:.1f}%", va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"output/top10_patent_classes_{subset_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()





def generate_figure_4(dataset_all_entities):
    """create figure 4: geographic distribution analysis"""
    # load and crop map
    europe = gpd.read_file('temp/europe.geojson')
    uk_bounds = {'minx': -11, 'miny': 49, 'maxx': 2, 'maxy': 61}
    europe_cropped = europe.cx[uk_bounds['minx']:uk_bounds['maxx'],
                             uk_bounds['miny']:uk_bounds['maxy']]

    # collect patent coordinates
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

    # create plots by period
    df = pd.DataFrame(patent_coords, columns=['year', 'longitude', 'latitude'])
    time_periods = [(1617, 1750), (1751, 1800), (1801, 1850), (1851, 1899)]
    
    for start, end in time_periods:
        # filter period data
        period_data = df[(df['year'] >= start) & (df['year'] <= end)]
        if period_data.empty:
            continue

        locations = period_data.groupby(['longitude', 'latitude']).size().reset_index(name='count')
        if locations.empty:
            continue

        # create scatter plot
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





def generate_figure_5(kpst_scores):
    """create figure 5: breakthrough scores analysis"""
    def process_dataset(dataset):
        """calculate percentiles for single dataset"""
        df = pd.DataFrame(dataset['train'])
        df = df[df['backward_similarity'] != 0]
        df = df[df['forward_similarity'] != 0]
        
        return df.groupby('year')['breakthrough_score'].agg([
            ('median', 'median'),
            ('75th', lambda x: x.quantile(0.75)),
            ('90th', lambda x: x.quantile(0.90)),
            ('95th', lambda x: x.quantile(0.95))
        ]).reset_index()

    # set plot style
    plt.style.use('ggplot')
    sns.set_palette("husl")

    # create plots for each horizon
    for horizon, dataset in kpst_scores.items():
        percentiles = process_dataset(dataset)
        
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
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
        plt.savefig(f'output/breakthrough_scores_{horizon}_BH5.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()






def main():
    """generate all main paper figures and tables"""
    # load data
    print("Loading datasets...")
    dataset_all_years, dataset_all_entities, entities_df, kpst_scores = load_data()
    
    # generate main paper content in order
    print("Generating Figure 2: Patents and inventors over time...")
    generate_figure_2(dataset_all_years, entities_df)
    
    print("Generating Table 1: Top occupations by period...")
    generate_table_1(entities_df)
    
    print("Generating Figure 3: Patent classes analysis...")
    generate_figure_3(dataset_all_years)
    
    print("Generating Figure 4: Geographic distribution...")
    generate_figure_4(dataset_all_entities)
    
    print("Generating Figure 5: Breakthrough scores...")
    generate_figure_5(kpst_scores)
    
    print("Done! All main paper content generated.")





if __name__ == "__main__":
    main()