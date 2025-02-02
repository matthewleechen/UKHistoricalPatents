"""
Generate appendix tables and figures for 300 Years of British Patents paper.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import Counter, defaultdict
from math import radians, sin, cos, sqrt, atan2
import random
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import load_base_datasets, load_kpst_scores, process_patent_inventors, standardize_occupation

# download nltk data
nltk.download('stopwords', quiet=True)

# set seeds
random.seed(42)
np.random.seed(42)

def load_data():
    """load all required datasets"""
    dataset_all_years, dataset_all_entities = load_base_datasets()
    entities_df = process_patent_inventors(dataset_all_entities)
    kpst_scores = load_kpst_scores()
    return dataset_all_years, dataset_all_entities, entities_df, kpst_scores

def generate_table_b1(entities_df):
    """create table b1: distribution of patents per inventor"""
    # get patent counts
    patent_counts = entities_df.groupby('inventor_id')['patent_id'].nunique()
    
    # count inventors for each number of patents
    distribution = {
        1: len(patent_counts[patent_counts == 1]),
        2: len(patent_counts[patent_counts == 2]),
        3: len(patent_counts[patent_counts == 3]),
        4: len(patent_counts[patent_counts == 4]),
        5: len(patent_counts[patent_counts >= 5])
    }
    total = sum(distribution.values())
    
    # create latex table
    latex_content = f"""\\begin{{table}}[H]
\\centering
\\renewcommand{{\\arraystretch}}{{.6}}
\\caption{{Distribution of Patents per Inventor}}
\\label{{tab:patents_by_inventor}}
\\begin{{tabular}}{{cc}}
\\hline
\\\\[-0.5em]
Number of Patents & Number of Inventors \\\\
\\\\[-0.5em]
\\hline
\\\\[-0.5em]
1 & {distribution[1]:,} \\\\
2 & {distribution[2]:,} \\\\
3 & {distribution[3]:,} \\\\
4 & {distribution[4]:,} \\\\
$\\geq 5$ & {distribution[5]:,} \\\\
\\\\[-0.5em]
\\hline
\\\\[-0.5em]
Total & {total:,} \\\\
\\\\[-0.5em]
\\hline
\\end{{tabular}}
\\caption*{{\\textit{{Note}}: Shows number of inventors by patent count across full sample period.}}
\\end{{table}}"""

    # save table
    with open('output/appendix_b1_patents_by_inventor.tex', 'w') as f:
        f.write(latex_content)

def generate_figure_b1(entities_df):
    """create figure b1: co-invention analysis plots"""
    # count inventors per patent
    patents_by_inventors = entities_df.groupby(['year', 'patent_id'])['inventor_id'].nunique()
    coinvented = patents_by_inventors > 1
    coinvention_count = coinvented.groupby('year').sum()
    coinvention_share = coinvented.groupby('year').mean()

    # plot number of co-invented patents
    plt.figure(figsize=(10, 6))
    plt.bar(coinvention_count.index, coinvention_count.values, width=0.8, color='#E24A33')
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Number of Co-invented Patents', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/appendix_b1a_coinvention_count.png", dpi=300, bbox_inches='tight')
    plt.close()

    # plot share of co-invented patents
    plt.figure(figsize=(10, 6))
    plt.bar(coinvention_share.index, coinvention_share.values * 100, width=0.8, color='#348ABD')
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Share of Co-invented Patents (%)', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/appendix_b1b_coinvention_share.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_table_b2(dataset_all_years, entities_df):
    """create table b2: most distinctive words by occupation"""
    # custom patent stopwords
    patent_stopwords = {
        'said', 'invention', 'described', 'end', 'figure', 'manner', 'means',
        'having', 'fig', 'thereof', 'whereby', 'thus', 'herein', 'being',
        'improved', 'improvement', 'improvements', 'claim', 'claims',
        'substantially', 'specification', 'complete', 'provisional'
    }

    # standardize occupations
    entities_df["std_occupation"] = entities_df["occupation"].apply(standardize_occupation)
    
    # get top 30 occupations
    occupation_counts = entities_df["std_occupation"].value_counts()
    top_occupations = occupation_counts.head(30).index.tolist()
    
    # build patent->occupations map
    patent_to_occupations = defaultdict(list)
    for occ in top_occupations:
        pids = entities_df.loc[entities_df["std_occupation"] == occ, "patent_id"].unique()
        for pid in pids:
            patent_to_occupations[pid].append(occ)
    
    # get needed patents
    needed_patents = set(patent_to_occupations.keys())
    filtered_data = dataset_all_years["train"].filter(
        lambda x: x["patent_id"] in needed_patents, 
        batched=False, 
        num_proc=4
    )

    # combine pages for each patent
    def combine_pages(example):
        pages_sorted = sorted(example["full_text"], key=lambda x: x["page_num"])
        combined_text = " ".join(page["page_text"] for page in pages_sorted)
        pid = example["patent_id"]
        return {"combined_text": combined_text, "occupations": patent_to_occupations[pid]}

    processed_data = filtered_data.map(combine_pages, batched=False, num_proc=4)

    # collect texts and occupation indices
    texts = []
    occupation_doc_indices = defaultdict(list)
    for doc_idx, row in enumerate(processed_data):
        texts.append(row["combined_text"])
        for occ in row["occupations"]:
            if occ in top_occupations:
                occupation_doc_indices[occ].append(doc_idx)

    # calculate tf-idf
    all_stopwords = set(stopwords.words("english")).union(patent_stopwords)
    tfidf = TfidfVectorizer(
        stop_words=list(all_stopwords),
        ngram_range=(1, 1),
        max_features=1000,
        min_df=5,
        max_df=0.7,
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()

    # get distinctive terms per occupation
    distinctive_terms = {}
    for occ in top_occupations:
        doc_indices = occupation_doc_indices.get(occ, [])
        if not doc_indices:
            distinctive_terms[occ] = []
            continue
        
        sum_scores = tfidf_matrix[doc_indices].sum(axis=0)
        mean_scores = (sum_scores / len(doc_indices)).A1
        top_indices = mean_scores.argsort()[-10:][::-1]
        distinctive_terms[occ] = [feature_names[idx] for idx in top_indices]

    # create latex table
    latex_content = r"""\begin{table}[H]
\centering
\caption{Most Distinctive Words by Occupation}
\begin{tabular}{|l|p{12cm}|}
\hline
\textbf{Occupation} & \textbf{Distinctive Terms} \\
\hline"""

    for occ in top_occupations[:20]:
        terms = distinctive_terms.get(occ, [])
        latex_content += f"\n{occ} & {', '.join(terms)} \\\\\n\\hline"

    latex_content += r"""
\end{tabular}
\caption*{\textit{Note}: Shows most distinctive words by TF-IDF score in patents by occupation.}
\label{table:distinctive_terms}
\end{table}"""

    # save table
    with open("output/appendix_b2_distinctive_terms.tex", "w") as f:
        f.write(latex_content)

def generate_figure_b2(dataset_all_entities):
    """create figure b2: communicated patents analysis"""
    # count patents
    total_patents = Counter(dataset_all_entities['train']['year'])
    communicated_patents = Counter()
    
    # identify communicated patents
    for row in dataset_all_entities['train']:
        if any(entity['class'] == 'COMM' for entity in row['front_page_entities']):
            communicated_patents[row['year']] += 1
    
    # create time series
    total_patents_series = pd.Series(total_patents).sort_index()
    communicated_patents_series = pd.Series(communicated_patents).sort_index()
    communicated_share = pd.Series({
        year: communicated_patents[year] / total_patents[year]
        for year in total_patents.keys()
    }).sort_index()

    # plot counts
    plt.figure(figsize=(10, 6))
    plt.bar(communicated_patents_series.index, communicated_patents_series.values,
            width=0.8, color='#E24A33')
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Number of Communicated Patents', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/appendix_b2a_communicated_count.png", dpi=300, bbox_inches='tight')
    plt.close()

    # plot shares
    plt.figure(figsize=(10, 6))
    plt.bar(communicated_share.index, communicated_share.values * 100,
            width=0.8, color='#348ABD')
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Share of Communicated Patents (%)', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/appendix_b2b_communicated_share.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_figure_b3(dataset_all_entities):
    """create figure b3: inventor mobility analysis"""
    def haversine_distance(lat1, lon1, lat2, lon2):
        """calculate great circle distance between points"""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return c * 6371  # radius in km

    # collect inventor locations
    inventor_locations = {}
    first_years = {}
    max_distances = {}

    # extract address coordinates
    for row in dataset_all_entities['train']:
        year = row['year']
        address_coords = {}
        
        # get coordinates from address entities
        for entity in row['front_page_entities']:
            if (entity['class'] == 'ADD' and 
                'person_id' in entity and 
                entity['person_id'] is not None and
                'latitude' in entity and 
                'longitude' in entity):
                
                coords = (entity['latitude'], entity['longitude'])
                if coords[0] is not None and coords[1] is not None:
                    for pid in entity['person_id']:
                        if pid not in address_coords:
                            address_coords[pid] = set()
                        address_coords[pid].add(coords)
        
        # map addresses to inventors
        for entity in row['front_page_entities']:
            if (entity['class'] == 'PER' and 
                'inventor_id' in entity and 
                'person_id' in entity and 
                entity['person_id']):
                
                inv_id = entity['inventor_id']
                pid = entity['person_id'][0]
                
                if pid in address_coords:
                    if inv_id not in inventor_locations:
                        inventor_locations[inv_id] = set()
                        first_years[inv_id] = year
                    else:
                        first_years[inv_id] = min(first_years[inv_id], year)
                    inventor_locations[inv_id].update(address_coords[pid])

    # calculate maximum distances
    for inv_id, locations in inventor_locations.items():
        locations_list = list(locations)
        if len(locations_list) > 1:
            max_dist = max(
                haversine_distance(lat1, lon1, lat2, lon2)
                for i, (lat1, lon1) in enumerate(locations_list)
                for lat2, lon2 in locations_list[i+1:]
            )
            max_distances[inv_id] = max_dist

    # aggregate yearly statistics
    year_locations = defaultdict(list)
    year_distances = defaultdict(list)
    
    for inv_id, locations in inventor_locations.items():
        first_year = first_years[inv_id]
        year_locations[first_year].append(len(locations))
        if inv_id in max_distances:
            year_distances[first_year].append(max_distances[inv_id])

    # create time series
    avg_locations = pd.Series({
        year: np.mean(counts) for year, counts in year_locations.items()
    }).sort_index()

    avg_distances = pd.Series({
        year: np.mean(dists) for year, dists in year_distances.items()
    }).sort_index()

    # plot locations
    plt.figure(figsize=(10, 6))
    plt.bar(avg_locations.index, avg_locations.values, width=0.8, color='#E24A33')
    plt.xlabel('Year of First Patent', fontsize=10)
    plt.ylabel('Average Number of Locations', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/appendix_b3a_avg_locations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plot distances
    plt.figure(figsize=(10, 6))
    plt.bar(avg_distances.index, avg_distances.values, width=0.8, color='#348ABD')
    plt.xlabel('Year of First Patent', fontsize=10)
    plt.ylabel('Average Maximum Distance (km)', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/appendix_b3b_avg_distances.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_table_b3(kpst_scores):
    """create table b3: breakthrough scores and quality measures regression"""
    # load quality data
    df_quality = pd.read_excel(
        "temp/Patent_Quality_England_1700_1850.xlsx", 
        sheet_name='Patent Quality Indicators'
    )
    
    # process patent IDs
    df_quality['Patent_number_str'] = df_quality['Patent numebr'].astype(str).str.zfill(5)
    df_quality['patent_id'] = (
        'GB' + 
        df_quality['Year'].astype(str) + 
        df_quality['Patent_number_str'] + 
        'A'
    )
    
    # merge datasets
    merged_data = {}
    for horizon, dataset in kpst_scores.items():
        df = pd.DataFrame(dataset['train'])
        df = df[df['year'] <= 1850].merge(df_quality, on='patent_id', how='inner')
        merged_data[horizon] = df
    
    # run regressions
    results_dict = {}
    y_vars = ['Patent_eminence', 'BCI']
    
    for name, df in merged_data.items():
        df = df[
            (df['forward_similarity'] != 0) & 
            (df['backward_similarity'] != 0)
        ].copy()
        
        results_dict[name] = {}
        for y_var in y_vars:
            results_dict[name][y_var] = {}
            
            # baseline regression
            results = sm.OLS.from_formula(
                f"{y_var} ~ breakthrough_score", 
                data=df
            ).fit(cov_type='HC3')
            
            results_dict[name][y_var]['no_fe'] = {
                'coef': results.params['breakthrough_score'],
                'se': results.bse['breakthrough_score'],
                'p': results.pvalues['breakthrough_score'],
                'n': int(results.nobs),
                'mean_dep': df[y_var].mean()
            }
            
            # with fixed effects
            results_fe = sm.OLS.from_formula(
                f"{y_var} ~ breakthrough_score + C(year) + C(Industry)", 
                data=df
            ).fit(cov_type='HC3')
            
            results_dict[name][y_var]['with_fe'] = {
                'coef': results_fe.params['breakthrough_score'],
                'se': results_fe.bse['breakthrough_score'],
                'p': results_fe.pvalues['breakthrough_score'],
                'n': int(results_fe.nobs),
                'mean_dep': df[y_var].mean()
            }
    
    # create latex table with significance stars
    def get_stars(p_value):
        """get significance stars based on p-value"""
        if p_value < 0.01: return "^{***}"
        elif p_value < 0.05: return "^{**}"
        elif p_value < 0.1: return "^{*}"
        return ""
    
    latex = """\\begin{table}[H]
\\setlength{\\tabcolsep}{4pt}
\\def\\arraystretch{0.9}  
\\caption{Breakthrough Scores and Patent Quality Measures}
\\centering 
\\begin{tabular}{lccccc}
\\hline\\hline
& \\multicolumn{2}{c}{Patent Eminence} && \\multicolumn{2}{c}{BCI} \\\\
& (1) & (2) && (3) & (4) \\\\
\\hline
"""
    
    # add results by horizon
    for name in ['fh1', 'fh5', 'fh10', 'fh20']:
        latex += f"\\multicolumn{{6}}{{l}}{{\\textbf{{{name.upper()}}}}}\\\\\n"
        
        # coefficients and stars
        latex += "Breakthrough Score "
        for y_var in y_vars:
            for fe in ['no_fe', 'with_fe']:
                res = results_dict[name][y_var][fe]
                stars = get_stars(res['p'])
                latex += f"& {res['coef']:.3f}{stars} "
            if y_var == 'Patent_eminence':
                latex += "&"
        latex += "\\\\\n"
        
        # standard errors
        latex += "& "
        for y_var in y_vars:
            for fe in ['no_fe', 'with_fe']:
                res = results_dict[name][y_var][fe]
                latex += f"({res['se']:.3f}) & "
            if y_var == 'Patent_eminence':
                latex += "&"
        latex = latex.rstrip("& ") + "\\\\\n"
        
        # mean dependent variable
        latex += "Mean Dep. Var. "
        for y_var in y_vars:
            res = results_dict[name][y_var]['no_fe']
            latex += f"& {res['mean_dep']:.3f} & {res['mean_dep']:.3f} "
            if y_var == 'Patent_eminence':
                latex += "&"
        latex = latex.rstrip() + "\\\\\n"
        
        # fixed effects indicators
        latex += "Year FE & No & Yes && No & Yes \\\\\n"
        latex += "Industry FE & No & Yes && No & Yes \\\\\n"
        
        # observations
        latex += "Observations "
        for y_var in y_vars:
            res = results_dict[name][y_var]['no_fe']
            latex += f"& {res['n']} & {res['n']} "
            if y_var == 'Patent_eminence':
                latex += "&"
        latex = latex.rstrip() + "\\\\\n"
        latex += "\\hline\n"
    
    latex += """\\hline
\\multicolumn{6}{l}{\\footnotesize{Notes: * p < 0.1, ** p < 0.05, *** p < 0.01. Robust SEs in parentheses.}} \\\\
\\end{tabular}
\\end{table}"""
    
    # save table
    with open('output/appendix_b3_regression_results.tex', 'w') as f:
        f.write(latex)



def generate_figure_b4(kpst_scores):
    """create figure b4: breakthrough patents over time (population-normalized)"""
    # load population data
    population = {}
    with open('temp/population-of-england-millennium.csv') as f:
        headers = f.readline().strip().split(',')
        year_idx = headers.index('Year')
        pop_idx = headers.index('Population (England)')
        for line in f:
            data = line.strip().split(',')
            population[int(data[year_idx])] = int(data[pop_idx])

    def process_dataset(dataset, population, forward_horizon):
        """process single dataset and create plot"""
        # prepare data
        df = pd.DataFrame({
            'year': dataset['train']['year'],
            'score': dataset['train']['breakthrough_score']
        })
        
        # residualize scores
        results = smf.ols(formula='score ~ C(year)', data=df).fit(cov_type='HC3')
        residuals = results.resid
        high_threshold = np.percentile(residuals, 90)
        
        # count high quality patents by year
        yearly_counts = defaultdict(int)
        for year, resid in zip(df['year'], residuals):
            if resid >= high_threshold:
                yearly_counts[year] += 1
        
        # normalize by population
        years = sorted(yearly_counts.keys())
        high_quality = [
            yearly_counts[year] / population[year] * 1000 if year in population else None
            for year in years
        ]
        
        # create plot
        plt.figure(figsize=(10, 6))
        plt.plot(years, high_quality)
        plt.title(f'High Quality Patents per 1000 people ({forward_horizon})')
        plt.xlabel('Year')
        plt.ylabel('Patents per 1000 people')
        plt.grid(False)
        plt.savefig(f'output/appendix_b4_{forward_horizon}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # generate plots for each horizon
    for horizon, dataset in kpst_scores.items():
        process_dataset(dataset, population, horizon)



def generate_figure_b5(kpst_scores):
    """create figure b5: breakthrough patenting around reform years"""
    def analyze_quality(dataset_dict, reform_year, window=5):
        """analyze quality distribution around reform year"""
        # prepare data
        df = pd.DataFrame(dataset_dict['train'])
        
        # residualize scores
        model = sm.OLS.from_formula('breakthrough_score ~ C(year)', data=df)
        results = model.fit()
        df['residualized_quality'] = results.resid
        
        # filter to reform window
        mask = (df['year'] >= reform_year - window) & (df['year'] <= reform_year + window)
        reform_df = df[mask].copy()
        
        # split at median
        median_quality = df['residualized_quality'].median()
        reform_df['above_median'] = reform_df['residualized_quality'] > median_quality
        
        # calculate shares
        yearly_counts = reform_df.groupby(['year', 'above_median']).size().unstack()
        total_patents = yearly_counts.sum(axis=1)
        shares_df = yearly_counts.div(total_patents, axis=0) * 100
        shares_df.columns = ['Below Median', 'Above Median']
        
        # create plot
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
        plt.grid(False)
        plt.savefig(f'output/appendix_b5_{reform_year}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()



def generate_table_b4(kpst_scores, entities_df):
    """create table b4: top 20 overrepresented occupations in breakthrough patents"""
    # process KPST scores using fh10
    kpst_df = pd.DataFrame(kpst_scores['fh10']['train'])
    kpst_df = kpst_df[
        (kpst_df["forward_similarity"] != 0) &
        (kpst_df["backward_similarity"] != 0)
    ]
    
    # standardize and merge occupation data
    entities_df = entities_df.copy()
    entities_df["std_occupation"] = entities_df["occupation"].apply(standardize_occupation)
    merged_df = pd.merge(
        entities_df[["patent_id", "std_occupation"]],
        kpst_df[["patent_id", "year", "breakthrough_score"]],
        on="patent_id",
        how="inner"
    )
    
    # residualize breakthrough scores
    merged_df["year"] = merged_df["year"].astype(int)
    model = smf.ols("breakthrough_score ~ C(year)", data=merged_df).fit()
    merged_df["residual"] = model.resid
    
    # identify top decile patents
    cutoff = merged_df["residual"].quantile(0.90)
    top_decile_df = merged_df[merged_df["residual"] >= cutoff]
    
    # compute occupation statistics
    full_counts = merged_df.groupby("std_occupation").size().reset_index(name="count_full")
    top_counts = top_decile_df.groupby("std_occupation").size().reset_index(name="count_top")
    
    # merge and filter to frequent occupations
    occ_stats = pd.merge(full_counts, top_counts, on="std_occupation", how="left")
    occ_stats = occ_stats[occ_stats["count_full"] >= 100].fillna({"count_top": 0})
    
    # calculate representation ratios
    total_full = occ_stats["count_full"].sum()
    total_top = occ_stats["count_top"].sum()
    occ_stats["frac_full"] = occ_stats["count_full"] / total_full
    occ_stats["frac_top"] = occ_stats["count_top"] / total_top
    occ_stats["overrep_ratio"] = occ_stats["frac_top"] / occ_stats["frac_full"]
    
    # get top 20
    top_20 = (occ_stats
        .sort_values("overrep_ratio", ascending=False)
        .head(20)
        .round({'overrep_ratio': 2})
    )
    
    # create latex table
    latex = """\\begin{table}[H]
\\centering
\\caption{Top 20 Overrepresented Occupations in Breakthrough Patents}
\\label{tab:overrep_occ}
\\begin{tabular}{lrrr}
\\toprule
Occupation & Count (Full) & Count (Top) & Overrep. Ratio \\\\
\\midrule"""

    for _, row in top_20.iterrows():
        latex += f"\n{row['std_occupation']} & {int(row['count_full']):,d} & "
        latex += f"{int(row['count_top']):,d} & {row['overrep_ratio']:.2f} \\\\"
    
    latex += """
\\bottomrule
\\caption*{\\textit{Note}: Shows occupations most overrepresented in top decile of 
breakthrough scores (residualized by year). Limited to occupations with ≥100 patents.}
\\end{tabular}
\\end{table}"""

    # save table
    with open("output/appendix_b4_overrep_occupations.tex", "w") as f:
        f.write(latex)

def main():
    """generate all appendix tables and figures"""
    # load data
    print("Loading datasets...")
    dataset_all_years, dataset_all_entities, entities_df, kpst_scores = load_data()
    
    # generate content in order
    print("Generating Table B1: Patents per inventor...")
    generate_table_b1(entities_df)
    
    print("Generating Figure B1: Co-invention analysis...")
    generate_figure_b1(entities_df)
    
    print("Generating Table B2: Distinctive words...")
    generate_table_b2(dataset_all_years, entities_df)
    
    print("Generating Figure B2: Communicated patents...")
    generate_figure_b2(dataset_all_entities)
    
    print("Generating Figure B3: Inventor mobility...")
    generate_figure_b3(dataset_all_entities)
    
    print("Generating Table B3: Breakthrough regression...")
    generate_table_b3(kpst_scores)
    
    print("Generating Figure B4: Population-normalized breakthrough patents...")
    generate_figure_b4(kpst_scores)
    
    print("Generating Figure B5: Reform analysis...")
    generate_figure_b5(kpst_scores)
    
    print("Generating Table B4: Overrepresented occupations...")
    generate_table_b4(kpst_scores, entities_df)
    
    print("Done! All appendix content generated.")

if __name__ == "__main__":
    main()