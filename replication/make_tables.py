"""
Generate all tables in the paper.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import re
import nltk
import datasets
from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from utils import (
    load_base_datasets,
    load_kpst_scores,
    process_patent_inventors,
    standardize_occupation
)

# Download required NLTK data
nltk.download('stopwords', quiet=True)


def load_data():
    
    # Load base datasets
    dataset_all_years, dataset_all_entities = load_base_datasets()
    
    # Process entities into dataframe
    entities_df = process_patent_inventors(dataset_all_entities)
    
    # Load KPST scores
    kpst_scores = load_kpst_scores()
    
    return dataset_all_years, dataset_all_entities, entities_df, kpst_scores



def generate_table_1(entities_df):
    """
    Generate table showing distribution of patents per inventor.
    """
    # Get count of patents per inventor
    patent_counts = entities_df.groupby('inventor_id')['patent_id'].nunique()
    
    # Count inventors for each number of patents
    distribution = {
        1: len(patent_counts[patent_counts == 1]),
        2: len(patent_counts[patent_counts == 2]),
        3: len(patent_counts[patent_counts == 3]),
        4: len(patent_counts[patent_counts == 4]),
        5: len(patent_counts[patent_counts >= 5])
    }
    total = sum(distribution.values())
    
    # Create LaTeX table
    latex_content = f"""\\begin{{table}}[H]
\\centering
\\renewcommand{{\\arraystretch}}{{.6}}
\\caption{{Distribution of Patents per Inventor}}
\\label{{tab:patents_by_inventor}}
\\bigskip
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
\\caption*{{\\textit{{Note}}: Table displays the number of inventors from our full coverage period that patented 1, 2, 3, 4, and at least 5, inventions.}}
\\end{{table}}"""

    # Save
    with open('output/patents_by_inventor.tex', 'w') as f:
        f.write(latex_content)




def generate_table_2(entities_df):
    """
    Generate table showing top 10 occupations by time period.
    """
    # Get first patent year and standardized occupation
    first_patents = entities_df.groupby('inventor_id').agg({
        'year': 'min',
        'occupation': lambda x: pd.Series.mode(x)[0] if len(pd.Series.mode(x)) == 1 else x.iloc[0]
    }).reset_index()
    
    # Standardize occupations
    first_patents['occupation'] = first_patents['occupation'].apply(standardize_occupation)
    
    # Define time periods
    periods = [
        (1617, 1750, '1617-1750'),
        (1751, 1800, '1751-1800'),
        (1801, 1850, '1801-1850'),
        (1851, 1899, '1851-1899')
    ]
    
    # Process each period
    results = []
    for start_year, end_year, period_name in periods:
        period_data = first_patents[
            (first_patents['year'] >= start_year) &
            (first_patents['year'] <= end_year)
        ]
        total_inventors = len(period_data)
        occupation_counts = period_data['occupation'].value_counts()
        
        # Get top 10 occupations
        for occ in occupation_counts.head(10).index:
            count = occupation_counts[occ]
            results.append({
                'period': period_name,
                'occupation': occ.title(),
                'count': count,
                'percentage': count/total_inventors * 100,
                'total': total_inventors
            })
    
    # Create LaTeX table
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

    # Save to file
    with open('output/occupations_total.tex', 'w') as f:
        f.write(latex_content)



    
def generate_table_3(dataset_all_years, entities_df):
    """
    Generate a LaTeX table of the most distinctive TF-IDF words for the top 30 occupations.
    """

    # Custom patent-specific stopwords
    PATENT_STOPWORDS = {
        'said', 'invention', 'described', 'end', 'figure', 'manner', 'means',
        'having', 'fig', 'thereof', 'whereby', 'thus', 'herein', 'being',
        'improved', 'improvement', 'improvements', 'claim', 'claims',
        'substantially', 'substantially as', 'specification', 'complete', 'provisional',
        'may', 'upon', 'part', 'parts', 'side', 'also', 'made', 'suitable',
        'used', 'position', 'nature', 'form', 'certain', 'required',
        'adapted', 'desired', 'connected', 'arranged', 'secured', 'formed',
        'placed', 'fixed', 'attached', 'provided', 'within', 'without',
        'through', 'whilst', 'subject', 'patent', 'apparatus', 'shown', 'one', 'two'
    }

    # Standardize occupations
    entities_df["std_occupation"] = entities_df["occupation"].apply(standardize_occupation)
    
    # Get the top 30 occupations by frequency
    occupation_counts = entities_df["std_occupation"].value_counts()
    top_occupations = occupation_counts.head(30).index.tolist()
    
    # Build patent->occupations map
    patent_to_occupations = defaultdict(list)
    for occ in top_occupations:
        pids = entities_df.loc[entities_df["std_occupation"] == occ, "patent_id"].unique()
        for pid in pids:
            patent_to_occupations[pid].append(occ)
    
    needed_patents = set(patent_to_occupations.keys())

    print(f"Number of needed patents: {len(needed_patents)}")

    # Filter dataset to needed patents
    train_data = dataset_all_years["train"]
    filtered_data = train_data.filter(lambda x: x["patent_id"] in needed_patents, batched=False, num_proc=4)

    # Process each patent to combine pages and get occupations
    def combine_pages(example):
        pages_sorted = sorted(example["full_text"], key=lambda x: x["page_num"])
        combined_text = " ".join(page["page_text"] for page in pages_sorted)
        pid = example["patent_id"]
        return {"combined_text": combined_text, "occupations": patent_to_occupations[pid]}

    processed_data = filtered_data.map(combine_pages, batched=False, num_proc=4)

    # Collect texts and occupation document indices
    texts = []
    occupation_doc_indices = defaultdict(list)
    for doc_idx, row in enumerate(tqdm(processed_data, desc="Collecting texts")):
        texts.append(row["combined_text"])
        for occ in row["occupations"]:
            if occ in top_occupations:
                occupation_doc_indices[occ].append(doc_idx)

    # TF-IDF Vectorization
    all_stopwords = set(stopwords.words("english")).union(PATENT_STOPWORDS)
    tfidf = TfidfVectorizer(
        stop_words=list(all_stopwords),
        ngram_range=(1, 1),
        max_features=1000,
        min_df=5,
        max_df=0.7,
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'  # Exclude short words <= 2 letters 
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()

    # Compute distinctive terms per occupation
    distinctive_terms = {}
    for occ in tqdm(top_occupations, desc="Processing occupations"):
        doc_indices = occupation_doc_indices.get(occ, [])
        if not doc_indices:
            distinctive_terms[occ] = []
            continue
        
        # Compute mean TF-IDF scores using sparse matrix operations
        sum_scores = tfidf_matrix[doc_indices].sum(axis=0)
        mean_scores = (sum_scores / len(doc_indices)).A1
        top_indices = mean_scores.argsort()[-10:][::-1]
        distinctive_terms[occ] = [feature_names[idx] for idx in top_indices]

    # Generate LaTeX table
    latex_content = r"""\begin{table}[H]
\centering
\caption{Most Distinctive Words (TF-IDF) for Top 20 Occupations}
\vspace{5mm}
\renewcommand{\arraystretch}{0.9}
\begin{tabular}{|l|p{12cm}|}
\hline
\textbf{Occupation} & \textbf{Distinctive Terms} \\
\hline"""

    for occ in top_occupations[:20]:
        terms = distinctive_terms.get(occ, [])
        latex_content += f"\n{occ} & {', '.join(terms)} \\\\\n\\hline"

    latex_content += r"""
\end{tabular}
\caption*{\textit{Note}: Table displays the most distinctive words by TF-IDF score in the full texts of patents associated with a given occupation. Occupations are standardized using a series of rules: we drop plural mentions of
occupations, and group together common abbreviations for `gentleman' and `esquire'. We then take the modal
occupation of inventors (or if there does not exist a mode for a particular inventor, we assign a random
occupation). TF-IDF scores are calculated using the \texttt{scikit-learn} TfidfVectorizer with the maximum number of features limited to 1,000. We restrict to words that are at least 3 letters in length, appear in at least 5 patents, and appear in at most 70\% of patents.}
\label{table:distinctive_terms}
\end{table}"""

    with open("output/distinctive_terms.tex", "w") as f:
        f.write(latex_content)

    print("Done! Table saved.")




def generate_table_4(kpst_scores):
    """
    Generate regression analysis table comparing breakthrough scores with quality measures.
    """
    # Load NTT quality data
    df_quality = pd.read_excel(
        "temp/Patent_Quality_England_1700_1850.xlsx", 
        sheet_name='Patent Quality Indicators'
    )
    
    # Process patent IDs
    df_quality['Patent_number_str'] = df_quality['Patent numebr'].astype(str).str.zfill(5)
    df_quality['patent_id'] = (
        'GB' + 
        df_quality['Year'].astype(str) + 
        df_quality['Patent_number_str'] + 
        'A'
    )
    
    # Create merged datasets
    merged_datasets = {}
    for horizon, dataset in kpst_scores.items():
        df = pd.DataFrame(dataset['train'])
        df = df[df['year'] <= 1850].merge(df_quality, on='patent_id', how='inner')
        merged_datasets[horizon] = df
    
    # Run regressions and collect results
    results_dict = {}
    y_vars = ['Patent_eminence', 'BCI']
    
    for name, df in merged_datasets.items():
        df = df.copy()
        
        # Filter zero similarities
        df = df[df['forward_similarity'] != 0]
        df = df[df['backward_similarity'] != 0]
        
        results_dict[name] = {}
        
        for y_var in y_vars:
            results_dict[name][y_var] = {}
            
            # No FE regression
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
    
    # Create LaTeX table
    def get_stars(p_value):
        if p_value < 0.01:
            return "^{***}"
        elif p_value < 0.05:
            return "^{**}"
        elif p_value < 0.1:
            return "^{*}"
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
    
    for name in ['fh1', 'fh5', 'fh10', 'fh20']:
        latex += f"\\multicolumn{{6}}{{l}}{{\\textbf{{{name.upper()}}}}} \\\\\n"
        
        # Coefficients
        latex += "Breakthrough Score "
        for y_var in y_vars:
            for fe in ['no_fe', 'with_fe']:
                res = results_dict[name][y_var][fe]
                stars = get_stars(res['p'])
                latex += f"& {res['coef']:.3f}{stars} "
            if y_var == 'Patent_eminence':
                latex += "&"
        latex += "\\\\\n"
        
        # Standard errors
        latex += "& "
        for y_var in y_vars:
            for fe in ['no_fe', 'with_fe']:
                res = results_dict[name][y_var][fe]
                latex += f"({res['se']:.3f}) & "
            if y_var == 'Patent_eminence':
                latex += "&"
        latex = latex.rstrip("& ") + "\\\\\n"
        
        # Mean dep var
        latex += "Mean Dep. Var. "
        for y_var in y_vars:
            res = results_dict[name][y_var]['no_fe']
            latex += f"& {res['mean_dep']:.3f} & {res['mean_dep']:.3f} "
            if y_var == 'Patent_eminence':
                latex += "&"
        latex = latex.rstrip() + "\\\\\n"
        
        # Fixed effects rows
        latex += "Year FE & No & Yes && No & Yes \\\\\n"
        latex += "Industry FE & No & Yes && No & Yes \\\\\n"
        
        # Observations
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
    
    # Save to file
    with open('output/regression_results_BCI.tex', 'w') as f:
        f.write(latex)




def generate_table_5(kpst_scores, entities_df):
    """
    Generate table showing occupations most overrepresented in breakthrough patents.
    """
    
    # Process KPST scores (use FH10-BH5)
    kpst_df = pd.DataFrame(kpst_scores['fh10']['train'])
    kpst_df = kpst_df[
        (kpst_df["forward_similarity"] != 0) &
        (kpst_df["backward_similarity"] != 0)
    ]
    
    # Standardize occupation & merge
    entities_df = entities_df.copy()
    entities_df["standardized_occupation"] = entities_df["occupation"].apply(lambda x: standardize_occupation(x))
    merged_df = pd.merge(
        entities_df[["patent_id", "standardized_occupation"]],
        kpst_df[["patent_id", "year", "breakthrough_score"]],
        on="patent_id",
        how="inner"
    )
    
    # Residualize breakthrough_score on year FE
    merged_df["year"] = merged_df["year"].astype(int)
    model = smf.ols("breakthrough_score ~ C(year)", data=merged_df).fit()
    merged_df["residual"] = model.resid
    
    # Get top decile
    cutoff = merged_df["residual"].quantile(0.90)
    top_decile_df = merged_df[merged_df["residual"] >= cutoff]
    
    # Compute statistics
    full_counts = merged_df.groupby("standardized_occupation").size().reset_index(name="count_full_sample")
    top_counts = top_decile_df.groupby("standardized_occupation").size().reset_index(name="count_top_decile")
    
    # Merge and filter
    occ_stats = pd.merge(
        full_counts, 
        top_counts, 
        on="standardized_occupation", 
        how="left"
    ).fillna({"count_top_decile": 0})
    
    # Filter to occupations with >= 100 appearances
    occ_stats = occ_stats[occ_stats["count_full_sample"] >= 100].copy()
    
    # Calculate ratios
    total_full_patents = occ_stats["count_full_sample"].sum()
    total_top_patents = top_decile_df.shape[0]
    
    occ_stats["fraction_in_full_sample"] = occ_stats["count_full_sample"] / total_full_patents
    occ_stats["fraction_in_top_decile"] = occ_stats["count_top_decile"] / total_top_patents
    occ_stats["overrepresentation_ratio"] = (
        occ_stats["fraction_in_top_decile"] / occ_stats["fraction_in_full_sample"]
    )
    
    # Handle infinities
    occ_stats["overrepresentation_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Sort and get top 20
    occ_stats_sorted = (occ_stats
        .dropna(subset=["overrepresentation_ratio"])
        .sort_values("overrepresentation_ratio", ascending=False)
    )
    top_20_overrep = occ_stats_sorted.head(20).copy()
    
    # Format columns
    top_20_overrep["count_full_sample"] = top_20_overrep["count_full_sample"].astype(int)
    top_20_overrep["count_top_decile"] = top_20_overrep["count_top_decile"].astype(int)
    top_20_overrep["overrepresentation_ratio"] = top_20_overrep["overrepresentation_ratio"].round(2)
    
    # Select columns for table
    cols_for_table = [
        "standardized_occupation",
        "count_full_sample",
        "count_top_decile",
        "overrepresentation_ratio"
    ]
    final_table = top_20_overrep[cols_for_table]
    
    # Manually create LaTeX table
    latex_table = "\\begin{table}[H]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Top 20 Overrepresented Occupations (Min 100 Appearances)}\n"
    latex_table += "\\label{tab:top_occ}\n"
    latex_table += "\\begin{tabular}{lrrr}\n"  
    latex_table += "\\toprule\n"
    
    # Add table headers
    latex_table += "Occupation & Count (Full Sample) & Count (Top Decile) & Overrepresentation Ratio \\\\\n"
    latex_table += "\\midrule\n"
    
    # Add table rows
    for index, row in final_table.iterrows():
        latex_table += f"{row['standardized_occupation']} & {row['count_full_sample']} & {row['count_top_decile']} & {row['overrepresentation_ratio']} \\\\\n"
    
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption*{\textit{Note}: This table shows occupations that are disproportionately represented in the top decile of the unconditional distribution of breakthrough scores (residualized by year fixed effects). Only occupations with at least 100 appearances in the full sample are included.}\n"
    latex_table += "\\end{table}\n"

    
    # Save to file
    with open("output/top_20_overrep_occupations_min100.tex", "w") as f:
        f.write(latex_table)




def main():

    # Load datasets
    dataset_all_years, dataset_all_entities, entities_df, kpst_scores = load_data()
    
    # Generate tables
    generate_table_1(entities_df)
    generate_table_2(entities_df)
    generate_table_3(dataset_all_years, entities_df)
    generate_table_4(kpst_scores)
    generate_table_5(kpst_scores, entities_df)



if __name__ == "__main__":
    main()