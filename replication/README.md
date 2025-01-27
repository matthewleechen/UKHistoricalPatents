## Replication 

To replicate all the tables and figures in our paper, first:
```bash
pip install -r requirements.txt
```
Then, prepare `temp/` by copying the following temporary data files:
1. `Patent_Quality_England_1700_1850.xlsx` (download link [here](https://www.openicpsr.org/openicpsr/project/142801/version/V1/view )) 
2. `europe.geojson` (original GEOJSON is [here](https://github.com/leakyMirror/map-of-europe/blob/master/GeoJSON/europe.geojson))
3. `population-of-england-millennium.csv` (download link [here](https://ourworldindata.org/grapher/population-of-england-millennium.csv?v=1&csvType=full&useColumnShortNames=true))

Then, run:
```terminal
python make_figures.py 
```
and 
```terminal 
python make_tables.py 
```
The code will output .tex and .png files in `output/`.
