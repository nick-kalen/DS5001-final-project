# text-analytics-final-project
DS 5001 Final Project on Harry Potter Books

This repository includes the following files:
- **data_model.ipynb**: this file contains the definitions of each table produced (and contained within the 'output' and 'output-viz' directories) along with definitions of each feature in each table. The file is linked in the final report.
- **exploratory_analysis.ipynb**: this file contains all work done to produce the tables needed for analysis
- **FINAL_REPORT.html**: this file is the HTML version of the final report which hides the code.
- **FINAL_REPORT.ipynb**: this file is the Jupyter Notebook version of the final report.
- **FINAL_REPORT.pdf**: this file is the PDF version of the final report which hides the code.
- **HarryPotterETA.py**: this file includes methods used to perform the exploratory analysis.
- **visualization.ipynb**: this file contains work done to produce many of the visualizations used in the final report.

This repository also contains the following directories:
- **data**: this directory contains the raw data in CSV format, embeddings from the semantic search, the sentiment analysis lexicon in CSV format, and the part of speech tagset used.
- **images**: this directory is where all images embedded in the final report are stored.
- **output**: this directory contains all CSV files listed as deliverables in the requirement for this project.
- **output-viz**: this directory contains all generated CSV files not listed as deliverables, but necessary to produce desired visualizations.

To export the FINAL_REPORT.ipynb file to HTML without losing images, run the following terminal command: `jupyter nbconvert FINAL_REPORT.ipynb --no-input --to html`