# Fusarium_standard_area_diagram_2024
The repository for the code to make most of the figures and conduct statistics for the paper entitled "Standard area diagram for rating Fusarium oxysporum in sugar beet (Beta vulgaris)" available at:

https://doi.org/XXXXXXXXXXX

See paper methods for collection and curation procedures.

Code/Input files

	Photo library:

		<BvFus_SAD_v1_imageset> The directory containing the images given to participants

		<BvFus_SAD_v2_imageset> The directory containing the images given to participants

	R:

		input data: 

		<alldata1.csv> is the file that contains version 1 rating data and version 2 rating data.

		<baysianPQ.csv> is the data that renames version 2 ratings as "human" and includes ratings from 5 machine learning models.

		code:

		<Fus_SAD_R.Rmd> R markdown script detailing the statistical analysis and figure generation.

PlantCV:

All files were made following the RGB tutorial on PlantCV's website.

The included files are not an exhaustive list of each script used in the manuscript, but rather an example of a base script that can be tailored for a specific use.

	<naive_bayes_pdf3.txt> The pixel sample file which is generated by the <plantcv-train.py> script, as part of the PlantCV suite.

	<configlocal.json> The configuration file required by the PlantCV suite, adjusted with metadata for this study.

	<machine_learning.py> The file which calculates the disease score as determined by the configuration file.

Note, the plantcv-workflow.py script is not included in these files, as there is nothing you need to change in the basic PlantCV provided script.
