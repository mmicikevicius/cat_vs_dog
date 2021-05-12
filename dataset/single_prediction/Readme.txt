Machine Learning - Project for TCU Course (10/2020) - by Fábio Gomes
Business challenge - Thousands of legislative proposals are presented annually to the Chamber of Deputies. The adoption of thematic classification carried out automatically would facilitate activities to monitor the processing of bills (PL) and other types, such as proposals for inspection and control (PFC), according to thematic health groups, increasing transparency for society and streamlining legislative drafting and enforcement activities. Research group of the Chamber of Deputies on Legislative and Health, of which I participate, developed a hierarchical typology for the classification of legislative proposals related to health, containing four thematic groups (also has subcategories) and it is intended to use this base to produce a model capable of classifying new propositions automatically.
Built solution - I developed this ML project for over a year (supervised classification of texts of bills - PL - related to health). There were many problems in converting pdf to text, as the Chamber used several types of pdf over time. In August 2019, some models were developed using the R (base with 7575 PL). Random Forest was the most promising model. This experience allowed the detection of classification errors in the training set and some intuitions about the database and even about problems in the typology used for the classification. This course allowed the learning of Python and the resumption of the project.
The current project, developed in the Machine Learning Course in Projects (TCU class - 2020 - Prof. Erick Muzart) is a supervised classification of PL texts from 2011 to 2014 related to health and its thematic groups.
The steps of this project included:
Stage 1 (in 2019): classify 8,327 PL from 2011 to 2014 (human coders) related to health: “yes” (2,328 projects), “no” (5,999 projects).
Step 2 (in 2019): convert the content of the files in "pdf" to "txt" (7,575).
Stage 3 (in 2019): build the database (PL ids, PL texts and health codes).
Step 4: apply supervised machine learning models (using Python codes from instructor Fernando Melo).
The models divide the data set for training (80%) and testing (20%) and generate predictions for the latter.
Step 5: calculate model accuracy, recall, precision and f1 score.
Distribution of variables:
Independent variable: average per PL -> 844 words (std 1,963), corpus of PL -> 33,700 tokens;
Dependent variable - unbalanced data (most non-health: 72%);
Relative frequency of thematic health groups: Assistance - 3.8%; Management and resources - 4.0%; Prevention- 15%; Rights- 6.7%; Others - 0.
26 experiments were carried out using notebooks on Google Colab (see summary table of results).

Results:
NLP improved performance obtained in Project R (RF).
The accuracy was higher (89.7% -> 91% - SGD, SVM and RF) and more health cases were captured (before there were 296 -> 326) - f1 score 0.82 in the 3 models.
Stemmer in general improved the performance of the models.
Classification of thematic groups had a lower performance than that of the health group (high accuracy, but low recall), but the health dimension was captured.
Errors detected to be corrected.
Data source - The entire text of the legislative proposals can be extracted from pdf files, made available on the Internet on the Chamber of Deputies' open data website. Ref: https://dadosabertos.camara.leg.br/swagger/api.html#staticfile. The list of propositions and their classification codes, identified by humans, are available in spreadsheets on this website.

The following are available on this website:
Study on the health agenda in progress at the Chamber of Deputies, to exemplify the application of this type of information;
Manual with description of health categories;
Google Colab notebook with Random Forest model application;
Databases (csv) - one containing PL characteristics and classification and the other with PL texts.
Synthesis table with results of experiments.