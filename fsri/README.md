The primary objective of this study was to explore and compare the effectiveness of machine learning models in classifying the severity of product recalls based on textual data. By analyzing 'Reason for Recall' and 'Product Description' from the FDA Data Dashboard, the goal was to determine which feature provides better predictive accuracy for categorizing recalls into Class I, II, and III. The study employed a robust Random Forest Classifier and TF-IDF vectorization to handle the high-dimensional and heterogeneous data effectively.

Biological hazards were the most frequent, with 12,559 occurrences, and had the highest average severity score of 2.702. These hazards, which include pathogens, bacteria, and other biological contaminants, pose significant health risks and are therefore classified as highly severe.

Chemical hazards, such as the presence of unauthorized additives or residues, followed with 7,498 instances and a severity score of 2.314. While not as severe as biological hazards, chemical contaminants can still cause serious health issues, necessitating prompt action.

Physical hazards, like the presence of foreign objects, and other miscellaneous hazards had fewer occurrences and lower severity scores of 1.992 and 1.959, respectively. While still concerning, these hazards are generally considered less severe than biological and chemical hazards.

The study revealed that Vegetables and Legumes, Dietary Specialties, and Packaged Meals and Kits are the product categories most affected by severe recalls, emphasizing the necessity for strict safety precautions and ongoing monitoring in these areas.

These findings have significant implications for public health and regulation. Regulatory agencies and manufacturers can better mitigate risks by prioritizing resources and interventions for the categories with the highest recall risk and severity.

Due to the study's limitations, future research should focus on developing more advanced analytical methods, such as deeper natural language processing, to improve recall classifications. Standardized recall reporting should also be promoted to improve data accuracy and uniformity across regulatory regimes.

Incorporating additional data sources, such as user feedback and real-time health data, could further enhance the study, enabling more dynamic and predictive food safety models that could forecast recall incidents and improve proactive public health actions.

## Bring your own data

No data ships with this repository. `food-recall-code-v2.ipynb` reads `file_path = '/content/FDA Data Dashboard v2.xlsx'`: an Excel export of FDA recalls from the FDA Data Dashboard (https://datadashboard.fda.gov), with at least the columns `Product Description`, `Reason for Recall` and `Event Classification` (values exactly `Class I`, `Class II` or `Class III`). Upload it to `/content` (the Colab default) or change `file_path` in each cell that loads it.
