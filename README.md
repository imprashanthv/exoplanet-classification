# **EXOPLANET CLASSIFICATION AND FINDING OUT WHY A PLANET MIGHT BE CLASSIFIED AS &#39;FALSE CANDIDATE&#39;**

**Introduction:**

The problem I choose to solve is to identify the classification of a Kepler object. A Kepler object is a celestial object transiting around a star. The classification of a Kepler object has two outcomes, whether the object is a planet or not. This stood out as a very important problem to tackle because of the following reasons:

1. There is a lot of data associated with every Kepler object, which made it a good machine learning problem.
2. Scientists around the world spend hundreds of hours to classify an object as a planet. In the process, they also spend the same amount of time to even classify an object as a false positive.

The input to the algorithm is measurements obtained from the Kepler space telescope, I then use Logistic Regression, SVM, Decision Tree, Random Forest to output a predicted classification of the object.

**Related Work:**

Although applying machine learning to astrophysics is a new concept and the depth of related work in this field is shallow, there were some fantastic works published in this area. Following are the related research works I came across.

The scientists at NASA Ames Research Center have developed a machine learning model to classify planets from &#39;Threshold crossing event&#39; data. A threshold crossing event (TCE) is a sequence of significant, periodic, planet transit-like features in the light curve of a target star. (McCauliff, et al. 2015). This work was done in 2015 and was the first of a kind in the field of exoplanet classification. They have used decision tree and random forest to create the machine learning models and ultimately achieved an error rate for classifying exoplanets of 2.81%. The most interesting thing about their work is that the techniques presented in their paper has helped scientists to identify new planets in the habitable zone. The main strength in their work is that the scientists who created the model were all astrophysicists with in-depth knowledge in their field. The approach they have selected is completely scientific and only a person of belonging to that field would be able to comprehend and quantify their work. the approach differs from theirs as I concentrated on the dataset and building the model without as much scientific approach, they have put in.

The next work I came across is done by researchers and professors at Southern Methodist University (Sturrock, et al. 2019). In this work, they have leveraged random forest classifier to classify the data. They have also created an API to access the model using a website. Although this work was produced more recently, the dataset they have selected contained a smaller number of data points and with the less amount of data, the model could be biased. Although there are few similarities in how they handled the data and how I handled the data, I tried to deviate from their approach of just building a classification model. In the work, I have also explored the data in-depth and have found out flags for an object to be considered a false positive.

The professors at University of Delaware have considered the similar approach to the work submitted by the researchers at Southern Methodist University (Priyadarshini, I., &amp; Puri, V. 2021). In their approach, they have used more advanced algorithms such as MLP and CNN and ultimately achieved an accuracy of 99.62% using their CNN model. They have done an incredible work in achieving a greater accuracy but the results of using this model in the real world are unknown.

With these previous works, I understood that the purpose of work so far is to aid the process of classifying exoplanets only and I saw that there is a need to understand and figure out a way to identify false positives as well. I strongly feel that early identification of false positives could result in a better and efficient exoplanet classification.

**Data:**

The dataset I obtained was from Kepler Object of Interest dataset. The dataset is publicly available in NASA archives. This dataset consists of information collected from Kepler Space Telescope. This data contains information of an object transiting around its host star NASA calls it threshold crossing event. During the orbiting of a planet around its host star, the light of its host star decreases to the observer (here, Kepler Space Telescope). The amount of light dimming helps to identify various characteristics of the object. With this information, I aimed to classify the objects as whether they are planets. The dataset was obtained directly from the Kepler Objects of Interest archive.

The dataset is available for the public to download on the following website [https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&amp;config=cumulative](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&amp;config=cumulative)

The dataset contains 9564 entries spawning over 50 columns. These columns represent the characteristics of the object such as transition time, orbital time, surface gravity of the object and its host star and so on. The feature column was selected as &#39;koi\_pdisposition&#39; which contains the classification of the objects.

![](RackMultipart20220616-1-u4hxt_html_c14678dd702d3cd4.png)

Figure showing the value counts of the koi\_pdisposition column

I went on exploring the data to find out what objects are classified as false positives and what objects are classified as candidates. The following are the results I found out.

Observation 1: Apparent Magnitude

![](RackMultipart20220616-1-u4hxt_html_fe543a845234b677.png) ![](RackMultipart20220616-1-u4hxt_html_bc1baa47d1ee789a.png)

Figures showing apparent magnitude of confirmed planets (left) and False Positives (right)

I found out that Objects containing greater than 18 Apparent Magnitude turned out to be false positives.

Observation 2: Orbital period

The next column I explored is the orbital period with respect to their koi score.

![](RackMultipart20220616-1-u4hxt_html_8a2a8f7adc4b72ab.png)

Figure showing orbital periods of objects with respect to their koi score

I found out that objects having orbital period greater than 550 had koi scores less than 0.5 meaning that these objects are less likely to be a planet.

Observation 3: Radius of the object and surface gravity

The next observation I made is the comparison of radius of the objects with respect to their surface gravities.

![](RackMultipart20220616-1-u4hxt_html_319fa24bd99bd2ff.png) ![](RackMultipart20220616-1-u4hxt_html_889a995d07e8ed71.png)

Figure showing radius of the object vs surface gravity of confirmed planets (left) and false candidates (right)

I found that most objects having surface gravity greater than 13000 turned out to be false positives.

Preprocessing:

The data consisted of few null values and some unimportant data. I removed the columns containing just IDs and names. I have filled few columns with the mean of the data. For other columns containing uncertainties, I dealt the uncertainties by applying linear regression by taking the base column as the X variable and the uncertainties column as the target variable. Previous works of others have either completely dropped the uncertainties columns are replaced with mean. Both the solutions are not feasible. dropping the columns means I are not considering the uncertainties. Replacing with means doesn&#39;t always mean that the value replaced is correct. Mean of the complete data tends toward higher values. I then have selected 30% of the entries (2869) for testing and the rest for training the data and have scaled the data to put all the values in the same range. I have also added a new column called density based on the surface gravity and object radius. The mathematical notation for adding density is as follows:

- Universal Gravitational Constant (G) = 6.67 () N /

Where N is Newton, m is meter, kg is kilogram

- Acceleration due to gravity (g) = (G\*M)/

Where M is the mass of the object and R is the radius of the object

- Since Mass of the object is unknown, I calculate mass by using the radius and the density.
 M = (4/3) \* \* \* d

Where is the irrational constant, R is the Radius and d is the density.

- Substituting the value of M in the equation of acceleration due to gravity gives
 g = (4/3) \* \* G \* d \* R
- And finally, density is given as:

density = (¾) \* g / (R \* \* G)

**Models/Methods:**

**Logistic Regression**

For the baseline algorithm, I used logistic regression, logistic regression is a statistical model that uses a logistic function to model a binary dependent variable. Also, logistic regression doesn&#39;t assume the type of the data and works well for any general classification.

**SVM Classifier**

The next method I used for modelling is Support Vector Machines Classifier. They were considered general purpose classifiers up until recently and Support Vector Machines work well in high dimensional spaces such as this dataset and hence, I used SVMs to classify the dataset. There are many parameters involved in SVM, So I experimented with various SVM parameters and finally set it to the default values as default values performed better than with tuning.

**Decision Tree Classifier**

Decision tree classifiers uses a decision tree to go from observations about an item to conclusions about the item&#39;s target. Since the data is not linear, I found out that tree-based classifiers work well for nonlinear data and can handle huge amounts of data containing many dimensions very well. Hence, I used decision tree classifiers to classify the objects.

**Random Forest**

Random Forests are an ensemble model, meaning that they are formed by multiple decision trees. The result of a random forest classifier is the average of n number of decision trees. To avoid bias and obtain a better result in general, I have used random forest classifier to classify the Kepler objects of interest.

**Evaluation:**

To evaluate the data, I have used accuracy score and confusion matrix as the metrics. For all the algorithms, the default values are used. The results of the model are as follows:

Logistic Regression

Accuracy Score: 0.9836236933797909

![](RackMultipart20220616-1-u4hxt_html_65ae3efb80e81021.png)

Figure showing the confusion matrix of Logistic Regression

SVM

Accuracy Score: 0.9822299651567944

![](RackMultipart20220616-1-u4hxt_html_2e7ce74a64f4b252.png)

Figure showing the confusion matrix of SVM Classifier

Decision Tree classifier

Accuracy Score: 0.9898954703832753

![](RackMultipart20220616-1-u4hxt_html_c86575933a03afa0.png)

Figure showing the confusion matrix of Decision Tree Classifier

Random Forest

Accuracy Score: 0.9878048780487805

![](RackMultipart20220616-1-u4hxt_html_67b99245ad0eaaea.png)

Figure showing the confusion matrix of Random Forest Classifier

**Conclusion:**

These conclusions are solely based on the understanding of the work done by me and the review of works done by others

Dataset consisted of 9564 entries spawning over 50 columns. In those 50 columns, there were 8 columns that held no significance. Hence, I dropped and continued without 8 columns.

I have split the data into train and test sets with the test size being 30% of the dataset

I have found the following flags for a False Positive object:

Based on the exploratory data analysis and visualization I have performed; I have noted the following flags:

- Apparent Magnitude - Objects containing greater than18 Apparent Magnitude turned out to be false positives.
- Orbital Period - Objects having orbital period greater than 550 had a koi\_score less than 0.5 meaning less likely to be a planet.
- Surface Gravity - Most objects having surface gravity greater than 13000 turned out to be false positives

Further Improvements:

- Within the given timeline, I was only able to research well into this topic and come up with one column, future works could benefit from the work I have already done and research about what other columns could be added to the dataset could be helpful.
- I have filled the missing values of few columns with their mean - researching on what other ways could the missing values be filled helps in increasing the accuracy

**References:**

[1] Sturrock, G. C., Manry, B., &amp; Rafiqi, S. (2019). Machine Learning Pipeline for Exoplanet

Classification. _SMU Data Science Review_, _2_(1), 9.

[2] McCauliff, S. D., Jenkins, J. M., Catanzarite, J., Burke, C. J., Coughlin, J. L., Twicken, J. D.,

&amp; Cote, M. (2015). Automatic classification of Kepler planetary transit candidates. _The Astrophysical__Journal_, _806_(1), 6.

[3] Priyadarshini, I., &amp; Puri, V. (2021). A convolutional neural network (CNN) based ensemble

model for exoplanet detection. _Earth Science Informatics_, _14_(2), 735-747.

[4] Bell, J., Harris, M., &amp; Salem, J. IDENTIFYING EXOPLANETS FROM TRANSIT

SURVEY DATA USING NEURAL NETWORKS.

[5] Pearson, K. A., Palafox, L., &amp; Griffith, C. A. (2018). Searching for exoplanets using artificial

intelligence. Monthly Notices of the Royal Astronomical Society, 474(1), 478-491.

[6] Pedregosa, F., Varoquaux, Ga&quot;el, Gramfort, A., Michel, V., Thirion, B., Grisel, O., … others.

(2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research,

12(Oct), 2825–2830.

[7] J. D. Hunter, &quot;Matplotlib: A 2D Graphics Environment&quot;, Computing in Science &amp;

Engineering, vol. 9, no. 3, pp. 90-95, 2007.

[8] Waskom, M. L., (2021). seaborn: statistical data visualization. Journal of Open Source

Software, 6(60), 3021, https://doi.org/10.21105/joss.03021