# Drama Reviews
I love watching dramas, and often look up this website 'https://mydramalist.com/shows/top' to find top dramas to watch. With some new Python skills that I have picked up, I wanted to do something related to dramas such as:

1. Web scrape the drama reviews - using BeautifulSoup
2. Preprocess the drama reviews
3. Conduct exploratory data analysis on the reviews - Data Visualization
4. Identify topics present in the reviews - Topic Modelling using Latent Dirichlet Allocation (LDA)
5. Uncover the sentiment of viewers - Sentiment Analysis using sklearn
6. Uncover the sentiment of viewers part II - Sentiment Analysis using Keras RNN

## (1) Web scraping drama reviews with BeautifulSoup

To get a list of drama reviews, I had to (1) identify the list of dramas posted on the dramas website, and (2) extract individual reviews from each dramas.

Information retrieved from the drama reviews include 

```
1. drama_title
2. user_name
3. overall_rating
4. story_rating
5. cast_rating
6. music_rating
7. rewatch_value_rating
8. reviews
```

These information were then stored in a DataFrame and saved as a CSV file for subsequent analysis.

## (2) Data preprocessing

The extracted information cannot be used directly since there were miscellaneous information. These include non-Enligh reviews, stopwords, and extra characters, such as those that demarcated a new paragrah. **Lemmatization** and **tokenization** were part of the preprocessing steps as well.

In this step, the sentiments were grouped into three classes - positive _(overall_rating 6-10)_, neutral _(overall_rating 5)_, and negative _(overall_rating 1-4)_.

After data preprocessing, the information was stored back into a DataFrame.

## (3) Exploratory data analysis

EDA is always useful for ask to get a broad overview of the data, including the distribution of ratings and length of reviews. **Word clouds** were also plotted to identify the most commonly used words. This has revealed that top words in positive and negative reviews differ.

## (4) Topic modelling using latent dirichlet allocation 

This is a separate analysis from the sentiment analysis with the purpose of learning something new - **latent dirichlet allocation**. The Gensim library was used to identify common topics amongst all the reviews. 

Although some topics were rather similar, these topics came out rather distinct.

```
(5,
  '0.025*"life" + 0.023*"school" + 0.021*"family" + 0.017*"people" + 0.016*"student" + 0.016*"love" + 0.015*"friend" + 0.014*"girl" + 0.013*"man" + 0.013*"young"')
```

This topic could indicate a youth drama surrounding college life, with the themes of friendship or romance.

```
(1,
  '0.068*"time" + 0.063*"actor" + 0.034*"day" + 0.029*"boy" + 0.029*"main" + 0.025*"break" + 0.018*"year" + 0.018*"drama" + 0.017*"love" + 0.017*"episode"')
```

Despite this seeming like a romance drama with the word "love", it may not be one with a happy storyline. A common word includes "break", which could signify a break up. Furthermore, besides the word love, it does not contain other positive keywords like "good" and "great" as in a more upbeat review as follows.

```
(0,
  '0.060*"drama" + 0.029*"story" + 0.029*"character" + 0.023*"good" + 0.017*"episode" + 0.013*"time" + 0.010*"love" + 0.010*"great" + 0.010*"music" + 0.009*"many"')
```


## (5) Sentiment analysing using Scikit Learn

### Approach

Before we input the reviews into the model, we had to first convert texts into integers using **vectorization**, namely term-frequency inverse document frequency (TF-IDF). Then, we trained four models - Random Forest Classifier, Logistic Regression, Support Vector Machine and K-Nearest Neighbours - with the standard paramaters. The RFC gave us the best results with 

```
Test accuracy = 0.945276612742161
```

Although a large proportion of positive sentiment was accurately predicted, a significant proportion of negative and neutral sentiment reviews were incorrectly predicted.

Confusion matrix:
```
Actual     0    1     2
Pred 0 [[  51    0  177]
     1 [   0   35   97]
     2 [   0    0 4647]]
```

Thus, this was changed to a binary classification problem instead, where we predict positive v negative sentiments. In addition, to minimise the impact of an unbalanced dataset, **downsampling of the majority class** was done. Next, the top model in the first part - **Random Forest Classifier** - was used.

### Results

```
Test accuracy = 0.8308351177730193
```

Although test accuracy has fallen, it can be attributed to a more accurate measurement with the balanced dataset.

Confusion matrix:
```
Actual    0    1
Pred 0 [[205  30]
     1 [ 49 183]]
```

Here, we can see that the proportion of wrongly classified positive and negative reviews remain small compared to the correctly classified ones.

## (6) Sentiment analysing using Keras RNN

### Approach

An LSTM model is trained on drama reviews to predict the sentiment of reviews. It was a binary classification problem where we tried to distinguish between positive _(label 1)_ and negative _(label 0)_ sentiments. The model that returned the best result was one that used the **sigmoid activation function** and had an **additional dropout layer**. **Early stopping** was used to prevent over-training of the dataset. 

### Results

Although the validation loss did not fall as significantly as the training loss, the model was still able to differentiate between positive and negative sentiments.

Test samples:
```
test_sample1 = 'This drama is horrible.'
test_sample2 = 'Love the actors and music. Best drama in 2019'
test_sample3 = 'I think this drama is so-so. Maybe a 5/10'
test_sample4 = 'Enjoyed it thoroughly. Although people say the plot progressed very slowly, it hooked me throughout the dramas.'
```

Output probability:
```
array([[0.25155318],
       [0.9995559 ],
       [0.77527547],
       [0.9981889 ]], dtype=float32)
```

Here, a value closer to 1 reveals a positive sentiment while a value closer to 0 reveals a negative sentiment. We can see that the first sample was accurrately predicted as negative while the second and forth ones were accurately predicted as positive. Interestingly, the model was able to distinguish a somewhat neutral one from negative and positive (sample 3). Therefore, a threshold can be set to differentiate between positive and negative sentiments.
