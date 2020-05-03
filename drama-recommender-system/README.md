# Drama Recommender Systems

I often use the filter function to find dramas to watch, or rely on the 'Recommendations' tab on the website https://mydramalist.com/shows/top, where viewers suggest similar dramas based on plot/characters etc.

There is definitely a smarter system, just like how YouTube recommends videos to us. I read up a few articles on building recommender systems and tried my own project. The process was similar to the drama reviews project:

1. Web scrape for drama data - using BeautifulSoup
2. Combine data
3. Exploratory data analysis - Data visalization
4. Recommender systems using machine learning 
   - Simple recommender system
   - Content-based recommender system - similarity matrix using scikit-learn
   - Collaborative filtering recommender system - matrix factorization & k-Nearest Neighbors of scikit-learn
5. Recommender systems using deep learning - collab learner of fast.ai

## (1) Web scraping drama information using BeautifulSoup

To get a list of drama reviews, I had to (1) identify the list of dramas posted on the dramas website, and (2) extract individual details from each dramas.

Information retrieved from the drama reviews include

```
drama_title
year
main_actors
genres
tags
synopsis
```

These information were then stored in a DataFrame and saved as a CSV file for subsequent analysis.

## (2) Combine data using Pandas

Since there were thousands of dramas on the website, I scraped the website in batches. Then I used Panda's concat function to join these CSV files together.

## (3) Exploratory data analysis

EDA was conducted to find out the most commonly used actors, genres and tags via **word clouds**.

## (4) Drama recommender systems using machine learning

A quick search of the web reveals different methods of recommending content. I explored three of them in my modelling - simple recommender system, content-based recommender system and collaborative filtering recommender system. 

There is no "better" recommender system since they each have their pros and cons. Furthermore, each returned a separate result given the same drama. 

As a start with not much information, a **simple recommender system** can be used. This returns the top rated dramas, taking into consideration the number of people who voted.

### Content-based recommender system

I tried the **plot description based recommender** as well as the **main actors, genres and keywords recommender**. Both of which uses the same concept of identifying the commonly used words, assigning a **cosine similarity** score and returning the values that give the highest similarity scores.

Results from plot description based recommender:
```
>> get_recommendations('Story of Yanxi Palace (2018)')
4306                                 The Switch II (2002)
3197                           The Emperor's Harem (2011)
1782                               Emperors and Me (2019)
1068                 Granting You a Dreamlike Life (2018)
1308                          Suddenly This Summer (2018)
4939                                     Tiger Mom (2015)
3128                          The Rippling Blossom (2011)
2886                           Empress Myeongseong (2001)
4178    The Voyage of Emperor Qian Long to Jiang Nan (...
1192                         Legend of the Phoenix (2019)
```

Results from main actors, genres and keywords recommender:
```
>> get_recommendations('Story of Yanxi Palace (2018)', cosine_sim2)
4836                    Jao Sao Ban Rai (2006)
774                The Flame's Daughter (2018)
4812                           Prissana (2000)
962                          Ngao Asoke (2016)
4941                     Kehas See Dang (2011)
3992                       Maiden's Vow (2006)
2477                      Rosy Business (2009)
1585                       Plerng Naree (2016)
4598                       I'm So Smart (2013)
555     Ruyi's Royal Love in the Palace (2018)
```

### Collaborative filtering

Instead of a drama based approach, we can reference the users to give recommendations. I used the **k-Nearest Neighbours** and **matrix factorization** approaches. The k-Nearest Neighbour model uses a similarity metrics, such as the cosine similarity, to find its k closest neighbours based on drama_title, user_name and overall_rating. 

On the other hand, matrix factorization uses the Pearson's R correlation coefficient to find dramas that have a correlation score.

Results from k-Nearest Neighbours:
```
Recommendations for Story of Yanxi Palace (2018):

1: The Legend of Hao Lan (2019), with distance of 0.6804742115494322:
2: Our Glamorous Time (2018), with distance of 0.7359146481821208:
3: The Story of Ming Lan (2018), with distance of 0.7376371435874067:
4: Scarlet Heart 2 (2014), with distance of 0.7408084469827373:
5: Nirvana in Fire 2: The Wind Blows in Chang Lin (2017), with distance of 0.7560108381816948:
```

Results from matrix factorization:
```
Recommendations for Story of Yanxi Palace (2018):

['Doctor Prisoner (2019)',
 'Secrets of Three Kingdoms (2018)',
 'Story of Yanxi Palace (2018)']
```

My personal take is that the collaborative filtering system could reflect the user sentiment better than the content based recommender system. Interestingly, the dramas suggested under the CF system are similar to the drama whose recommendations were asked for in terms of plot and story characters. This could be due to the users having a predisposed preference on the type of dramas and acting as a "content based recommender".

## (5) Drama recommender systems using deep learning

fast.ai introduced a **collab learner** that can be used in collaborative filtering. Interesting findings include the generalization of some dramas being rated highly/lowly regardless of what userws are rating them. For instance, dramas that have high bias are generally highly-rated and vice versa.

Highly-rated dramas:
```
drama_title
Healer (2014)                         231
Goblin (2016)                         204
You Who Came from the Stars (2013)    182
Kill Me, Heal Me (2015)               162
City Hunter (2011)                    151
W (2016)                              150
The Master's Sun (2013)               150
Eternal Love (2017)                   149
Descendants of the Sun (2016)         149
I Hear Your Voice (2013)              148
```

High-biased dramas:
```
[(tensor(0.8848), 'The Untamed (2019)', 9.48611111111111, 108),
 (tensor(0.8408), 'Addicted Heroin (2016)', 8.645833333333334, 72),
 (tensor(0.8181), 'Eternal Love (2017)', 9.241610738255034, 149),
 (tensor(0.8141), 'Love O2O (2016)', 8.940972222222221, 144),
 (tensor(0.7768), 'Signal (2016)', 9.712765957446809, 94),
 (tensor(0.7685), 'Goblin (2016)', 9.272058823529411, 204),
 (tensor(0.7129), 'Nirvana in Fire (2015)', 9.54, 75),
 (tensor(0.7114),
  'Weightlifting Fairy Kim Bok Joo (2016)',
  9.132575757575758,
  132),
 (tensor(0.6913), 'My Mister (2018)', 9.358024691358025, 81),
 (tensor(0.6862), 'Love by Chance (2018)', 9.237179487179487, 78)]
```

As we can see, there are several overlaps between the dramas found in the highly-rated drama list and the high-biased drama list. Same goes for the low bias and lowly-rated drama list. This may mean that the general perception of this highly-rated dramas are good, and can be suggested to new users even before knowing their preferences. We should also seek to avoid recommending dramas with low bias. 
