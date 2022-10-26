# Twitter-Sentiment-Analysis

Import Libraries and Datasets
------------------------
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
!apt jupyterthemes
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 
```

```
# Load the data
tweets_df = pd.read_csv('twitter.csv')
tweets_df
```

![tweetsdf](tweetsdf.png)

```
tweets_df.info()
```

![tweetsInfo](tweetsInfo.png)

```
tweets_df.describe()
```

![tweetsDescribe](tweetsdescribe.png)

```
# Drop the 'id' column
tweets_df['tweet']
tweets_df = tweets_df.drop(['id'], axis=1)
```

Perform Data Exploration
-----------------------------
```
sns.heatmap(tweets_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
```

![heatmap](emptyHeatmap.png)

```
tweets_df.hist(bins = 30, figsize = (13,5), color = 'r')
```

![firstHis](firstHisRed.png)

```
sns.countplot(tweets_df['label'], label = "Count") 
```

![secondHis](2ndHisBlue.png)

Let's get the length of the messages
```
tweets_df['length'] = tweets_df['tweet'].apply(len)
tweets_df
tweets_df.describe()
# Let's see the shortest message 
tweets_df[tweets_df['length'] == 11]['tweet'].iloc[0] # it's 'i love you '
# Let's view the message with mean length 
tweets_df[tweets_df['length'] == 84]['tweet'].iloc[0]
'my mom shares the same bihday as @user   bihday snake! see you this weekend ð\x9f\x99\x8cð\x9f\x8f¼'
# Plot the histogram of the length column
tweets_df['length'].plot(bins=100, kind='hist')
```

![Historgram of the length column](histogramoflength.png)

Plot the WordCloud (My fav part!)
------------------


