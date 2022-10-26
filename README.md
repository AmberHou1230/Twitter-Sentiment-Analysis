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
```

