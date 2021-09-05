import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
# first calculate their BMI by dividing their weight in kilograms by the square of their height in meters. As height is in CM, multiply by 100 to get meters. Using Lambda to compare the BMI and apply a value based on if overweight or not.
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x : 1 if x >25 else 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
# Using lambda to apply values and normalize the data.
df['cholesterol'] = df['cholesterol'].apply(lambda x : 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x : 0 if x == 1 else 1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    # unpivot the dataframe using cardio as the identifier variable, and the others as the value variables. 
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    # rename the column
    df_cat["total"] = 1
    # group by cardio value, and show total counts, but do not include cardio as index
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index =False).count()

    # Draw the catplot with 'sns.catplot()'
    # using seaborn, creating a bar chart that uses the categorical data frame variables on the x axis, the total count values on the y axis, and separates them based on cardio value. 
    fig = sns.catplot(x= 'variable', y= 'total', data = df_cat, hue='value', kind='bar', col = 'cardio').fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    # filter the data by selecting correct values for heart rate, height, and weight
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    # use Pandas dataframe correlation on df_heat with pearson method for standard correlation coefficient
    corr = df_heat.corr(method='pearson')

    # Generate a mask for the upper triangle
    # Get the upper triangle of the corr matrix using The NumPy triu() function
    mask = np.triu(corr)



    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize =(12,12) )

    # Draw the heatmap with 'sns.heatmap()'
    # Seaborn uses the figure created by matplotlib, and the mask set by the triu function to generate this heatmap
    sns.heatmap(corr, linewidths=1, annot=True, square=True, mask = mask, fmt = '.1f', center =0.08, cbar_kws = {'shrink':0.5})


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
