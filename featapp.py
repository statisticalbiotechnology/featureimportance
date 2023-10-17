import streamlit as st
import pandas as pd
import numpy as np
import bisect
import altair as alt

df_w = pd.read_table('percolator.weights.txt')
weights = [ float(w) for w in df_w.loc[1,:].values.flatten() ]
feature_labels = df_w.loc[5,:].values.flatten().tolist()[:-1]

df_X = pd.read_table("perc_input_test.tsv")

feature_data = {}   # aggregated values to visualize for each feature
N_PSM = len(df_X)
N_BINS = 500
min_weighted = 0
max_weighted = 0

# compute the weighted feature values, and aggregate by feature label instead of per PSM
for index,row in df_X.iterrows():
    for i,feature in enumerate(feature_labels):
        weighted_value = row[feature] * weights[i] #+ weights[-1]
        min_weighted = min(min_weighted, weighted_value)
        max_weighted = max(max_weighted, weighted_value)

        if (feature in feature_data):
            feature_data[feature]['weighted_values'].append(weighted_value)
            feature_data[feature]['orig_values'].append(row[feature])
        else:
            feature_data[feature] = {'weighted_values' : [weighted_value], 'orig_values' : [row[feature]]}

BIN_SIZE = (max_weighted - min_weighted) / N_BINS
bins = [bin_left for bin_left in np.arange(min_weighted, max_weighted, BIN_SIZE)]
bins.append(max_weighted - BIN_SIZE)

# compute the normalized histogram values per feature
for feature in feature_data:
    histogram = [ [bin_left + BIN_SIZE/2, 0] for bin_left in bins]

    for feature_value in feature_data[feature]['weighted_values']:
        bin_idx = bisect.bisect_left(bins, feature_value) - 1
        histogram[bin_idx][1] += 1

    for bin_idx in range(len(bins)):
        histogram[bin_idx][1] = histogram[bin_idx][1] / N_PSM

    feature_data[feature]['histogram'] = histogram

plot_data = []
MIN_BAR_VAL = 0.05

# create a data frame to plot
for i,feature in enumerate(feature_data):
    min_orig = min(feature_data[feature]['orig_values'])
    max_orig = max(feature_data[feature]['orig_values'])
    #if (float(weights[i]) == 0):
    #    continue

    for bin in feature_data[feature]['histogram']:
        if (bin[1] > 0):
            bin_center_orig = (bin[0] / weights[i]) if (weights[i] != 0) else -1
            plot_data.append([bin[0], bin[1], 0.5 - max(MIN_BAR_VAL, bin[1]) / 2, 0.5 + max(MIN_BAR_VAL, bin[1]) / 2, (bin_center_orig - min_orig) / (max_orig - min_orig) if (float(weights[i]) != 0) else -1, bin_center_orig, feature.strip()])

plot_df = pd.DataFrame(data=plot_data, columns=['Contribution', 'PointCount', 'Y1', 'Y2', 'OrigValueNorm', 'OrigValue', 'FeatureLabel'])

row_height = 20

c = ( alt.Chart(plot_df, width=600).mark_rule(size=2,)
    .encode(
        x=alt.X("Contribution", title="Feature contribution", axis=alt.Axis(gridColor="red", tickCount=1)),
        x2="Contribution",
        row=alt.Row("FeatureLabel", spacing=8, header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=12)),
        y=alt.Y("Y1", title=None, scale=alt.Scale(domain=[0, 1]), axis=None),
        y2="Y2",
        color=alt.Color("OrigValueNorm", scale=alt.Scale(domainMid=0.5, domain=[0,1], reverse=True), legend=alt.Legend(title="Feature value (normalized)", titleFontSize=13)),
        tooltip=['PointCount', 'OrigValue'])
    .properties(
        height=row_height,  # Set the height of the rows
        title="")
    .interactive()
)

st.altair_chart(c, use_container_width=True)

