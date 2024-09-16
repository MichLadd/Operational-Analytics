import matplotlib.pyplot as plt
import pandas as pd

# The Time series (1 - Predictive Analytics)

fig, axs = plt.subplots(2, 2)  # return x row and y columns to fill with plots
fig.tight_layout(pad=3.0)  # padding of chart (distance)
fig.set_figheight(6)  # height of container
fig.set_figwidth(10)  # width of container

fig.suptitle('Lesson 2 Lab py')


# International airline passengers: monthly totals in thousands. Jan 49 – Dec 60 (G.E.P. Box, G.M. Jenkins, 1976).
dfBox = pd.read_csv("data/BoxJenkins.csv")
axs[0][0].plot(dfBox.iloc[:].Month,
               dfBox.iloc[:].Passengers, label="Passengers")
axs[0][0].set(ylabel='Passengers', xlabel=('Month'))
axs[0][0].legend()
axs[0][0].title.set_text('International airline passengers')

# Slides example (fil rouge)
dfFil = pd.read_csv("data/FilRouge.csv")
axs[0][1].plot(dfFil.iloc[:].t, dfFil.iloc[:].sales, label="Sales")
axs[0][1].set(ylabel='Sales', xlabel='Period')
axs[0][1].legend()
axs[0][1].title.set_text('Fil Rouge - Tv Sales')

# Another time series Jewelry market sales, USA (millions of $).
dfGioiellerie = pd.read_csv(
    "data/gioiellerie.csv")
axs[1][0].plot(dfGioiellerie.iloc[:].year,
               dfGioiellerie.iloc[:].sales, label="Sales")
axs[1][0].set(ylabel='Sales', xlabel=('Year'))
axs[1][0].legend()
axs[1][0].title.set_text('Jewelry market sales - Year')

axs[1][1].bar(dfGioiellerie.iloc[:].month,
              dfGioiellerie.iloc[:].sales, label="Sales")
axs[1][1].set(ylabel='Sales', xlabel=('Month'))
axs[1][1].legend()
axs[1][1].title.set_text('Jewelry market sales - Month')

plt.show()