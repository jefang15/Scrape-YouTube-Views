""" Scrape - YouTube Views """



# Import packages
from requests_html import HTMLSession
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker
from tabulate import tabulate
import mplcursors
import seaborn as sns



" List of Video Links to Track Views "
# Add new video links here as needed

video_links = [
    # Some Hearts
    'https://www.youtube.com/watch?v=lydBPm2KRaU',  # Jesus, Take the Wheel
    'https://www.youtube.com/watch?v=lmAi_qJoPbU',  # Don't Forget to Remember Me
    'https://www.youtube.com/watch?v=WaSy8yy-mr8',  # Before He Cheats
    'https://www.youtube.com/watch?v=paMzF1lnwGg',  # Wasted

    # Carnival Ride
    'https://www.youtube.com/watch?v=nEQj6RrQbgA',  # So Small
    'https://www.youtube.com/watch?v=m36xv75MJ4U',  # All-American Girl
    'https://www.youtube.com/watch?v=f27zNlmRMWU',  # Last Name
    'https://www.youtube.com/watch?v=jLntFKtR66g',  # Just A Dream
    'https://www.youtube.com/watch?v=NzpR4eRxd8E',  # I Told You So

    # Play On
    'https://www.youtube.com/watch?v=oM7NQQ0Lfu4',  # Cowboy Casanova
    'https://www.youtube.com/watch?v=LraOiHUltak',  # Temporary Home
    'https://www.youtube.com/watch?v=ywtJYvDBKek',  # Undo It
    'https://www.youtube.com/watch?v=bpFW4Yhy08k',  # Mama's Song

    'https://www.youtube.com/watch?v=7qzhngp7jh8',  # Remind Me

    # Blown Away
    'https://www.youtube.com/watch?v=7-uothzTaaQ',  # Good Girl
    'https://www.youtube.com/watch?v=pJgoHgpsb9I',  # Blown Away
    'https://www.youtube.com/watch?v=oVEBZLrjpw4',  # Two Black Cadillacs
    'https://www.youtube.com/watch?v=vTnWFT3DvVA',  # See You Again

    'https://www.youtube.com/watch?v=o4Yzj-m_SBk',  # Somethin' Bad

    # Greatest Hits: Decade 1
    'https://www.youtube.com/watch?v=mH9kYn4L8TI',  # Something in the Water
    'https://www.youtube.com/watch?v=kxOYvI0pfLo',  # Little Toy Guns

    # Storyteller
    'https://www.youtube.com/watch?v=3ealNayCkaU',  # Smoke Break
    'https://www.youtube.com/watch?v=eg50JGPEoo8',  # Heartbeat
    'https://www.youtube.com/watch?v=N2-yVryNjUM',  # Church Bells
    'https://www.youtube.com/watch?v=lNzHARgbCG8',  # Dirty Laundry

    # Cry Pretty
    'https://www.youtube.com/watch?v=KUUjtUP2CrE',  # Cry Pretty
    'https://www.youtube.com/watch?v=-Py8OWAMkns',  # Love Wins
    'https://www.youtube.com/watch?v=pc4zBnvLt3g',  # Southbound
    'https://www.youtube.com/watch?v=vfkq-Ov7WSw',  # Drinking Alone

    # My Gift
    'https://www.youtube.com/watch?v=4KiFSKLTj0U',  # Hallelujah

    'https://www.youtube.com/watch?v=o6teH-xJn5o',  # Tears of Gold

    # My Savior
    'https://www.youtube.com/watch?v=Yf6C0L_7-CA',  # How Great Thou Art
    'https://www.youtube.com/watch?v=L6XMSPJzfEU',  # Nothing But The Blood Of Jesus

    'https://www.youtube.com/watch?v=d4mh4jq_MYU',  # I Wanna Remember
    
    'https://www.youtube.com/watch?v=kt7VSlX1HgY',  # Only Us

    'https://www.youtube.com/watch?v=Zc3cxj5pDIs',  # If I Didn't Love You

    # CU7
    'https://www.youtube.com/watch?v=4Y7flhznvnE'  # Ghost Story (Official Lyric Video)

    ]


" Blank List to Store Scraped Data "
out = []


" Function "
def youtube_info(url):
    session = HTMLSession()  # init an HTML Session
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"}
    video_url = url  # YouTube video url
    response = session.get(video_url, headers=headers)  # Get the URL contents
    response.html.render(sleep=20, timeout=20)  # Set timeout
    soup = bs(response.html.html, features="html.parser")  # Make URL contents easy to read
    # print(soup.prettify())  # Open html
    # open("video.html", "w", encoding='utf8').write(response.html.html)  # Write all HTML code into a file

    # Date pulled
    pull_date = date.today()

    # Upload Date
    upload_date = soup.find("div", {"id": "info-strings"}).find("yt-formatted-string").text

    # Channel Name
    channel_name = soup.find("yt-formatted-string", {"class": "ytd-channel-name"}).find("a").text

    # Video Title
    video_title = soup.find("div", {"id": "container", "class": "style-scope ytd-video-primary-info-renderer"}).find(
        "h1").text
    print(video_title)

    # Duration
    # video_duration = soup.find("span", {"class": "ytp-time-duration"}).text

    # Views
    video_views = int(''.join([c for c in soup.find("span", attrs={"class": "view-count"}).text if c.isdigit()]))

    # Likes & Dislikes
    # text_yt_formatted_strings = soup.find_all("yt-formatted-string",
    #                                           {"id": "text", "class": "ytd-toggle-button-renderer"})
    # video_likes = text_yt_formatted_strings[0].text
    # video_dislikes = text_yt_formatted_strings[1].text

    # Append information about each individual video to compiled list
    innerlist = [pull_date, upload_date, channel_name, video_title, video_views]

    # Append inner list to outer list
    out.append(innerlist)


" Apply Function: Scrape Data for Each Video Link in List "
for video in video_links:
    youtube_info(video)


" Create DataFrame from List of Scraped Video Information "
df_columns = ['Date', 'Upload Date', 'Channel', 'Video', 'Views']
scraped_data = pd.DataFrame(out, columns=df_columns)


" Clean new DataFrame "
scraped_data["Date"] = pd.to_datetime(scraped_data["Date"])
scraped_data["Upload Date"] = scraped_data["Upload Date"]\
    .str.replace(".*(Premiered)", "", regex=True)\
    .str.lstrip()
scraped_data["Upload Date"] = pd.to_datetime(scraped_data["Upload Date"])
scraped_data['Channel'] = scraped_data['Channel'].str.replace("BRADPAISLEY", "Brad Paisley")
scraped_data['Video'] = scraped_data['Video']\
    .str.replace(".*(- )", "", regex=True)\
    .str.replace(".*(– )", "", regex=True)\
    .str.replace("(\().*", "", regex=True)\
    .str.replace("(ft.).*", "", regex=True)\
    .str.replace('"', "")\
    .str.rstrip()


" Import Previously Scraped Data "
existing_data = pd.read_csv("Projects/Scrape-Youtube-Views/Output/YouTube Views.csv", parse_dates=[0])


" Append New Data to Existing Data "
df_export = pd.concat([existing_data, scraped_data]).reset_index(drop=True)
# df_export["Date"] = pd.to_datetime(df_export["Date"])
# df_export["Upload Date"] = pd.to_datetime(df_export["Upload Date"])


" Export to CSV "
df_export.to_csv("Projects/Scrape-Youtube-Views/Output/YouTube Views.csv", index=False)
# Print as table
print(tabulate(df_export, headers='keys', tablefmt='plain', showindex=False))







""" Analysis - Views Since Day of Release """
# Show number of views based on days from upload date

day_of_release = df_export

# Calculate days since upload date
day_of_release['Days Since Release'] = pd.to_datetime(day_of_release['Date']) - pd.to_datetime(day_of_release['Upload Date'])

# Convert timedelta to integer
day_of_release['Days Since Release'] = day_of_release['Days Since Release'].dt.days.astype('int16')
day_of_release.dtypes

# Sort
day_of_release = day_of_release.sort_values(by=['Video', 'Date']).reset_index(drop=True)
day_of_release.to_csv("Projects/Scrape-Youtube-Views/test.csv", index=False)

# Filter
day_of_release_filter = day_of_release.loc[(day_of_release['Upload Date'] >= min(day_of_release['Date']))]

# TODO: Create small multiples graph showing views since release date with all video lines greyed out in each subplot .




""" Analysis - Views per Month """

df_analysis = df_export

# Extract year and month and create new datetime
df_analysis["Year"] = df_analysis["Date"].dt.year
df_analysis["Month"] = df_analysis["Date"].dt.month
df_analysis["Month_Year_Str"] = df_analysis["Year"].astype(str) + df_analysis["Month"].astype(str).str.zfill(2) + str(
    1).zfill(2)
df_analysis["Date"] = pd.to_datetime(df_analysis['Month_Year_Str'], format='%Y-%m-%d')
df_analysis.drop(columns=['Month_Year_Str'], inplace=True)

# Sort by Video and Date
df_analysis_sorted = df_analysis.sort_values(by=['Video', 'Date'])

# Calculate daily gains in views by video
df_analysis_sorted['Monthly_Views'] = df_analysis_sorted.groupby('Video')['Views'].diff()

# Calculate new views for each video by month
df_monthly_views = df_analysis_sorted.groupby(['Date', 'Video'])["Monthly_Views"].sum().reset_index()
df_monthly_views["Monthly_Views"] = df_monthly_views["Monthly_Views"].astype(int)  # Convert float to integer

# Sort by video with highest view gains
# df_monthly_views = df_monthly_views.sort_values(['Monthly_Views'], ascending=False)
df_monthly_views = df_monthly_views.sort_values(['Date', 'Monthly_Views'], ascending=(True, False))

# Format variables
# df_monthly_views["Monthly_Views"] = (df_monthly_views.Views_Delta.apply(lambda x: "{:,}".format(x)))
# df_monthly_views["Monthly_Views"] = pd.to_datetime(df_monthly_views["Monthly_Views"], format='%Y-%m-%d').dt.date

# Print as table
df_monthly_views_final = df_monthly_views[['Date', 'Video', 'Monthly_Views']]
print(tabulate(df_monthly_views_final, headers='keys', tablefmt='plain', showindex=False))


" Scatter Plot "
# Single plot with monthly view counts for each video via Matplotlib

# TODO: change to line plot

plt.style.use('seaborn-notebook')  # Set style
fig, ax = plt.subplots(figsize=(13, 8))  # Set figure

# Color
color_labels = df_monthly_views_final['Video'].unique()
rgb_values = sns.color_palette("plasma", len(color_labels))
color_map = dict(zip(color_labels, rgb_values))

# Plot
plt.scatter(x=df_monthly_views_final['Date'],
            y=df_monthly_views_final['Monthly_Views'],
            c=df_monthly_views_final['Video'].map(color_map),
            cmap='plasma',
            s=150,
            alpha=0.9,
            edgecolors='white',
            linewidth=1,
            label=df_monthly_views_final['Video'].astype('category').cat.codes)

# Titles and axis labels
plt.title('Monthly Youtube Views of Carrie Underwood Singles',
          fontweight='bold',
          fontsize=20)
plt.xlabel('Date',
           fontweight='bold',
           size=12)
plt.ylabel('Monthly Views',
           fontweight='bold',
           size=12)

# Format axes
ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))  # Format x axis date
ax.xaxis.set_major_locator(md.MonthLocator(range(1, 13)))  # Format frequency of x axis marks
ax.set_ylim([0, 3000000])  # Y axis range
plt.yticks(np.arange(0, 3000001, 500000))  # Set y axis tick range and intervals; +1 needed to show last tick
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))  # Format y
# axis thousands

plt.tight_layout()  # Fit graph to window
plt.legend()
# mplcursors.cursor(hover=True)
plt.show()



""" Analysis - Total Views Over Time """
# Small Multiples
# Total view counts for all videos over time

# TODO: fix plot, likely due to no longer formatting views variable as string. Use code from plot above to format y
#  axis values.

# Set style
plt.style.use('seaborn')

fig, axes = plt.subplots(nrows=7, ncols=6, sharex='all', sharey='all', figsize=(13, 8))  # Set figure size
axes_list = [item for sublist in axes for item in sublist]  # List comprehension

ordered_videos = df_monthly_views_final["Video"].head(len(video_links))  # Set order of graphs
grouped = df_export.groupby("Video")

first_date = df_export['Date'].min()  # First date for x axis
last_date = df_export['Date'].max()  # Last date for x axis

for video in ordered_videos:
    selection = grouped.get_group(video)

    ax = axes_list.pop(0)
    selection.plot(x='Date', y='Views', label=video, ax=ax, legend=False, clip_on=False)
    ax.set_title(video)  # Set subplot titles

    " Ticks "
    ax.tick_params(
        which='both',  # Applies to both major and minor ticks
        # bottom=False,  # Turn off tick marks on x-axis
        # left=False,  # Turn off tick marks on y-axis
        right=False,  # Turn off tick marks on y2-axis
        top=False,  # Turn off tick marks on top of charts
        )

    " Format x-axis "
    ax.set_xlim((first_date, last_date))
    ax.set_xticks((first_date, last_date))
    ax.xaxis.label.set_visible(False)  # Turn off axis label
    ax.minorticks_off()  # Turn off minor ticks
    ax.xaxis.set_major_formatter(DateFormatter("%d\n%b\n%Y"))  # Format day, month, year format and line break
    for label in ax.get_xticklabels():
        label.set_ha('center')  # Set tick label location
        label.set_rotation(0)  # Set tick label rotation

    " Format y-axis "
    ax.set_ylim((0, 200000000))  # Set range min and max
    plt.yticks(np.arange(0, 200000000, 50000000))  # Set tick range and intervals
    # ax.axes.yaxis.set_visible(False)  # Turn off tick labels

    def millions(x, pos):  # Set tick labels to millions
        return '%1.0fM' % (x * 1e-6)
    formatter = FuncFormatter(millions)
    ax.yaxis.set_major_formatter(formatter)

    " Subplot frame lines "
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    " Add annotation on each graph of the latest view count "
    max_date = selection["Date"].max()
    views = float(selection[df_export["Date"] == max_date]["Views"])
    ax.scatter(x=[max_date], y=[views], s=70, clip_on=False, linewidth=0)
    ax.annotate(str(int(views/1000000)) + "M", xy=[max_date, views], xytext=[7, -2], textcoords='offset points')

for ax in axes_list:
    ax.remove()

fig.suptitle('YouTube Views', size=15, ha='center')  # Global figure title

plt.tight_layout()
plt.show()





"""
Sources
https://betterprogramming.pub/the-only-step-by-step-guide-youll-need-to-build-a-web-scraper-with-python-e79066bd895a
https://github.com/angelicadietzel/data-projects/blob/master/single-page-web-scraper/imdb_scraper.py
https://www.thepythoncode.com/article/get-youtube-data-python
https://stackoverflow.com/questions/44502482/how-to-create-and-fill-a-list-of-lists-in-a-for-loop
"""