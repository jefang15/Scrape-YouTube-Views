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


out = []


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


for video in video_links:
    youtube_info(video)


" Create DataFrame from List of Scraped Video Information "
df_columns = ['Date', 'Upload Date', 'Channel', 'Video', 'Views']
scraped_data = pd.DataFrame(out, columns=df_columns)


" Clean New DataFrame "
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
scraped_data

" Import Previously Scraped Data "
existing_data = pd.read_csv("Projects/Scrape-YouTube-Views/Output/YouTube Views.csv", parse_dates=[0, 1])


" Append New Data to Existing Data "
df_export = pd.concat([existing_data, scraped_data]).reset_index(drop=True)
# df_export["Date"] = pd.to_datetime(df_export["Date"])
# df_export["Upload Date"] = pd.to_datetime(df_export["Upload Date"])


" Export to CSV "
df_export.to_csv("Projects/Scrape-YouTube-Views/Output/YouTube Views.csv", index=False)

# Print as table
print(tabulate(df_export, headers='keys', tablefmt='plain', showindex=False))










" Analysis - Views Since Day of Release "

premiere = pd.read_csv("Projects/Scrape-YouTube-Views/Output/YouTube Views.csv", parse_dates=[0, 1])
premiere['Days Since Release'] = (pd.to_datetime(premiere['Date']) - pd.to_datetime(premiere['Upload Date'])).dt.days.astype('int16')
premiere = premiere.sort_values(by=['Video', 'Date']).reset_index(drop=True)

# Color
premiere_sort = premiere.sort_values('Upload Date')
unique_videos = premiere_sort['Video'].unique()
color_values = sns.color_palette('inferno', len(unique_videos))
color_map = pd.DataFrame(zip(unique_videos, color_values), columns=['Video', 'Color'])
premiere_plot = premiere.merge(color_map, how='left', left_on=['Video'], right_on=['Video'])



" Plot - Views Since Day of Release "

plt.style.use('default')  # Set style
fig, ax = plt.subplots(figsize=(14, 8))

grouped = premiere_plot.groupby('Video')

for video, grp in premiere_plot.groupby(['Video']):
    selection = grouped.get_group(video)

    ax = grp.plot(
        ax=ax,
        kind='line',
        x='Days Since Release',
        y='Views',
        color=selection['Color'],
        label=video,
        legend=False)

    selection = grouped.get_group(video)
    max_views = selection['Views'].max()
    color = selection['Color'].max()
    max_days = selection['Days Since Release'].max()  # Max number of days since release for each video
    views = float(selection[premiere_plot['Days Since Release'] == max_days]['Views'])  # Max views for each video
    ax.scatter(x=[max_days], y=[views], s=70, color=[color], clip_on=False, linewidth=0)

    ax.annotate(
        video + ', ' + str(int(views/1000000)) + 'MM',
        xy=[max_days, views],
        xytext=[7, -2],
        textcoords='offset points')

ax.tick_params(
        which='both',  # Applies to both major and minor ticks
        # bottom=False,  # Turn off tick marks on x-axis
        left=False,  # Turn off tick marks on y-axis
        right=False,  # Turn off tick marks on y2-axis
        top=False,  # Turn off tick marks on top of charts
        )

# Format x axis
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))  # Format y axis thousands
plt.xticks(np.arange(0, premiere_plot['Days Since Release'].max().round(-3)+501, 500))
ax.set_xlim([0, premiere_plot['Days Since Release'].max().round(-3)+501])

# Format y axis
# ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))  # Format y axis thousands
# plt.yticks(np.arange(0, 150000000 + 1, 25000000))
# ax.set_ylim([0, 150000000 + 1])
ax.axes.yaxis.set_ticklabels([])

ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

# Titles
plt.title('Music Video Views Since Premiere',
          fontweight='bold',
          fontsize=20)
plt.xlabel('Days Since Release',
           # fontweight='bold',
           size=12)
plt.ylabel('Views',
           # fontweight='bold',
           size=12)

plt.tight_layout()
plt.show()

# plt.savefig('Projects/Scrape-YouTube-Views/Output - Graphs/Views Since Premiere.png')



" Plot - Views Since Day of Release - Subset "

premiere_subset = pd.read_csv("Projects/Scrape-YouTube-Views/Output/YouTube Views.csv", parse_dates=[0, 1])
premiere_subset['Days Since Release'] = (pd.to_datetime(premiere_subset['Date']) - pd.to_datetime(premiere_subset['Upload Date'])).dt.days.astype('int16')
premiere_subset = premiere_subset.sort_values(by=['Video', 'Date']).reset_index(drop=True)
premiere_subset = premiere_subset.loc[(premiere_subset['Upload Date'] >= min(premiere_subset['Date']))]

# Color
premiere_subset_sort = premiere_subset.sort_values('Upload Date')
unique_videos_subset = premiere_subset_sort['Video'].unique()
color_values_subset = sns.color_palette('inferno', len(unique_videos_subset))
color_map_subset = pd.DataFrame(zip(unique_videos_subset, color_values_subset), columns=['Video', 'Color'])
premiere_subset_plot = premiere_subset.merge(color_map_subset, how='left', left_on=['Video'], right_on=['Video'])



plt.style.use('default')  # Set style
fig, ax = plt.subplots(figsize=(14, 8))

grouped = premiere_subset_plot.groupby('Video')

for video, grp in premiere_subset_plot.groupby(['Video']):
    selection = grouped.get_group(video)

    ax = grp.plot(
        ax=ax,
        kind='line',
        x='Days Since Release',
        y='Views',
        color=selection['Color'],
        label=video,
        legend=False)

    max_views = selection['Views'].max()
    color = selection['Color'].max()
    max_days = selection['Days Since Release'].max()  # Max number of days since release for each video
    views = float(selection[premiere_subset_plot['Days Since Release'] == max_days]['Views'])  # Max views for each video
    ax.scatter(x=[max_days], y=[views], s=70, color=[color], clip_on=False, linewidth=0)

    ax.annotate(
        video + ', ' + str(int(views/1000000)) + 'MM',
        xy=[max_days, views],
        xytext=[7, -2],
        textcoords='offset points')

ax.tick_params(
        which='both',  # Applies to both major and minor ticks
        # bottom=False,  # Turn off tick marks on x-axis
        left=False,  # Turn off tick marks on y-axis
        right=False,  # Turn off tick marks on y2-axis
        top=False,  # Turn off tick marks on top of charts
        )

# Format x axis
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))  # Format y axis thousands
plt.xticks(np.arange(0, 360 + .1, 30))
ax.set_xlim([0, 360 + .1])

# Format y axis
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))  # Format y axis thousands
plt.yticks(np.arange(0, 20000000 + 1, 2500000))
ax.set_ylim([0, 20000000 + 1])
# ax.axes.yaxis.set_ticklabels([])

ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

# Titles
plt.title('Music Video Views Since Premiere',
          fontweight='bold',
          fontsize=20)
plt.xlabel('Days Since Release',
           # fontweight='bold',
           size=12)
plt.ylabel('Views',
           # fontweight='bold',
           size=12)

plt.tight_layout()
plt.show()

# plt.savefig('Projects/Scrape-YouTube-Views/Output - Graphs/Views Since Premiere - Subset.png')





" Analysis - Views per Month "
# Shows which videos consistently garner the most views and any deviations from average

monthly = pd.read_csv("Projects/Scrape-YouTube-Views/Output/YouTube Views.csv", parse_dates=[0, 1])

monthly["Year"] = monthly["Date"].dt.year
monthly["Month"] = monthly["Date"].dt.month
monthly["Month_Year_Str"] = monthly["Year"].astype(str) + monthly["Month"].astype(str).str.zfill(2) + str(1).zfill(2)
monthly["Date"] = pd.to_datetime(monthly['Month_Year_Str'], format='%Y-%m-%d')

monthly.drop(columns=['Month_Year_Str'], inplace=True)
monthly = monthly.sort_values(by=['Video', 'Date'])
monthly['Monthly Views'] = monthly.groupby('Video')['Views'].diff()

monthly_views = monthly.groupby(['Date', 'Video'])["Monthly Views"].sum().reset_index()
monthly_views["Monthly Views"] = monthly_views["Monthly Views"].astype(int)
monthly_views = monthly_views.sort_values(['Date', 'Monthly Views'], ascending=(True, False))
monthly_views = monthly_views.loc[(monthly_views['Date'] == monthly_views['Date'].max())]


" Plot - Views per Month "
# Color
monthly_views_sort = monthly_views.sort_values('Monthly Views', ascending=False)
unique_videos_monthly = monthly_views_sort['Video'].unique()
color_values_monthly = sns.color_palette('inferno', len(unique_videos_monthly))
color_map_monthly = pd.DataFrame(zip(unique_videos_monthly, color_values_monthly), columns=['Video', 'Color'])
monthly_views_plot = monthly_views.merge(color_map_monthly, how='left', left_on=['Video'], right_on=['Video']).sort_values(
    'Monthly Views', ascending=True)



# TODO: create scatter plot with each video along x axis and multiple vertical dots for views each month and add months as labels
plt.style.use('default')  # Set style
fig, ax = plt.subplots(figsize=(14, 8))  # Set figure

ordered_videos = monthly_views_plot['Video']
grouped = monthly_views_plot.groupby('Video')

for video in ordered_videos:
    selection = grouped.get_group(video)

    selection.plot(
        ax=ax,
        kind='scatter',
        x='Video',
        y='Monthly Views',
        s=125,
        edgecolors='white',
        linewidth=1,
        label=video,
        color=selection['Color'])

# Titles and axis labels
plt.title('Monthly YouTube Views \n March 2022',
          fontweight='bold',
          fontsize=20)
# plt.xlabel('Video',
#            fontweight='bold',
#            size=12)
plt.ylabel('Views',
           size=12)

# Format both axes
ax.tick_params(
        which='both',  # Applies to both major and minor ticks
        # bottom=False,  # Turn off tick marks on x-axis
        # left=False,  # Turn off tick marks on y-axis
        right=False,  # Turn off tick marks on y2-axis
        top=False,  # Turn off tick marks on top of charts
        )

# Plot borders
ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

# Format x-axis
for label in ax.get_xticklabels():  # Set tick label location and angle
    label.set_ha('right')
    label.set_rotation(45)
# ax.axes.xaxis.set_ticklabels([])  # Turn off tick labels
ax.xaxis.label.set_visible(False)  # Turn off x-axis label

# Format y-axis
ax.set_ylim([0, 2500000])
plt.yticks(np.arange(0, 2500001, 500000))
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# plt.legend(loc='upper center', ncol=4)  # Format legend
ax.legend().set_visible(False)  # Hide legend
plt.tight_layout()
plt.show()

# plt.savefig('Projects/Scrape-YouTube-Views/Output - Graphs/Monthly Views - March 2022.png')





" Analysis - Total Views Over Time "

plt.style.use('default')

fig, axes = plt.subplots(nrows=7, ncols=6, sharex='all', sharey='all', figsize=(14, 8))  # Set figure size
axes_list = [item for sublist in axes for item in sublist]  # List comprehension

ordered_videos = scraped_data['Video'].head(len(video_links))  # Set order of graphs
grouped = df_export.groupby('Video')

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
    max_date = selection['Date'].max()
    views = float(selection[df_export['Date'] == max_date]['Views'])
    ax.scatter(x=[max_date], y=[views], s=70, clip_on=False, linewidth=0)
    ax.annotate(str(int(views/1000000)) + 'M', xy=[max_date, views], xytext=[7, -2], textcoords='offset points')

for ax in axes_list:
    ax.remove()

fig.suptitle('YouTube Views', size=15, ha='center')  # Global figure title

# plt.tight_layout()
plt.show()

