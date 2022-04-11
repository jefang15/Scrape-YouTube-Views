


from requests_html import HTMLSession
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker
from tabulate import tabulate
import seaborn as sns



" Videos to Track "
# Add new video links here as desired

cu_music_videos_list = [

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

    'https://www.youtube.com/watch?v=GzT4p-OaJ5c',  # The Fighter

    # Cry Pretty
    'https://www.youtube.com/watch?v=KUUjtUP2CrE',  # Cry Pretty
    'https://www.youtube.com/watch?v=-Py8OWAMkns',  # Love Wins
    'https://www.youtube.com/watch?v=pc4zBnvLt3g',  # Southbound
    'https://www.youtube.com/watch?v=vfkq-Ov7WSw',  # Drinking Alone

    # My Gift
    'https://www.youtube.com/watch?v=4KiFSKLTj0U',  # Hallelujah

    'https://www.youtube.com/watch?v=Zc3cxj5pDIs',  # If I Didn't Love You

    ]


cu_other_videos_list = [

    'https://www.youtube.com/watch?v=s9gAXwYZtfk',  # Forever Country

    'https://www.youtube.com/watch?v=HgknAaKNaMM',  # The Champion

    'https://www.youtube.com/watch?v=o6teH-xJn5o',  # Tears of Gold

    # My Savior
    'https://www.youtube.com/watch?v=Yf6C0L_7-CA',  # How Great Thou Art
    'https://www.youtube.com/watch?v=L6XMSPJzfEU',  # Nothing But The Blood Of Jesus

    'https://www.youtube.com/watch?v=d4mh4jq_MYU',  # I Wanna Remember

    'https://www.youtube.com/watch?v=kt7VSlX1HgY',  # Only Us

    # Denim & Rhinestones
    'https://www.youtube.com/watch?v=4Y7flhznvnE',  # Ghost Story (Official Lyric Video)
    'https://www.youtube.com/watch?v=TYJVEYbwR4o'  # Denim & Rhinestones (Official Lyric Video)

    ]



def scrapeYoutubeViews(list_of_videos):
    out = []
    for video in list_of_videos:
        session = HTMLSession()  # init an HTML Session
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"}
        video_url = video  # YouTube video url
        response = session.get(video_url, headers=headers)  # Get the URL contents
        response.html.render(sleep=20, timeout=20)  # Set timeout
        soup = bs(response.html.html, features="html.parser")  # Make URL contents easy to read
        # print(soup.prettify())  # Open html
        # open("video.html", "w", encoding='utf8').write(response.html.html)  # Write all HTML code into a file

        # Date scraped
        pull_date = date.today()

        # Upload date
        upload_date = soup.find("div", {"id": "info-strings"}).find("yt-formatted-string").text

        # Channel name
        channel_name = soup.find("yt-formatted-string", {"class": "ytd-channel-name"}).find("a").text

        # Video title
        video_title = soup.find("div", {"id": "container", "class": "style-scope ytd-video-primary-info-renderer"}).find(
            "h1").text
        print(video_title)

        # Length of video
        # video_length = soup.find("span", {"class": "ytp-time-duration"}).text

        # Views
        video_views = int(''.join([c for c in soup.find("span", attrs={"class": "view-count"}).text if c.isdigit()]))

        # Likes & Dislikes
        # text_yt_formatted_strings = soup.find_all("yt-formatted-string",
        #                                           {"id": "text", "class": "ytd-toggle-button-renderer"})
        # video_likes = text_yt_formatted_strings[0].text
        # video_dislikes = text_yt_formatted_strings[1].text

        # Append information about each individual video to compiled list
        inner_list = [pull_date, upload_date, channel_name, video_title, video_views]

        # Append inner list to outer list
        out.append(inner_list)

    " Create DataFrame from List of Scraped Video Information "
    df_headers = ['Date', 'Upload Date', 'Channel', 'Video', 'Views']
    df = pd.DataFrame(out, columns=df_headers)

    " Clean New DataFrame "
    df['Date'] = pd.to_datetime(df['Date'])
    df['Upload Date'] = df['Upload Date'] \
        .str.replace(".*(Premiered)", "", regex=True) \
        .str.lstrip()
    df['Upload Date'] = pd.to_datetime(df['Upload Date'])
    df['Channel'] = df['Channel'].str.replace('BRADPAISLEY', 'Brad Paisley')
    df['Video'] = df['Video'] \
        .str.replace(".*(- )", "", regex=True) \
        .str.replace(".*(– )", "", regex=True) \
        .str.replace("(\().*", "", regex=True) \
        .str.replace("(ft.).*", "", regex=True) \
        .str.replace('"', "") \
        .str.rstrip()

    return df


" Apply Function to Videos "
scraped_cu_music_videos = scrapeYoutubeViews(cu_music_videos_list)
scraped_cu_other_videos = scrapeYoutubeViews(cu_other_videos_list)


# Move scraped date back one day if run past midnight
# from datetime import timedelta
# scraped_cu_music_videos['Date'] = scraped_cu_music_videos['Date'] - timedelta(days=1)
# scraped_cu_other_videos['Date'] = scraped_cu_other_videos['Date'] - timedelta(days=1)


" Import Previously Scraped Data "
existing_cu_music_videos = pd.read_csv('Projects/Scrape-YouTube-Views/Output/CU Music Video Views.csv', parse_dates=[0, 1])
existing_cu_other_videos = pd.read_csv('Projects/Scrape-YouTube-Views/Output/CU Other Video Views.csv', parse_dates=[0, 1])


" Concatenate New Data to Existing Data "
cu_music_videos_export = pd.concat([existing_cu_music_videos, scraped_cu_music_videos]).reset_index(drop=True)
cu_other_videos_export = pd.concat([existing_cu_other_videos, scraped_cu_other_videos]).reset_index(drop=True)
# df_export["Date"] = pd.to_datetime(df_export["Date"])
# df_export["Upload Date"] = pd.to_datetime(df_export["Upload Date"])


" Export to CSV "
cu_music_videos_export.to_csv('Projects/Scrape-YouTube-Views/Output/CU Music Video Views.csv', index=False)
cu_other_videos_export.to_csv('Projects/Scrape-YouTube-Views/Output/CU Other Video Views.csv', index=False)


# Print as table
# print(tabulate(cu_music_videos_export, headers='keys', tablefmt='plain', showindex=False))
# print(tabulate(cu_other_videos_export, headers='keys', tablefmt='plain', showindex=False))










" Analysis - Views Since Day of Release "


def clean_views_since_premiere(df):
    df['Days Since Release'] = (pd.to_datetime(df['Date']) - pd.to_datetime(df['Upload Date'])).dt.days.astype('int16')
    df = df.sort_values(by=['Video', 'Date']).reset_index(drop=True)

    # Color
    df_sort = df.sort_values('Upload Date')
    unique_videos = df_sort['Video'].unique()
    color_values = sns.color_palette('inferno', len(unique_videos))
    color_map = pd.DataFrame(zip(unique_videos, color_values), columns=['Video', 'Color'])
    df_plot = df.merge(color_map, how='left', left_on=['Video'], right_on=['Video'])
    return df_plot


df_premiere = pd.read_csv('Projects/Scrape-YouTube-Views/Output/CU Music Video Views.csv', parse_dates=[0, 1])
plot_premiere = clean_views_since_premiere(df_premiere)


" Plot - Views Since Day of Release "
plt.style.use('default')  # Set style
fig, ax = plt.subplots(figsize=(14, 8))

grouped = plot_premiere.groupby('Video')

for video, grp in plot_premiere.groupby(['Video']):
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
    views = float(selection[plot_premiere['Days Since Release'] == max_days]['Views'])  # Max views for each video
    ax.scatter(x=[max_days], y=[views], s=70, color=[color], clip_on=False, linewidth=0)

    ax.annotate(
        video + ' ' + str(int(views/1000000)) + 'M',
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
plt.xticks(np.arange(0, plot_premiere['Days Since Release'].max().round(-3)+501, 500))
ax.set_xlim([0, plot_premiere['Days Since Release'].max().round(-3)+501])

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



" Analysis - Views Since Day of Release Subset "


def clean_views_since_premiere(df):
    df['Days Since Release'] = (pd.to_datetime(df['Date']) - pd.to_datetime(df['Upload Date'])).dt.days.astype('int16')
    df = df.sort_values(by=['Video', 'Date']).reset_index(drop=True)
    df = df.loc[(df['Upload Date'] >= min(df['Date']))]

    # Color
    df_sort = df.sort_values('Upload Date')
    unique_videos = df_sort['Video'].unique()
    color_values = sns.color_palette('inferno', len(unique_videos))
    color_map = pd.DataFrame(zip(unique_videos, color_values), columns=['Video', 'Color'])
    df_plot = df.merge(color_map, how='left', left_on=['Video'], right_on=['Video'])
    return df_plot

# CU Music Videos
df_premiere_subset = pd.read_csv('Projects/Scrape-YouTube-Views/Output/CU Music Video Views.csv', parse_dates=[0, 1])
# CU Other Videos
# df_premiere_subset = pd.read_csv('Projects/Scrape-YouTube-Views/Output/CU Other Video Views.csv', parse_dates=[0, 1])

plot_premiere_subset = clean_views_since_premiere(df_premiere_subset)


" Plot - Views Since Day of Release Subset "
plt.style.use('default')  # Set style
fig, ax = plt.subplots(figsize=(14, 8))

grouped = plot_premiere_subset.groupby('Video')

for video, grp in plot_premiere_subset.groupby(['Video']):
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
    views = float(selection[plot_premiere_subset['Days Since Release'] == max_days]['Views'])  # Max views for each video
    ax.scatter(x=[max_days], y=[views], s=70, color=[color], clip_on=False, linewidth=0)

    ax.annotate(
        video + ' ' + str(int(views/1000000)) + 'M',
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

monthly = pd.read_csv('Projects/Scrape-YouTube-Views/Output/CU Music Video Views.csv', parse_dates=[0, 1])

monthly['Year'] = monthly['Date'].dt.year
monthly['Month'] = monthly['Date'].dt.month
monthly['Month_Year_Str'] = monthly['Year'].astype(str) + monthly['Month'].astype(str).str.zfill(2) + str(1).zfill(2)
monthly['Date'] = pd.to_datetime(monthly['Month_Year_Str'], format='%Y-%m-%d')

monthly.drop(columns=['Month_Year_Str'], inplace=True)
monthly = monthly.sort_values(by=['Video', 'Date'])
monthly['Monthly Views'] = monthly.groupby('Video')['Views'].diff()

monthly_views = monthly.groupby(['Date', 'Video'])['Monthly Views'].sum().reset_index()
monthly_views['Monthly Views'] = monthly_views['Monthly Views'].astype(int)
monthly_views = monthly_views.sort_values(['Date', 'Monthly Views'], ascending=(True, False))
latest_month = monthly_views.loc[(monthly_views['Date'] == monthly_views['Date'].max())]


" Plot - Views per Month for Latest Month "
# Color
latest_month_sort = latest_month.sort_values('Monthly Views', ascending=False)
unique_videos_latest_month = latest_month_sort['Video'].unique()
color_values_latest_month = sns.color_palette('inferno', len(unique_videos_latest_month))
color_map_latest_month = pd.DataFrame(zip(unique_videos_latest_month, color_values_latest_month), columns=['Video', 'Color'])
latest_month_plot = latest_month.merge(color_map_latest_month, how='left', left_on=['Video'], right_on=['Video']).sort_values(
    'Monthly Views', ascending=True)



plt.style.use('default')  # Set style
fig, ax = plt.subplots(figsize=(14, 8))  # Set figure

ordered_videos = latest_month_plot['Video']
grouped = latest_month_plot.groupby('Video')

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
plt.title('Monthly YouTube Views \n April 2022',  # TODO: Change date here as necessary
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

# plt.savefig('Projects/Scrape-YouTube-Views/Output - Graphs/Monthly Views (March 2022).png')





" Plot - Views per Month for Each Month "
# Color
monthly_views_sort = monthly_views.sort_values(['Date', 'Monthly Views'], ascending=[False, True]).copy()
unique_videos_monthly = monthly_views_sort['Video'].unique()
color_values_monthly = sns.color_palette('inferno', len(unique_videos_monthly))
color_map_monthly = pd.DataFrame(zip(unique_videos_latest_month, color_values_monthly), columns=['Video', 'Color'])
monthly_views_plot = monthly_views.merge(color_map_monthly, how='left', left_on=['Video'], right_on=['Video']).sort_values(
    'Monthly Views', ascending=True)



# TODO: Add latest month as labels
plt.style.use('default')  # Set style
fig, ax = plt.subplots(figsize=(14, 8))  # Set figure

ordered_videos = monthly_views_sort['Video']
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
plt.title('Monthly YouTube Views',
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
ax.set_ylim([0, 3000000])
plt.yticks(np.arange(0, 3000001, 500000))
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# plt.legend(loc='upper center', ncol=4)  # Format legend
ax.legend().set_visible(False)  # Hide legend
plt.tight_layout()
plt.show()

# plt.savefig('Projects/Scrape-YouTube-Views/Output - Graphs/Monthly Views.png')





" Plot - Total Views (Scatter) "


def clean_total_views(df):
    df = df.loc[(df['Date'] == df['Date'].max())]

    # Color
    df_reverse = df.iloc[::-1]
    unique_videos = df_reverse['Video'].unique()
    color_values = sns.color_palette('inferno', len(unique_videos))
    color_map = pd.DataFrame(zip(unique_videos, color_values), columns=['Video', 'Color'])
    df_plot = df.merge(color_map, how='left', left_on=['Video'], right_on=['Video'])
    return df_plot


total = pd.read_csv('Projects/Scrape-YouTube-Views/Output/CU Music Video Views.csv', parse_dates=[0, 1])
total_plot = clean_total_views(total)


plt.style.use('default')  # Set style
fig, ax = plt.subplots(figsize=(14, 8))  # Set figure

ordered_videos = total_plot['Video']
grouped = total_plot.groupby('Video')

for video in ordered_videos:
    selection = grouped.get_group(video)

    selection.plot(
        ax=ax,
        kind='scatter',
        x='Video',
        y='Views',
        s=125,
        edgecolors='white',
        linewidth=1,
        label=video,
        color=selection['Color'])

# Titles and axis labels
plt.title('Total YouTube Views',
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
ax.set_ylim([0, 150000000])
plt.yticks(np.arange(0, 150000001, 25000000))
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# plt.legend(loc='upper center', ncol=4)  # Format legend
ax.legend().set_visible(False)  # Hide legend
plt.tight_layout()
plt.show()

# plt.savefig('Projects/Scrape-YouTube-Views/Output - Graphs/Total Views.png')



" Plot - Total Views (Small Multiples) "
plt.style.use('default')

fig, axes = plt.subplots(nrows=7, ncols=6, sharex='all', sharey='all', figsize=(14, 8))  # Set figure size
axes_list = [item for sublist in axes for item in sublist]  # List comprehension

ordered_videos = scraped_cu_music_videos['Video'].head(len(cu_music_videos_list))  # Set order of graphs
grouped = cu_music_videos_export.groupby('Video')

first_date = cu_music_videos_export['Date'].min()  # First date for x axis
last_date = cu_music_videos_export['Date'].max()  # Last date for x axis

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
    views = float(selection[cu_music_videos_export['Date'] == max_date]['Views'])
    ax.scatter(x=[max_date], y=[views], s=70, clip_on=False, linewidth=0)
    ax.annotate(str(int(views/1000000)) + 'M', xy=[max_date, views], xytext=[7, -2], textcoords='offset points')

for ax in axes_list:
    ax.remove()

fig.suptitle('YouTube Views', size=15, ha='center')  # Global figure title

plt.tight_layout()
plt.show()

# plt.savefig('Projects/Scrape-YouTube-Views/Output - Graphs/Total Views Small Multiples.png')
