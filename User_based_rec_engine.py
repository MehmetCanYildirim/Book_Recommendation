# 2 - User Based Collaborative Filtering Recommendation Engine

# Importing Libraries
import pandas as pd

# Adjusting pandas dataframe settings
pd.set_option('display.max_columns', 500)

# Reading Dataframes
books = pd.read_csv("Books.csv", low_memory='False')
ratings = pd.read_csv('Ratings.csv', low_memory='False')
users = pd.read_csv("Users.csv", low_memory='False')

# Merging Dataframes
temp_df = books.merge(ratings, how='left', on='ISBN')
final_df = temp_df.merge(users, how='left', on='User-ID')
final_df.shape

# Creating user book matrix.
def user_book_df(dataframe):
    dataframe.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
    dataframe.dropna(inplace=True)
    dataframe['User-ID'] = dataframe['User-ID'].astype('int')
    final_df = dataframe[dataframe["Book-Rating"] > 0]

    rating_book = pd.DataFrame(final_df['Book-Title'].value_counts())

    rare_books = rating_book[rating_book['Book-Title'] <= 85].index

    common_books = final_df[~final_df["Book-Title"].isin(rare_books)]
    user_book_matrix = common_books.pivot_table(index=['User-ID'], columns=['Book-Title'], values='Book-Rating')
    return user_book_matrix

user_book_matrix = user_book_df(final_df)

# Let's choose a user and recommend books to him/her.
# User Based Collaborative Filtering Engine provide us a recommendation system by using user similarities instead of item similarities.
# In that case, a score is calculated for users according to their ratings and correlations. This calculation can be weighted, that means,
# if you are thinking that the ratings are more important than readers correlations, you can give more weight to ratings. In this case, i will make it like this.
user = int('153662')

user_df = user_book_matrix[user_book_matrix.index == user]

user_read_books = user_df.columns[user_df.notna().any()].tolist()

user_book_matrix.loc[user_book_matrix.index == user, user_book_matrix.columns == "White Oleander : A Novel (Oprah's Book Club)"]

len(user_read_books) # 23

# Choosing users book
read_books = user_book_matrix[user_read_books]
read_books.head()

user_book_count_df = read_books.T.notnull().sum().reset_index()
user_book_count_df.columns = ['User-ID','Count']
user_book_count_df['Count'].describe()

# Choosing same readers with user
same_readers_with_user = user_book_count_df[user_book_count_df['Count'] > 5]['User-ID']

rec_df = pd.concat([read_books[read_books.index.isin(same_readers_with_user)],user_df[user_read_books]])

correlation_df = rec_df.T.corr().unstack().sort_values(ascending=False).drop_duplicates()
correlation_df = pd.DataFrame(correlation_df,columns=['correlation'])
correlation_df.index.names = ['User-ID-1','User-ID-2']
correlation_df = correlation_df.reset_index()
correlation_df

similar_users = correlation_df[(correlation_df['correlation'] >= 0.70) & (correlation_df['User-ID-1'] == user)][['User-ID-2','correlation']]
similar_users = similar_users.reset_index(drop=True)
similar_users.rename(columns={'User-ID-2':'User-ID',
                              'correlation':'Correlations'}, inplace=True)

# You can reach the correlations between our user which has User-ID (153662) and other readers.
#       User-ID    Correlation
# 0     102702     1.000000
# 1      22625     0.979796
# 2     238889     0.846332

# In this case our user has 153662 User-ID and you can reach similar readers with our user by using correlation each others.
# However this may not be good recommendation engine for the system since we did not calculate ratings. What i am trying to say,
# for example, a reader can be good correlate with our user, but it maybe voted low rating points. Therefore, we should take into account also ratings.
# Hence, we can recommend good rating books to readers.


# Weighted Score

similar_users_df = similar_users.merge(ratings, how='inner',on='User-ID')
similar_users_df = similar_users_df.merge(books,how='inner', on='ISBN')
similar_users_df = similar_users_df[similar_users_df['User-ID'] != user]

similar_users_df['Weighted Score'] = (0.4 * similar_users_df['Correlations'] + 0.6 * (similar_users_df['Book-Rating'] / 10)) / 2

# similar_users_df['Score'] = similar_users_df['Correlations'] * similar_users_df['Book-Rating']

similar_users_df.sort_values(by='Weighted Score',ascending=False)

similar_users_df.groupby('User-ID').agg({'Weighted Score':'mean'})
#         Weighted Score
# User-ID
# 22625          0.313127
# 102702         0.344213
# 238889         0.411321

# We can say that the 238889 User-ID has most recommended reader with our user on average.

# Finally, we can calculate the weighted score on books. Additionally, a threshold has to be determined for the score, in this case 8.5,
# so that the recommender system do not recommend to the users all of the books which have scores.
recommendation_df = similar_users_df.groupby('Book-Title').agg({'Weighted Score':'mean'}).reset_index()
recommendation_df_by_score = recommendation_df[recommendation_df["Weighted Score"] > 0.40].sort_values(by='Weighted Score',ascending=False)
recommendation_df_by_score = recommendation_df_by_score.merge(books, how='inner',on='Book-Title')

# Lastly, we can determine a list to will be recommended.
recommended_list = recommendation_df_by_score.sort_values(by='Weighted Score',ascending=False).drop_duplicates(subset='Book-Title')['Book-Title'].head(10).to_list()

