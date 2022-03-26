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

# Data Preprocessing
final_df.drop(['Image-URL-S', 'Image-URL-M','Image-URL-L'], axis=1, inplace=True)
final_df.dropna(inplace=True)

final_df['User-ID'] = final_df['User-ID'].astype('int')
final_df=final_df[final_df["Book-Rating"]>0]

final_df.columns
final_df['User-ID'].nunique() # 40543 unique Readers
final_df['ISBN'].nunique() # 119945 unique ISBN number (books)
final_df['Book-Title'].nunique() # 109210 unique Books.
final_df['Book-Author'].nunique() # 50883 unique Author.
final_df['Publisher'].nunique() # 9749 unique Publisher.
final_df['Book-Rating'].nunique() # 10 unique Rating (1-10).

final_df.groupby('User-ID')['Book-Title'].agg('count').sort_values(ascending=False)
final_df.groupby('Book-Author')['ISBN'].agg('count').sort_values(ascending=False)
final_df.groupby('ISBN').agg({'Book-Title':'count'}).sort_values(by='Book-Title',ascending=False).head(100)
final_df.groupby('User-ID')['Book-Rating'].agg('count').sort_values(ascending=False)


# We can calculate the ratings of the books and eliminate the rare books by determining a threshold which is 85 in this case by removing the main dataframe.
# Finally, we have common books which are rated more than 85.
rating_book = pd.DataFrame(final_df['Book-Title'].value_counts())
final_df['Book-Title'].value_counts().mean() # 2.4688

rare_books = rating_book[rating_book['Book-Title'] <= 85].index

common_books = final_df[~final_df["Book-Title"].isin(rare_books)]
common_books['Book-Title'].value_counts().mean() # 139.6589

# Creating User-Item Matrix Dataframe
user_book_matrix = common_books.pivot_table(index=['User-ID'], columns=['Book-Title'], values='Book-Rating')
user_book_matrix.shape


# 1 - Item Based Collaborative Filtering Recommendation Engine
# Item based collaborative filtering engine provide us a relations between items that is books in this case, by looking their correlations each other.
# If the correlations are quite similar, that means these books are similar according to maybe their properties, genres. Hence, we can recommend them to user.
book_name = "Bridget Jones's Diary"

def book_to_ISBN(name):
    """
    This function gives us the ISBN number of the book for the name of the book which you want to learn ISBN number.
    :param name: the name of the book
    :return: ISBN; the books are identified by their respective ISBN
    """
    ISBN = final_df[final_df['Book-Title'] == name][['ISBN']].iloc[0,0]
    return ISBN

ISBN_no = book_to_ISBN(book_name)

book_df = user_book_matrix[book_name]
book_df.sort_values(ascending=False)

similar_to_book_name = user_book_matrix.corrwith(book_df)

corr_book_name = pd.DataFrame(similar_to_book_name, columns=['Correlation'])
corr_book_name.dropna(inplace=True)
corr_book_name.sort_values(by='Correlation',ascending=False).head(50)


