# Model Based (Matrix Factorization) Collaborative Filtering Recommendation Engine

# Importing Libraries
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate

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

final_df.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
final_df.dropna(inplace=True)
final_df['User-ID'] = final_df['User-ID'].astype('int')
final_df = final_df[final_df["Book-Rating"] > 0]

rating_book = pd.DataFrame(final_df['Book-Title'].value_counts())

book_id = ["0316666343", "0971880107", "0385504209", "0142001740", "0330332775"]
book_titles = ["The Lovely Bones: A Novel",
               "Wild Animus",
               "The Da Vinci Code",
               "The Secret Life of Bees",
               "Bridget Jones's Diary"]

sample_df = final_df[final_df['ISBN'].isin(book_id)]
sample_df



user_book_matrix = sample_df.pivot_table(index=['User-ID'],columns=['Book-Title'],values='Book-Rating')
user_book_matrix

# Model

sample_df['Book-Rating'].sort_values(ascending=False)
reader = Reader(rating_scale=(1,10))
my_data = Dataset.load_from_df(sample_df[['User-ID','Book-Title','Book-Rating']],reader=reader)

# Hold-Out
train, test = train_test_split(data=my_data,test_size=.2,random_state=32,shuffle=True)

svd_model = SVD()
svd_model.fit(train)
prediction = svd_model.test(test)

# Model Evaluation
accuracy.rmse(prediction) # RMSE score = 1.9066

# We can also make cross validation.
cross_validate(svd_model, data=my_data, measures=['rmse', 'mae'], cv=5, n_jobs=4, verbose=3)
user_book_matrix


# Predictions

# The user-id 254 has no attribute for "The Da Vinci Code" book as rating. That's why we can predict it.
svd_model.predict(uid=254,iid='The Da Vinci Code',verbose=3) #  Prediction(uid=254, iid='The Da Vinci Code', r_ui=None, est=8.42028038806337, details={'was_impossible': False})
# According to prediction from their latent features, it will be 8.42028038806337.

# The user-id 882 has no attribute for "The Lovely Bones: A Novel" book as rating. That's why we can predict it.
svd_model.predict(uid=882,iid='The Lovely Bones: A Novel',verbose=3) # Prediction(uid=882, iid='The Lovely Bones: A Novel', r_ui=None, est=8.27295817207916, details={'was_impossible': False})
# As seen above, the rating will be 8.27295817207916.

# We can also validate the estimations ourselves. When we look at the user_book_matrix, we can see that 278176 user gave 9.0 rating to "Wild Animus" book.
svd_model.predict(uid=278176, iid='Wild Animus',verbose=3) # Prediction(uid=278176, iid='Wild Animus', r_ui=None, est=8.688627437567334, details={'was_impossible': False})
# As you can see, the estimation is 8.688627437567334 and the real rating was 9.0.


# Model Tuning
param_grid = {'n_factors':[15,25],
              'n_epochs':[15,25],
              'lr_all':[0.005,0.009],
              'reg_all':[0.01,0.03]}

grid_searchcv = GridSearchCV(SVD,param_grid,measures=['rmse', 'mae'],cv=5,n_jobs=4,joblib_verbose=3)
grid_searchcv.fit(my_data)
grid_searchcv.best_params
# {'rmse': {'n_factors': 25, 'n_epochs': 15, 'lr_all': 0.005, 'reg_all': 0.01},
#  'mae': {'n_factors': 15, 'n_epochs': 25, 'lr_all': 0.005, 'reg_all': 0.01}}

# Final model after hyperparameter optimization
svd_model = SVD(**grid_searchcv.best_params['rmse'])
data = my_data.build_full_trainset()
svd_model.fit(data)

user_book_matrix
# Let's make prediction for users and compare the before and after model tuning.

svd_model.predict(uid=254,iid='The Da Vinci Code',verbose=3)  # Prediction(uid=254, iid='The Da Vinci Code', r_ui=None, est=8.2275096075138, details={'was_impossible': False})
# it will be 8.2275096075138. However, it was 8.42028038806337 before optimization.

svd_model.predict(uid=882,iid='The Lovely Bones: A Novel',verbose=3) # Prediction(uid=882, iid='The Lovely Bones: A Novel', r_ui=None, est=8.58612975659019, details={'was_impossible': False})
# it will be 8.58612975659019. However, it was 8.27295817207916.

isim = ["y","a","a","v","s","c","a","i","r"]
tek_liste = []
cift_liste = []
[cift_liste.append(isim[index]) if index % 2 == 0 else tek_liste.append(isim[index]) for index in range(len(isim))]