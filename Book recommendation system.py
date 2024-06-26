#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


book_df = pd.read_csv('Books.csv')
ratings_df = pd.read_csv('Ratings.csv').sample(40000)
user_df = pd.read_csv('Users.csv')
user_rating_df = ratings_df.merge(user_df, left_on = 'User-ID', right_on = 'User-ID')


# In[3]:


book_user_rating = book_df.merge(user_rating_df, left_on = 'ISBN',right_on = 'ISBN')
book_user_rating = book_user_rating[['ISBN', 'Book-Title', 'Book-Author', 'User-ID', 'Book-Rating']]
book_user_rating.reset_index(drop=True, inplace = True)


# In[4]:


d ={}
for i,j in enumerate(book_user_rating.ISBN.unique()):
    d[j] =i
book_user_rating['unique_id_book'] = book_user_rating['ISBN'].map(d)


# In[5]:


users_books_pivot_matrix_df = book_user_rating.pivot(index='User-ID', 
                                                          columns='unique_id_book', 
                                                          values='Book-Rating').fillna(0)


# In[6]:


users_books_pivot_matrix_df.head()


# In[7]:


users_books_pivot_matrix_df = users_books_pivot_matrix_df.values
users_books_pivot_matrix_df


# In[8]:


from scipy.sparse.linalg import svds

NUMBER_OF_FACTORS_MF = 15

U, sigma, Vt = svds(users_books_pivot_matrix_df, k = NUMBER_OF_FACTORS_MF)


# In[9]:


sigma = np.diag(sigma)
sigma.shape


# In[10]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
all_user_predicted_ratings


# In[11]:


def top_cosine_similarity(data, book_id, top_n=10):
    index = book_id 
    book_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(book_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

def similar_books(book_user_rating, book_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    book_user_rating[book_user_rating.unique_id_book == book_id]['Book-Title'].values[0]))
    for id in top_indexes + 1:
        print(book_user_rating[book_user_rating.unique_id_book == id]['Book-Title'].values[0])


# In[12]:


k = 50
movie_id =25954  
top_n = 3
sliced = Vt.T[:, :k]

similar_books(book_user_rating, 25954, top_cosine_similarity(sliced, movie_id, top_n))


# In[ ]:




