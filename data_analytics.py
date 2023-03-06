import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import movielens_utils
import pickle as pk
import torch
from utils.architecture import MLP, MLP_DropOut
from pytorch_tabular import TabularModel


print("Data Analytics Demo")

#path_movie = "./csv_files_input/" + sys.argv[1]
#movie_dataframe = pd.read_csv(path_movie)
#path_genome_scores = "./csv_files_input/" + sys.argv[2]
#tag_relevance_dataframe = pd.read_csv(path_genome_scores)
#path_genome_tag = "./csv_files_input/" + sys.argv[3]
#tag_name_dataframe = pd.read_csv(path_genome_tag)

pca_df = pd.read_csv("./csv_files/pca_df.csv")
print(pca_df)
pca_df = pca_df.drop(columns='rating')

print("*****************DATASET********************\n")

#print("Movie: \n")
#print(movie_dataframe)
#print("\n")

#print("Tag Relevance: \n")
#print(tag_relevance_dataframe)
#print("\n")

#print("Tag Name: \n")
#print(tag_name_dataframe)
#print("\n")



#movie_splitted_genre, titles = movielens_utils.movie_with_splitted_genre(movie_dataframe)
#tag_relevance_movies = movielens_utils.tag_relevance_movies_creation(tag_name_dataframe, tag_relevance_dataframe)

new_dataframe = movielens_utils.return_new_dataframe()
new_dataframe.to_csv("./utils/new_df.csv", index=False)

new_dataframe = pd.read_csv("./utils/new_df.csv")
final_dataframe = pd.concat([new_dataframe, pca_df])
final_dataframe = final_dataframe.drop(columns='movieId')

print("****************** FINAL DATASET **************\n")
print(final_dataframe)
print("\n")

print("**********************NON-DEEP METHODS*******************************")
#final_dataframe.columns = final_dataframe.columns.astype('str')
X = final_dataframe

X = X.drop(columns='title')
print(X)
X.columns = X.columns.astype('str')

scaler = StandardScaler()
X_s = scaler.fit_transform(X)
print("Shape after scaling: " + str(X_s.shape))
print("Scaled dataset")
print(X_s)

pca = PCA(n_components=120)
X_t = pca.fit_transform(X_s)

print("PCA result")
print(X_t.shape)


print("Here's the new films catalog")
print(new_dataframe.title)

film = int(input(("Insert a number between 1 and 30\n")))

X_nd = [X_t[film,:]]

knn_regression = pk.load(open('nd_supervised_models/knn_regression.pkl', "rb"))
knn_predicted = knn_regression.predict(X_nd)
print("Prediction with KNN: " + str(knn_predicted))

linear_regression = pk.load(open('nd_supervised_models/linear_regression.pkl', "rb"))
linear_predicted = linear_regression.predict(X_nd)
print("Prediction with Linear regression: " + str(linear_predicted))

random_forest_regression = pk.load(open('nd_supervised_models/random_forest_regression.pkl', "rb"))
random_predicted = random_forest_regression.predict(X_nd)
print("Prediction with Random forest: " + str(random_predicted))


print("\n*******************************************DEEP LEARNING*****************************************")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dropout_model = MLP_DropOut(679,32,16,0.25,2)
dropout_model.load_state_dict(torch.load('./neural_network/best_model_dropout/dim132dim216ep50bs16lr0.0001d2'))

no_dropout_model = MLP(679,16,8)
no_dropout_model.load_state_dict(torch.load('./neural_network/best_model_without_dropout/dim116dim28ep50bs8lr0.0001'))

dl_pca = PCA(n_components=679)
X_dl = dl_pca.fit_transform(X)
print("PCA result")
print(X_dl.shape)

X_dl = [X_dl[3,:]]

X_dl = torch.FloatTensor(X_dl) 

dropout_model.eval()
dropout_pred = dropout_model(X_dl)
#dropout_pred = dropout_model.predict(dl_pca)
print("Prediction with dropout: " + str(dropout_pred.item()))

no_dropout_model.eval()
no_dropout_pred = no_dropout_model(X_dl)
#no_dropout_pred = dropout_model.predict(dl_pca)
print("Prediction without dropout: " + str(no_dropout_pred.item()))

print("\n******************************************TABNET*************************************************")

#final_dataframe = pd.concat([new_dataframe, pca_df])
#final_dataframe = final_dataframe.drop(columns='movieId')
X_title = final_dataframe
X_tn = X_title.iloc[[3]]
print(X_tn)
tabnet_model = TabularModel.load_from_checkpoint("./tabnet/model")
#result = tabnet_model.evaluate(X_tn)
tabnet_pred = tabnet_model.predict(X_tn)

print("Prediction with tabnet model: " + str(tabnet_pred.rating_prediction))


