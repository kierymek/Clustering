import xml.etree.ElementTree as tree_element
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

tree = tree_element.parse('dataset.xml')
root = tree.getroot()
articles = []

for element in root:
    doc = {}
    doc['contributors_number'] = len(element[0].findall('author')) + len(element[0].findall('editors'))
    try:
        doc['year'] = int(element[0].find('year').text)
        pages = element[0].find('pages').text
        separator = pages.find('-')
        doc['pages'] =  int(pages[separator+1:]) - int(pages[:separator])
        articles.append(doc)
    except AttributeError:
        continue
    except ValueError:
        continue
    
pages = []
contributors = []
years = []

for item in articles:
    if item['pages'] > 1000: continue
    pages.append(item['pages'])
    contributors.append(item['contributors_number'])
    years.append(item['year'])

vec1 =  pd.DataFrame(data={'contributors': contributors, 'pages': pages})
vec2 =  pd.DataFrame(data={'years': years, 'contributors': contributors})
vec3 =  pd.DataFrame(data={'years': years, 'pages': pages})
vectors = [vec3, vec2, vec1]

df = pd.DataFrame(data={'contributors': contributors, 'pages': pages, 'years': years})

distortions = []
for i in range(1, 21):
    KM = KMeans(n_clusters=i, random_state=0).fit(df)
    distortions.append(KM.inertia_)

plt.figure('clustering records')
plt.subplot(224)
plt.plot([i for i in range(1, 21)], distortions,'r-')
plt.xlabel('clusters')
plt.ylabel('distortion')

pca = PCA()
pca_data = pca.fit_transform(pd.DataFrame(data=df))

for i, vec in enumerate(vectors):

    kmeans = KMeans(n_clusters=4, random_state=0).fit(vec)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    data = vec.values

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    h = 0.3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.subplot(221+i)
    plt.imshow(
        z, 
        interpolation='gaussian',
        extent=(x_min, x_max, y_min, y_max),
        aspect='auto',
        origin='lower',
        cmap='RdPu_r'
    )
    if i==0 :
        plt.xlabel(df.columns[2])
        plt.ylabel(df.columns[1])
    if i==1:
        plt.xlabel(df.columns[2])
        plt.ylabel(df.columns[0])
    if i==2:
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])

    plt.plot(data[:, 0], data[:, 1], 'cx', markersize=7)
    

plt.show()
