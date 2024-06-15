import os
import json
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from utils import *
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
import argparse

def most_frequent_element(lst):
    if not lst:  
        return None, 0
    counter = Counter(lst)
    most_common_element, count = counter.most_common(1)[0]
    return most_common_element, count

def find_max_indices(array, offset = 10):
    return offset + np.argmax(array[offset:])

content_path = Path('./')
dset_config = json.load(open(os.path.join(content_path,'config.json')))

def evaluate_clustering(dset, model_name, is_single_only=False):
    if is_single_only:
        model_name = model_name + '_single'
    processed = json.load(open(content_path / 'resources'/ dset/ f'processed_{model_name}_embedings.json'))
    print(dset)
    descriptions = []
    embeddings = []
    names = []
    cnt = 0
    for i in processed:
        for kc in i['kcs']:
            descriptions.append(kc['description'])
            embeddings.append(kc['embedding'])
            names.append(kc['name'])
            kc['id'] = cnt
            cnt += 1
    wcss = []
    silhouette_scores = []


    for i in tqdm(range(2, 100)): 
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(embeddings)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))

    json.dump({'wcss':wcss,'silhouette': silhouette_scores}, open(content_path / 'resources'/ dset/ f'{model_name}_cluster_scores.json','w'))
    plt.rc('font', family='Times New Roman')
    # Plot WCSS to find the elbow
    plt.figure(figsize=(12, 4))
    plt.suptitle(f'{dset}', fontsize=16) 
    plt.subplot(1, 2, 1)
    plt.plot(range(2, 200), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')

    # Plot Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(range(2, 200), silhouette_scores)
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.tight_layout(rect=[0, 0, 1, 1])    
    plt.savefig(f'figures/{model_name}_{dset}_clustering_score.png', dpi=1200)
    plt.savefig(f'figures/{model_name}_{dset}_clustering_score.pdf', dpi=1200)
    knee_locator = KneeLocator(range(2,200), wcss, curve='convex', direction='decreasing')
    # elbow_point = knee_locator.elbow
    # print(f'Picked elbow: {elbow_point}')
# We tried to use elbow_point, but since it's not stable, we decided to use silhouette_scores instead.

    optimal_clusters = find_max_indices(silhouette_scores) + 2
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(embeddings)
    clusters = kmeans.labels_
    embeddings = np.array(embeddings)
    kc_list = []
    # Select a representative sentence for each cluster
    if not is_single_only:
        for i in range(optimal_clusters):
            cluster_indices = np.where(clusters == i)[0]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            similarities = np.matmul(cluster_embeddings, centroid)
            representative_idx = np.argmax(similarities)
            print(f"Cluster {i+1}: {descriptions[cluster_indices[representative_idx]]}\n")
            kc_list.append(descriptions[cluster_indices[representative_idx]])    

    idx2cluster = {}
    cluster_id2name = {}
    for i in range(optimal_clusters):
        cluster_indices = np.where(clusters == i)[0]
        names_in_cluster = []
        for idx in cluster_indices:
            idx2cluster[idx] = i
            if not is_single_only:
                names_in_cluster.append(names[idx])
            
        if not is_single_only:
            name, _ = most_frequent_element(names_in_cluster)
            cluster_id2name[i] = name
        

    for idx, i in enumerate(processed):
        if is_single_only:
            i['kcs'] = [{
                'kc_id': idx2cluster[idx],
                'kc_name': idx2cluster[idx],
            }]
        else:
            for kc in i['kcs']:
                kc['kc_id'] = idx2cluster[kc['id']]
                kc['kc_name'] = cluster_id2name[kc['kc_id']]
    json.dump(convert_ndarrays(processed), open(content_path / 'resources'/ dset/ f'{model_name}_processed_embedings.json','w'))
    json.dump(cluster_id2name, open(content_path / 'resources'/ dset/ f'{model_name}_cluster_id2name.json','w'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get dataset name')
    dataset_choices = list(dset_config.keys())
    parser.add_argument(
        'dataset', 
        type=str, 
        choices=dataset_choices + ['all'],
        default='oli_statics',
        nargs='?',
        help='The dataset string to be processed. Choices: ' + ', '.join(dataset_choices)
    )
    parser.add_argument(
        'model', 
        type=str, 
        choices=['t5', 'openai_3'],
        default='t5',
        nargs='?',
        help='select embedding model. t5 or openai_3 '
    )
    parser.add_argument(
        '--single_kc',
        action='store_true',
        help='Disable multiple KCs'
    )
    args = parser.parse_args()
    target_dsets = [args.dataset] if args.dataset != 'all' else dataset_choices
    for dset in target_dsets:
        evaluate_clustering(dset,args.model, args.single_kc)