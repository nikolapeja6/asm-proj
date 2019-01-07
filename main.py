
import sys
import os
import zipfile
import os.path as path
import csv
import networkx as nx
import matplotlib.pyplot as pl
import math

DATA_DIR: str = 'data'
ZIP_FILE_NAME: str = 'ASM_PZ2_podaci_1819.zip'
PRIMARY_DATASET: str = 'IMDB-Movie-Data.csv'
SECONDARY_DATASET_ACTORS: str = 'secondary_dataset_actors.csv'

# Primary dataset.
primary_dataset_header = None
primary_dataset = list()

# Secondary dataset.
actor_dictionary = dict()
movie_array = list()


def data_path(file_name: str):
    "Returns relative path to data file passed as the argument."
    return os.path.join(DATA_DIR, file_name)


def extract_csv_from_zip(clean: bool = False):
    "Extracts the data from the provided zip file if no extracted data is found."
    if (not clean) and path.isfile(data_path(PRIMARY_DATASET)):
        print(PRIMARY_DATASET + ' already extracted.')
    else:
        print('Extracting data from '+ZIP_FILE_NAME)
        exists = os.path.isfile(data_path(ZIP_FILE_NAME))

        if not exists:
            raise OSError("Error -file '"+ZIP_FILE_NAME+"' not found. Aborting.")

        with zipfile.ZipFile(data_path(ZIP_FILE_NAME), 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)


def extract_secondary_dataset(clean: bool = False):
    global primary_dataset_header
    global primary_dataset
    global actor_dictionary
    global movie_array
    print("Extracting secondary dataset.")

    primary_dataset.clear()
    actor_dictionary.clear()
    movie_array.clear()

    extract_csv_from_zip(clean)

    with open(data_path(PRIMARY_DATASET), 'r') as csvFile:
        reader = csv.reader(csvFile)

        # Read primary dataset.
        for row in reader:
            if primary_dataset_header is None:
                primary_dataset_header = row
            else:
                primary_dataset.append(row)

    csvFile.close()

    # Get indexes of columns of interest.
    title_index: int = primary_dataset_header.index("Title")
    actors_index: int = primary_dataset_header.index("Actors")
    movie_index: int = primary_dataset_header.index("Rank")

    print("Creating secondary actor dataset.")

    for row in primary_dataset:
        movie_id: int = row[movie_index]
        actors: str = row[actors_index]

        tmp_list = list()

        for actor in actors.split(","):
            actor = actor.strip()
            tmp_list.append(actor)
            if actor in actor_dictionary:
                actor_dictionary[actor].append(movie_id)
            else:
                actor_dictionary[actor] = [movie_id]

        movie_array.append(tmp_list)

    with open(data_path(SECONDARY_DATASET_ACTORS), 'w') as csvFile:
        # Write header.
        csvFile.write("Actor, Movies\n")
        for key, value in actor_dictionary.items():
            csvFile.write(key.lstrip())
            for id in value:
                csvFile.write(",")
                csvFile.write(id)
            csvFile.write("\n")
    csvFile.close()

    print("Finished creating secondary dataset.")


# Main.
extract_secondary_dataset()

actor_graph = nx.Graph()

actor_graph.add_nodes_from(actor_dictionary.keys())


for actors in movie_array:
    for actor1 in actors:
        for actor2 in actors:
            if actor_graph.has_edge(actor1, actor2):
                actor_graph[actor1][actor2]['weight'] += 1
            else:
                actor_graph.add_edge(actor1, actor2, weight=1)

print(len(actor_graph.edges()))
number_of_nodes: int = len(actor_graph.nodes())
print(len(actor_graph.nodes()))

#pos = nx.spring_layout(actor_graph, iterations=5000, )
#pos = nx.random_layout(actor_graph)
n: int = 3
pos = nx.spring_layout(actor_graph, k=(1/math.sqrt(number_of_nodes))*n)
pl.figure(figsize=(20, 20))  # Don't create a humongous figure
nx.draw_networkx(actor_graph, pos, node_size=30, font_size='xx-small', with_labels=False)
pl.axis('off')
#pl.show()
pl.savefig('actors_default_dstx'+str(n)+'.pdf', format='pdf', dpi=900)

print("End")