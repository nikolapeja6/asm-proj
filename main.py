import os
import zipfile
import os.path as path
import csv
import networkx as nx
import matplotlib.pyplot as pl
import math
import datetime
import time
from networkx.algorithms import community
from matplotlib import colors as mcolors
import numpy as np
from difflib import SequenceMatcher
from collections import OrderedDict
from collections import Counter
import matplotlib.backends.backend_pdf
import operator
import winsound
import inspect


DATA_DIR: str = 'data'
RESULTS_DIR: str = 'results'
ZIP_FILE_NAME: str = 'ASM_PZ2_podaci_1819.zip'
PRIMARY_DATASET: str = 'IMDB-Movie-Data.csv'
SECONDARY_DATASET_ACTORS: str = 'secondary_dataset_actors.csv'

# Primary dataset.
primary_dataset_header = None
primary_dataset = list()

# Secondary dataset.
actor_dictionary = dict()
movie_array = list()
genre_dictionary = dict()
genres_by_movies = list()
movie_dictionary = dict()
director_dictionary = dict()
link_director_actor = list()

list_of_movies = list()
list_of_actors = set()
list_of_genres = set()
link_acted_in = list()
link_is_genre = list()

def data_path(file_name: str):
    "Returns relative path to data file passed as the argument."
    return os.path.join(DATA_DIR, file_name)


def results_path(file_name: str):
    return os.path.join(RESULTS_DIR, file_name)

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


def suffix():
        ts = time.time()
        return "___"+datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def my_average_shortest_path(graph: nx.Graph):
    cnt: int = 0
    length: int = 0

    for node1 in graph.nodes():
        for node2 in graph.nodes():
            if node1 != node2 and nx.has_path(graph, node1, node2):
                length += len(nx.shortest_path(graph, node1, node2))-1
                cnt += 1

    return length/cnt


def extract_secondary_dataset(clean: bool = False, revenue_filter: int = 0):
    global primary_dataset_header
    global primary_dataset
    global actor_dictionary
    global movie_array
    global genre_dictionary
    global genres_by_movies
    global movie_dictionary

    global list_of_movies
    global list_of_actors
    global list_of_genres
    global link_acted_in
    global link_is_genre
    global director_dictionary
    global link_director_actor
    print("Extracting secondary dataset.")

    primary_dataset.clear()
    actor_dictionary.clear()
    movie_array.clear()
    genre_dictionary.clear()
    genres_by_movies.clear()
    movie_dictionary.clear()

    list_of_movies.clear()
    list_of_actors.clear()
    list_of_genres.clear()
    link_acted_in.clear()
    link_is_genre.clear()
    director_dictionary.clear()
    link_director_actor.clear()

    extract_csv_from_zip(clean)
    primary_dataset_header = None

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
    genre_index: int = primary_dataset_header.index("Genre")
    year_index: int = primary_dataset_header.index("Year")
    revenue_index: int = primary_dataset_header.index("Revenue (Millions)")
    director_index: int = primary_dataset_header.index("Director")

    print("Creating secondary actor dataset.")

    for row in primary_dataset:
        movie_id: int = row[movie_index]
        actors: str = row[actors_index]
        genres: str = row[genre_index]
        year: int = row[year_index]
        title: str = row[title_index]
        title = title.strip()
        director: str = row[director_index]
        revenue: float = row[revenue_index].strip()

        if revenue is None or revenue == "":
            revenue = 0
        else:
            revenue = float(revenue)

        # Filer
        if revenue < revenue_filter:
            continue

        if director in director_dictionary:
            director_dictionary[director]+=1
        else:
            director_dictionary[director] = 1

        list_of_movies.append([title, year])

        tmp_list1 = list()
        tmp_list2 = list()
        tmp_list3 = list()

        tmp_list1.append(title)
        tmp_list1.append(year)
        movie_dictionary[movie_id] = tmp_list1

        for actor in actors.split(","):
            actor = actor.strip()
            #clean
            if actor == "Evan Rachel Wood":
                actor = "Rachel Wood"

            list_of_actors.add(actor)
            link_acted_in.append([int(movie_id)-1, actor])

            link_director_actor.append([director, actor])

            tmp_list2.append(actor)
            if actor in actor_dictionary:
                actor_dictionary[actor].append(movie_id)
            else:
                actor_dictionary[actor] = [movie_id]

        movie_array.append(tmp_list2)

        for genre in genres.split(","):
            genre = genre.strip()

            list_of_genres.add(genre)
            link_is_genre.append([int(movie_id)-1, genre])

            tmp_list3.append(genre)
            if genre in genre_dictionary:
                genre_dictionary[genre].append(movie_id)
            else:
                genre_dictionary[genre] = [movie_id]

        genres_by_movies.append(tmp_list3)

    '''
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
    '''

    print("Finished creating secondary dataset.")

    # Found inconsistency:
    #   - Evan Rachel Wood : Rachel Wood
    '''
    print("***")
    for actor1 in list_of_actors:
        for actor2 in list_of_actors:
            if actor1 != actor2 and similar(actor1, actor2) > 0.8:
                print(actor1+" : "+actor2)
    print("***")
    for genre1 in list_of_genres:
        for genre2 in list_of_genres:
            if genre1 != genre2 and similar(genre1, genre2) > 0.5:
                print(genre1+" : "+genre2)
    
    print("***")

    for key1, value1 in director_dictionary.items():
        for key2, value2 in director_dictionary.items():
            if key1 != key2 and similar(key1, key2) > 0.8:
                print(key1+" : "+key2)
    print("***")
    '''

def create_actor_network():
    actor_graph = nx.Graph()

    actor_graph.add_nodes_from(actor_dictionary.keys())

    for actors in movie_array:
        for actor1 in actors:
            for actor2 in actors:
                if actor_graph.has_edge(actor1, actor2):
                    actor_graph[actor1][actor2]['weight'] += 0.5
                else:
                    actor_graph.add_edge(actor1, actor2, weight=0.5)


    actor_graph.remove_edges_from(actor_graph.selfloop_edges())
    return actor_graph


def save_actor_graph_as_pdf(actor_graph: nx.Graph, color='r', fileName = ""):
    #pos = nx.spring_layout(actor_graph, iterations=5000, )
    #pos = nx.random_layout(actor_graph)
    number_of_nodes: int = len(actor_graph.nodes())
    n: int = 4
    pos = nx.spring_layout(actor_graph, k=(1/math.sqrt(number_of_nodes))*n)
    pl.figure(figsize=(20, 20))  # Don't create a humongous figure
    nx.draw_networkx(actor_graph, pos, node_size=30, font_size='xx-small', with_labels=False, node_color=color)
    pl.axis('off')
    #pl.show()
    if fileName != "":
        pl.savefig(fileName, format='pdf', dpi=900)
    else:
        pl.savefig('actors_default_dstx'+str(n)+suffix()+'.pdf', format='pdf', dpi=900)


def create_genre_network():
    genre_graph = nx.Graph()

    genre_graph.add_nodes_from(genre_dictionary.keys())

    for genres in genres_by_movies:
        for genre1 in genres:
            for genre2 in genres:
                if genre_graph.has_edge(genre1, genre2):
                    genre_graph[genre1][genre2]['weight'] += 0.5
                else:
                    genre_graph.add_edge(genre1, genre2, weight=0.5)

    genre_graph.remove_edges_from(genre_graph.selfloop_edges())
    return genre_graph


def save_genre_graph_as_pdf(genre_graph: nx.Graph):
    #pos = nx.spring_layout(actor_graph, iterations=5000, )
    #pos = nx.random_layout(actor_graph)
    number_of_nodes: int = len(genre_graph.nodes())
    n: int = 5
    pos = nx.spring_layout(genre_graph, iterations=200)
    pl.figure(figsize=(30, 30))  # Don't create a humongous figure
    nx.draw_networkx(genre_graph, pos, node_size=5000, font_size='medium', style='dotted', with_labels=True, node_shape='s', font_color='white')
    pl.axis('off')
    #pl.show()
    pl.savefig('genre_default'+str(n)+suffix()+'.pdf', format='pdf', dpi=900)


def create_movie_network():
    movie_graph = nx.DiGraph()

    for movie_id in movie_dictionary.keys():
        movie_graph.add_node(movie_dictionary[movie_id][0])

    for key, value in actor_dictionary.items():
        for movie_id1 in value:
            for movie_id2 in value:
                year1: int = movie_dictionary[movie_id1][1]
                year2: int = movie_dictionary[movie_id2][1]

                node1: str = movie_dictionary[movie_id1][0]
                node2: str = movie_dictionary[movie_id2][0]

                if year2 < year1:
                    node1, node2 = node2, node1

                if movie_graph.has_edge(node1, node2):
                    movie_graph[node1][node2]['weight'] += 1
                else:
                    movie_graph.add_edge(node1, node2, weight=1)

    movie_graph.remove_edges_from(movie_graph.selfloop_edges())
    return movie_graph


def save_movie_graph_as_pdf(movie_graph: nx.Graph):
    #pos = nx.spring_layout(actor_graph, iterations=5000, )
    #pos = nx.random_layout(actor_graph)
    number_of_nodes: int = len(movie_graph.nodes())
    n: int = 2
    pos = nx.spring_layout(movie_graph, k=(1/math.sqrt(number_of_nodes))*n)
    pl.figure(figsize=(20, 20))  # Don't create a humongous figure
    nx.draw_networkx(movie_graph, pos, node_size=30, font_size='xx-small', with_labels=False, node_shape='o',
                     font_color='white', edge_color='grey')
    pl.axis('off')
    #pl.show()
    pl.savefig('movie_default_20_x'+str(n)+suffix()+'.pdf', format='pdf', dpi=900)


def sort_nodes_by_degree(graph: nx.Graph):
    ret = list(graph.degree(graph.nodes()))
    ret.sort(key=lambda x: x[1], reverse=True)
    return ret

def sort_nodes_by_weighed_degree(graph: nx.Graph):
    ret = list(graph.degree(graph.nodes(), 'weight'))
    ret.sort(key=lambda x: x[1], reverse=True)
    return ret

def sort_edges_by_weight(graph: nx.Graph):
    #return sorted(list(graph.edges_iter(data='weight')), key=lambda x: x['weight'], reverse=True)
    return sorted(graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)


def check():
    print("Executing "+inspect.stack()[1].function)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def q1(actors: nx.Graph, top: int = 10):
    check()

    lst1 = sort_nodes_by_degree(actors)[0:top]
    lst2 = sort_nodes_by_weighed_degree(actors)[0:top]

    with open(results_path("q1.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Rank", "Top actors", "Degree", "Top Actors", "Weighed degree"])

        for rank in range(0, top):
            writer.writerow([rank+1, lst1[rank][0], lst1[rank][1], lst2[rank][0], lst2[rank][1]])

    csvFile.close()


def q2(actor_network: nx.Graph):
    check()

    n: int = actor_network.number_of_nodes()
    with open(results_path("q2.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Avg degree", "Avg weighed degree"])

        avg_degree = sum(degrees[1] for degrees in sort_nodes_by_degree(actor_network))/n
        avg_wdegree = sum(degrees[1] for degrees in sort_nodes_by_weighed_degree(actor_network))/n

        writer.writerow([avg_degree, avg_wdegree])
    csvFile.close()


def q3(actor_network: nx.Graph, top: int = 10):
    check()

    answer = list()

    for actor,foo in sort_nodes_by_weighed_degree(actor_network)[0:top]:
        cnt: int = 0
        genres = set()
        for acted_in in link_acted_in:
            if acted_in[1] == actor:
                cnt = cnt+1
                for is_genre in link_is_genre:
                    if acted_in[0] == is_genre[0]:
                        genres.add(is_genre[1])
        genres = sorted(genres)
        answer.append([actor, cnt, get_actors_most_played_genre(actor)])
        #answer.append([actor, cnt, ",".join(genres)])

    with open(results_path("q3.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Actor", "Movies num", "Genre"])

        writer.writerows(answer)
    csvFile.close()

def random_color():
    #dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    cls = [name for hsv, name in by_hsv]
    return cls[np.random.choice(range(len(cls)))]
    #rgb = list(np.random.choice(range(256), size=3))
    #R, G, B = rgb[0], rgb[1], rgb[2]
    #R + G * (256) + B * (256 ^ 2)
    #return R + G * (256) + B * (256 ^ 2)
    #'%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])


def generate_communities(actor_network: nx.Graph):
    communities_generator = community.girvan_newman(actor_network)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    answer = sorted(map(sorted, next_level_communities))
    return answer


def q4(actor_network: nx.Graph):
    check()

    answer = generate_communities(actor_network)

    with open(results_path("q4.csv"), 'w', newline='') as csvFile:
        #writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        for index in range(-1, len(answer[0])+1):
            if index == -1:
                csvFile.write("")
                for i in range(0, len(answer)):
                    csvFile.write(", Commune "+str(i))
            else:
                for commune in answer:
                    csvFile.write(", ")
                    if index < len(commune):
                        csvFile.write(commune[index])
            csvFile.write("\r\n")

    csvFile.close()

    colors = list()
    community_colors = list()
    for comm in answer:
        community_colors.append(random_color())

    for node in actor_network.nodes():
        for index in range(0, len(answer)):
            if answer[index].__contains__(node):
                break
        colors.append(community_colors[index])

    #save_actor_graph_as_pdf(actor_network, color=colors, fileName=results_path("q4.pdf"))

    pdf = matplotlib.backends.backend_pdf.PdfPages(results_path("q4.pdf"))
    number_of_nodes: int = len(actor_network.nodes())
    n: int = 4
    pos = nx.spring_layout(actor_network, k=(1 / math.sqrt(number_of_nodes)) * n)
    pl.figure(figsize=(20, 20))  # Don't create a humongous figure
    nx.draw_networkx(actor_network, pos, node_size=30, font_size='xx-small', with_labels=False, node_color=colors)
    pl.axis('off')
    pdf.savefig(pl.gcf(), dpi=900)

    fig = pl.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    pl.title('Distribution of actors across clusters')

    cluter_counter = Counter(colors)

    frequencies = cluter_counter.values()
    names = list(cluter_counter.keys())

    x_coordinates = np.arange(len(cluter_counter))
    ax.bar(x_coordinates, frequencies, align='center', color=names)

    ax.xaxis.set_major_locator(pl.FixedLocator(x_coordinates))
    ax.xaxis.set_major_formatter(pl.FixedFormatter(names))
    pdf.savefig(fig, dpi=900)

    pdf.close()


def get_actors_most_played_genre(actor: str):
    check()

    gnrs = dict()
    for genre in list_of_genres:
        gnrs[genre] = 0

    for item in link_is_genre:
        movie = item[0]
        genre = item[1]
        movie += 1
        if str(movie) in actor_dictionary[actor]:
            gnrs[genre] +=1


    max: int = 0
    for value in gnrs.values():
        if value > max:
            max = value

    tmp = list()
    for genre, value in gnrs.items():
        if value == max:
            tmp.append(genre)

    tmp.sort()
    #gnrs = sorted(gnrs, key=gnrs.get, reverse=True)
    return tmp[0]


def get_dict_of_most_genres_per_actor():
    ret = dict()
    for actor in list_of_actors:
        ret[actor] = get_actors_most_played_genre(actor)

    return ret

def q5(actor_graph: nx.Graph):
    check()

    actor_genre_dict = get_dict_of_most_genres_per_actor()
    genre_counter = Counter(actor_genre_dict.values())

    color_genre_dic = dict()
    for genre in list_of_genres:
        color_genre_dic[genre] = random_color()

    colors = list()
    for actor in actor_graph.nodes():
        colors.append(color_genre_dic[actor_genre_dict[actor]])

    n: int = 4
    number_of_nodes: int = len(actor_graph.nodes())
    pos = nx.spring_layout(actor_graph, k=(1 / math.sqrt(number_of_nodes)) * n)

    graph = nx.Graph()
    graph.add_nodes_from(actor_graph.nodes())
    for edge in actor_graph.edges():
        actor1 = edge[0]
        actor2 = edge[1]
        if actor_genre_dict[actor1] == actor_genre_dict[actor2]:
            graph.add_edge(actor1, actor2)

    pos2 = nx.spring_layout(graph, k=(1 / math.sqrt(number_of_nodes)) * n)
    
    gnrs = dict()
    for genre in list_of_genres:
        gnrs[genre] = 0
    for actor, genre in actor_genre_dict.items():
        gnrs[genre] += 1

    pdf = matplotlib.backends.backend_pdf.PdfPages(results_path("q5.pdf"))

    pl.figure(figsize=(20, 20))  # Don't create a humongous figure
    nx.draw_networkx(actor_graph, pos, node_size=30, font_size='xx-small', with_labels=False, node_color=colors)
    pl.axis('off')
    pdf.savefig(pl.gcf(), dpi=900)

    pl.figure(figsize=(20, 20))  # Don't create a humongous figure
    nx.draw_networkx(graph, pos, node_size=30, font_size='xx-small', with_labels=False, node_color=colors)
    pl.axis('off')
    pdf.savefig(pl.gcf(), dpi=900)

    pl.figure(figsize=(20, 20))  # Don't create a humongous figure
    nx.draw_networkx(graph, pos2, node_size=30, font_size='xx-small', with_labels=False, node_color=colors)
    pl.axis('off')
    pdf.savefig(pl.gcf(), dpi=900)

    fig = pl.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    pl.title('Distribution of actors across genres')

    frequencies = genre_counter.values()
    names = list(genre_counter.keys())

    tmp = list()
    for name in names:
        tmp.append(color_genre_dic[name])

    x_coordinates = np.arange(len(genre_counter))
    ax.bar(x_coordinates, frequencies, align='center', color=tmp)

    ax.xaxis.set_major_locator(pl.FixedLocator(x_coordinates))
    ax.xaxis.set_major_formatter(pl.FixedFormatter(names))
    pdf.savefig(fig, dpi=900)

    pdf.close()


def sored_nodes_on_betweenness_centrality(graph: nx.Graph):
    ret = list()
    bc = nx.betweenness_centrality(graph)
    for node in graph.nodes():
        ret.append([node,  bc[node]])

    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


def sorted_nodes_on_degree_centrality(graph: nx.Graph):
    ret = list()
    dc = nx.degree_centrality(graph)
    for node in graph.nodes():
        ret.append([node, dc[node]])

    ret.sort(key=lambda x: x[1], reverse=True)
    return ret

def sorted_nodes_on_bc_dc(bc: list, dc: list):
    ret = list()
    for item1 in bc:
        for item2 in dc:
            if item1[0] == item2[0]:
                ret.append([item1[0], item1[1]*item2[1]])
                break

    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


def q6(actor_network: nx.Graph, genre_network: nx.Graph, movie_network: nx.Graph, top: int=10):
    check()

    with open(results_path("q6.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["", "Actor", "Top BC", "Actor", "Top DC", "Actor", "Top DC*BC"]
        writer.writerow(row)

        bc = sored_nodes_on_betweenness_centrality(actor_network)
        dc = sorted_nodes_on_degree_centrality(actor_network)
        bc_dc = sorted_nodes_on_bc_dc(bc, dc)

        for i in range(0,top):
            row = [i,bc[i][0], bc[i][1],dc[i][0], dc[i][1],bc_dc[i][0], bc_dc[i][1]]
            writer.writerow(row)

        row = []
        writer.writerow(row)

        row = ["", "Genre", "Top BC", "Genre", "Top DC", "Genre", "Top DC*BC"]
        writer.writerow(row)

        bc = sored_nodes_on_betweenness_centrality(genre_network)
        dc = sorted_nodes_on_degree_centrality(genre_network)
        bc_dc = sorted_nodes_on_bc_dc(bc, dc)

        for i in range(0, top ):
            row = [i, bc[i][0], bc[i][1], dc[i][0], dc[i][1], bc_dc[i][0], bc_dc[i][1]]
            writer.writerow(row)

        row = []
        writer.writerow(row)

        row = ["", "Movie", "Top BC", "Movie", "Top DC", "Movie", "Top DC*BC"]
        writer.writerow(row)

        bc = sored_nodes_on_betweenness_centrality(movie_network)
        dc = sorted_nodes_on_degree_centrality(movie_network)
        bc_dc = sorted_nodes_on_bc_dc(bc, dc)

        for i in range(0, top ):
            row = [i, bc[i][0], bc[i][1], dc[i][0], dc[i][1], bc_dc[i][0], bc_dc[i][1]]
            writer.writerow(row)

        row = []
        writer.writerow(row)

    csvFile.close()


def q7(actor_network: nx.Graph, genre_network: nx.Graph, movie_network: nx.Graph):
    check()

    with open(results_path("q7.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Actor density", "Genre Density", "Movie density"]
        writer.writerow(row)
        row = [nx.density(actor_network),nx.density(genre_network),nx.density(movie_network)]
        writer.writerow(row)

    csvFile.close()


def my_sum(lst: dict):
    return sum(x for x in lst.values())


def my_avg(lst: dict):
    return my_sum(lst)/len(lst)


def q8(actor_network: nx.Graph, genre_network: nx.Graph, movie_network: nx.Graph):
    check()

    with open(results_path("q8.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["", "Actor", "Genre", "Movie"]
        writer.writerow(row)

        n1 = my_avg(nx.closeness_centrality(actor_network))
        n2 = my_avg(nx.closeness_centrality(genre_network))
        n3 = my_avg(nx.closeness_centrality(movie_network))

        row = ["Closeness centrality", n1, n2, n3]
        writer.writerow(row)

        n1 = my_avg(nx.betweenness_centrality(actor_network))
        n2 = my_avg(nx.betweenness_centrality(genre_network))
        n3 = my_avg(nx.betweenness_centrality(movie_network))

        row = ["Betweenness centrality", n1, n2, n3]
        writer.writerow(row)

        n1 = my_avg(nx.degree_centrality(actor_network))
        n2 = my_avg(nx.degree_centrality(genre_network))
        n3 = my_avg(nx.degree_centrality(movie_network))

        row = ["Normalized degree centrality", n1, n2, n3]
        writer.writerow(row)

        n1 = my_avg(nx.eigenvector_centrality_numpy(actor_network))
        n2 = my_avg(nx.eigenvector_centrality_numpy(genre_network))
        n3 = my_avg(nx.eigenvector_centrality_numpy(movie_network))

        row = ["Eigenvector centrality", n1, n2, n3]
        writer.writerow(row)

        n1 = my_avg(nx.katz_centrality_numpy(actor_network))
        n2 = my_avg(nx.katz_centrality_numpy(genre_network))
        n3 = my_avg(nx.katz_centrality_numpy(movie_network))

        row = ["Katz centrality", n1, n2, n3]
        writer.writerow(row)

        n1 = my_avg(nx.edge_betweenness_centrality(actor_network))
        n2 = my_avg(nx.edge_betweenness_centrality(genre_network))
        n3 = my_avg(nx.edge_betweenness_centrality(movie_network))

        row = ["Edge betweenness centrality", n1, n2, n3]
        writer.writerow(row)

        # Error
        '''
        print(nx.percolation_centrality(actor_network))
        n1 = my_avg(nx.percolation_centrality(actor_network))
        n2 = my_avg(nx.percolation_centrality(genre_network))
        n3 = my_avg(nx.percolation_centrality(movie_network))
        
        row = ["Percolation centrality", n1, n2, n3]
        writer.writerow(row)
        '''

        n1 = my_avg(nx.pagerank(actor_network))
        n2 = my_avg(nx.pagerank(genre_network))
        n3 = my_avg(nx.pagerank(movie_network))

        row = ["PageRank centrality", n1, n2, n3]
        writer.writerow(row)

        n1 = nx.global_reaching_centrality(actor_network)
        n2 = nx.global_reaching_centrality(genre_network)
        n3 = nx.global_reaching_centrality(movie_network)

        row = ["Global reaching centrality", n1, n2, n3]
        writer.writerow(row)

        row = []
        writer.writerow

        n1 = nx.node_connectivity(actor_network)
        n2 = nx.node_connectivity(genre_network)
        n3 = nx.node_connectivity(movie_network)

        row = ["Node connectivity", n1, n2, n3]
        writer.writerow(row)

        n1 = nx.edge_connectivity(actor_network)
        n2 = nx.edge_connectivity(genre_network)
        n3 = nx.edge_connectivity(movie_network)

        row = ["Edge connectivity", n1, n2, n3]
        writer.writerow(row)
    
        '''
        n1 = nx.average_node_connectivity(actor_network.subgraph(generate_communities(actor_network)[0]))
        print(n1)
        n2 = nx.average_node_connectivity(genre_network)
        print(n2)
        n3 = nx.average_node_connectivity(movie_network.subgraph(generate_communities(movie_network)[0]))
        print(n3)
        row = ["Average Node Connectivity", n1, n2, n3]
        writer.writerow(row)
        '''
    csvFile.close()


def q9(actor_network: nx.Graph, genre_network: nx.Graph, movie_network: nx.Graph):
    check()

    with open(results_path("q9.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Actor average distance", "Genre average distance", "Movie average distance"]
        writer.writerow(row)

        try:
            n1 = my_average_shortest_path(actor_network)
        except nx.exception.NetworkXError:
            n1 = 'graph is not connected'

        try:
            n2 = my_average_shortest_path(genre_network)
        except nx.exception.NetworkXError:
            n2 = 'graph is not connected'

        try:
            n3 = my_average_shortest_path(movie_network)
        except nx.exception.NetworkXError:
            n3 = 'graph is not connected'

        row = [n1,n2,n3]
        writer.writerow(row)

        row = ["","",""]
        writer.writerow(row)

        row = ["Actor diameter", "Genre diameter", "Movie diameter"]
        writer.writerow(row)
        try:
            n1 = nx.diameter(actor_network)
        except nx.exception.NetworkXError:
            n1 = 'graph is not connected'

        try:
            n2 = nx.diameter(genre_network)
        except nx.exception.NetworkXError:
            n2 = 'graph is not connected'

        try:
            n3 = nx.diameter(movie_network)
        except nx.exception.NetworkXError:
            n3 = 'graph is not connected'

        row = [n1,n2,n3]
        writer.writerow(row)

    csvFile.close()


def q10(actor_network: nx.Graph, genre_network: nx.Graph, movie_network: nx.Graph):
    check()

    with open(results_path("q10.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Actor Node", "Actor clustering coefficient", "Genre Node", "Genre clustering coefficient",
               "Movie Node", "Movie clustering coefficient"]
        writer.writerow(row)

        n1 = nx.average_clustering(actor_network)
        n2 = nx.average_clustering(genre_network)
        n3 = nx.average_clustering(movie_network)

        row = ["***Network***", n1,"***Network***", n2,"***Network***", n3]
        writer.writerow(row)

        for i in range(0, max(len(actor_network.nodes()),len(genre_network.nodes()),len(movie_network.nodes()))):
            if i < len(actor_network.nodes()):
                node1 = list(actor_network.nodes())[i]
                n1 = nx.clustering(actor_network, node1)
            else:
                node1 = ""
                n1 = ""
            if i < len(genre_network.nodes()):
                node2 = list(genre_network.nodes())[i]
                n2 = nx.clustering(genre_network, node2)
            else:
                node2 = ""
                n2= ""
            if i < len(movie_network.nodes()):
                node3 = list(movie_network.nodes())[i]
                n3 = nx.clustering(movie_network, node3)
            else:
                node3 = ""
                n3 = ""

            row = [node1, n1, node2, n2, node3, n3]
            writer.writerow(row)

    csvFile.close()


def q11(actor_network: nx.Graph, genre_network: nx.Graph, movie_network: nx.Graph):
    check()

    pdf = matplotlib.backends.backend_pdf.PdfPages(results_path("q11.pdf"))

    fig = pl.figure()
    pl.title('Actor network - distribution of node degrees')
    pl.hist([val for (node, val) in actor_network.degree()], bins=50)
    pdf.savefig(fig, dpi=900)

    fig = pl.figure()
    pl.title('Genre network - distribution of node degrees')
    pl.hist([val for (node, val) in genre_network.degree()], bins=20)
    pdf.savefig(fig, dpi=900)

    fig = pl.figure()
    pl.title('Movie network - distribution of node degrees')
    pl.hist([val for (node, val) in movie_network.degree()], bins=50)
    pdf.savefig(fig, dpi=900)

    pdf.close()


def number_of_triangles(graph: nx.Graph):
    return sum(x for x in nx.triangles(actor_network).values())/3

def number_of_paths_len_2(graph: nx.Graph):
    cnt: int = 0
    for node1 in graph.nodes():
        for node2 in graph.nodes():
            if node1 != node2:
                for path in nx.all_simple_paths(graph, source=node1, target=node2, cutoff=2):
                    if len(path) == 2:
                        cnt+=1
    return cnt

def c_del(graph: nx.Graph):
    return (3*number_of_triangles(graph))/number_of_paths_len_2(graph)

def gam(graph: nx.Graph, rand: nx.Graph):
    return c_del(graph)/c_del(rand)

def lam(graph: nx.Graph, rand: nx.Graph):
    return my_average_shortest_path(graph)/ my_average_shortest_path(rand)

def s(graph: nx.Graph):
    n: int = graph.number_of_nodes()
    m: int = graph.number_of_edges()
    p: int = 2*m/(n*(n-1))
    rand = nx.erdos_renyi_graph(n,p)

    return gam(graph, rand)/lam(graph,rand)


def is_small_world(graph: nx.Graph):
    return s(graph) > 1


def q12(actor_network: nx.Graph, genre_network: nx.Graph, movie_network: nx.Graph):
    check()

    with open(results_path("q12.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Network", "Is small world?"]
        writer.writerow(row)

        row = ["Actor", is_small_world(actor_network)]
        writer.writerow(row)

        row = ["Genre", is_small_world(genre_network)]
        writer.writerow(row)

        row = ["Movie", is_small_world(movie_network)]
        writer.writerow(row)

    csvFile.close()


def q13(actor_network: nx.Graph):
    check()

    with open(results_path("q13.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Avg Distance from Kevin Bacon", "Max distance from Kevin Beacon"]
        writer.writerow(row)

        lst = nx.single_source_shortest_path_length(actor_network, "Kevin Bacon")
        avg_val = sum(int(y) for x,y in lst.items())/len(lst)
        max_val = max(int(y) for x,y in lst.items())

        row = [avg_val, max_val]
        writer.writerow(row)

    csvFile.close()

def q14(actor_network: nx.Graph, top: int=10):
    check()

    with open(results_path("q14.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Actors in the center of the network"]
        writer.writerow(row)

        subgraph = actor_network.subgraph(generate_communities(actor_network)[0])

        #periphery = nx.periphery(subgraph)


        answer = nx.closeness_centrality(subgraph)

        answer = sorted(answer.items(), key= lambda x: x[1], reverse=True)[:top]

        writer.writerows(answer)

        '''
        for item in generate_communities(actor_network)[0]:
            if actor_network.degree(item) > degree_treshold:
                row = [item]
                writer.writerow(row)
        '''

    csvFile.close()


def q15(genre_network: nx.Graph, top: int = 10):
    check()

    answer1 = sort_nodes_by_weighed_degree(genre_network)[:top]
    answer2 = sort_edges_by_weight(genre_network)[:top]

    with open(results_path("q15.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Genres", "Times"]
        writer.writerow(row)

        for item in answer1:
            row = [item[0], item[1]]
            writer.writerow(row)

        row = []
        writer.writerow(row)

        row = ["Genre 1", "Genre 2", "Times"]
        writer.writerow(row)

        for item in answer2:
            row = [item[0], item[1], item[2]['weight']]
            writer.writerow(row)

    csvFile.close()


def q16(top: int = 10):
    check()

    answer = list()
    for i in range(0, len(list_of_movies)):
        id = i+1
        title = list_of_movies[i][0]
        year = int(list_of_movies[i][1])

        cnt: int =  0
        for actor in movie_array[i]:
            for item in link_acted_in:
                movie_id = item[0]
                if actor == item[1] and int(list_of_movies[movie_id][1]) > year:
                    cnt += 1

        answer.append([title, cnt])

    answer.sort(key=lambda x: x[1], reverse=True)

    answer = answer[:top]

    with open(results_path("q16.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        writer.writerow(["Movie", "Number of parts the actors got in the following years"])
        writer.writerows(answer)

    csvFile.close()


def q17_helper(filter: int):
    check()

    actor_graph = create_actor_network()
    genre_graph = create_genre_network()
    movie_graph = create_movie_network()

    n: int = 4
    number_of_nodes: int = len(actor_graph.nodes())
    pos = nx.spring_layout(actor_graph, k=(1 / math.sqrt(number_of_nodes)) * n)

    pdf = matplotlib.backends.backend_pdf.PdfPages(results_path("q17_"+str(filter)+".pdf"))

    pl.figure(figsize=(20, 20))  # Don't create a humongous figure
    nx.draw_networkx(actor_graph, pos, node_size=30, font_size='xx-small', with_labels=False)
    pl.axis('off')
    pdf.savefig(pl.gcf(), dpi=900)

    n: int = 5
    pos = nx.spring_layout(genre_graph, iterations=200)
    fig = pl.figure(figsize=(30, 30))  # Don't create a humongous figure
    nx.draw_networkx(genre_graph, pos, node_size=5000, font_size='medium', style='dotted', with_labels=True,
                     node_shape='s', font_color='white')
    pl.axis('off')
    pdf.savefig(fig, dpi=900)

    number_of_nodes: int = len(movie_graph.nodes())
    n: int = 2
    pos = nx.spring_layout(movie_graph, k=(1/math.sqrt(number_of_nodes))*n)
    fig = pl.figure(figsize=(20, 20))  # Don't create a humongous figure
    nx.draw_networkx(movie_graph, pos, node_size=30, font_size='xx-small', with_labels=False, node_shape='o',
                     font_color='white', edge_color='grey')
    pl.axis('off')
    pdf.savefig(fig, dpi=900)

    pdf.close()


def q17():
    check()

    n = 20
    extract_secondary_dataset(True, n)
    q17_helper(n)

    n = 50
    extract_secondary_dataset(True, n)
    q17_helper(n)

    n = 100
    extract_secondary_dataset(True, n)
    q17_helper(n)

    n = 200
    extract_secondary_dataset(True, n)
    q17_helper(n)

    n = 500
    extract_secondary_dataset(True, n)
    q17_helper(n)

    extract_secondary_dataset(True)


def q18(top: int = 1):
    check()

    answer = sorted(director_dictionary.items(), key= lambda x: x[1], reverse=True)[:top]
    with open(results_path("q18.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Director", "Times"]
        writer.writerow(row)

        for item in answer:
            row = [item[0], item[1]]
            writer.writerow(row)

    csvFile.close()


def q19(filter: int = 1):
    check()

    with open(results_path("q19.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Director", "Actor", "Times"]
        writer.writerow(row)

        tmp = list()
        for item in link_director_actor:
            tmp.append(item[0]+","+item[1])

        answer = list()

        for item in set(tmp):
            cnt: int = tmp.count(item)
            if cnt > filter:
                item = item.split(",")
                row = [item[0], item[1], cnt]
                answer.append(row)
        writer.writerows(sorted(answer, reverse=True, key=lambda x: x[2]))

    csvFile.close()


def q20(top:int = 1):
    check()

    answer = dict()
    for item in list_of_movies:
        year = item[1]
        if year in answer:
            answer[year] += 1
        else:
            answer[year] = 1

    answer = sorted(answer.items(), key=operator.itemgetter(1), reverse=True)

    with open(results_path("q20.csv"), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, quoting=csv.QUOTE_MINIMAL)

        row = ["Year", "Movies"]
        writer.writerow(row)
        writer.writerows(answer[:top])

    csvFile.close()
# Main.
extract_secondary_dataset()

actor_network = create_actor_network()
genre_network = create_genre_network()
movie_network = create_movie_network()

print()
print("Starting execution...")
print()

#print('Generating actor graph.')
#save_actor_graph_as_pdf(actor_network)

#print("Generating genre network.")
#save_genre_graph_as_pdf(create_genre_network())

#print("Generating movie network.")
#save_movie_graph_as_pdf(create_movie_network())

#q1(actor_network, 20)
#q2(actor_network)
#q3(actor_network)
#q4(actor_network)
#q5(actor_network)
#q6(actor_network, genre_network, movie_network)
#q7(actor_network, genre_network, movie_network)
#q8(actor_network, genre_network, movie_network)
#q9(actor_network, genre_network, movie_network)
#q10(actor_network, genre_network, movie_network)
#q11(actor_network, genre_network, movie_network)
#q12(actor_network, genre_network, movie_network)
#q13(actor_network)
#q14(actor_network)
#q15(genre_network)
#q16()
#q17()
#q18()
#q19(4)
#q20(10)

print()
print("End")
print()

# write_gexf(G, path, encoding='utf-8', prettyprint=True, version='1.1draft')[source]

duration = 1000  # millisecond
freq = 440  # Hz
winsound.Beep(freq, duration)